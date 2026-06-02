from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np

from .core.transformer import ReasoningCore
from .core.config import SBIConfig
from .memory.episodic import EpisodicMemory, MemoryEntry
from .memory.hebbian import HebbianMemoryGraph
from .memory.compression import MetaStateCompressor
from .search.fingerprint import StateFingerprintLayer
from .search.search_layer import SearchLayer


class MemoryInjectionLayer(nn.Module):
    """Projects retrieved memory fingerprints into the transformer residual stream."""

    def __init__(self, fingerprint_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(fingerprint_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        # fingerprints: (K, fingerprint_dim) → (K, d_model)
        return self.proj(fingerprints)


class SBISystem(nn.Module):
    """
    Search-Based Intelligence — full system.

    Two-pass memory injection:
      Pass 1 (no_grad): input → transformer → fingerprint  → query memory
      Pass 2 (grad):    input + memory_tokens → transformer → logits → loss

    Memory stores (query_fingerprint, answer_token) so retrieved entries
    carry actual semantic hints, not just past training state snapshots.
    """

    def __init__(self, config: SBIConfig):
        super().__init__()
        self.config = config

        self.reasoning_core = ReasoningCore(config.reasoning)
        self.fingerprint_layer = StateFingerprintLayer(
            hidden_dim=config.reasoning.hidden_dim,
            fingerprint_dim=config.memory.fingerprint_dim,
        )
        self.memory_injection = MemoryInjectionLayer(
            fingerprint_dim=config.memory.fingerprint_dim,
            d_model=config.reasoning.d_model,
        )

        self.episodic_memory = EpisodicMemory(
            fingerprint_dim=config.memory.fingerprint_dim,
            max_size=config.memory.max_memory_size,
        )
        self.hebbian_graph = HebbianMemoryGraph(
            hebbian_lr=config.memory.hebbian_lr,
            decay_rate=config.memory.decay_rate,
        )
        self.compressor = MetaStateCompressor(
            threshold=config.memory.compression_threshold,
        )
        self.search_layer = SearchLayer(
            memory=self.episodic_memory,
            graph=self.hebbian_graph,
            top_k=config.memory.top_k,
        )

        self._step = 0

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Two-pass forward.

        Returns:
            logits:       (B, T, vocab_size)
            query_fp_np:  (B, fingerprint_dim) — fingerprint of the INPUT query,
                          used for storing in memory (not the output state).
        """
        B = input_ids.shape[0]

        # Pass 1 — no grad, get query fingerprint for the current input
        with torch.no_grad():
            _, hidden_nograd = self.reasoning_core(input_ids, return_hidden_state=True)
            query_fp = self.fingerprint_layer(hidden_nograd)
            query_fp_np = query_fp.cpu().numpy()

        # Retrieve memories relevant to this input
        memory_tokens = self._build_memory_tokens(query_fp_np, input_ids.device)
        if memory_tokens is not None:
            memory_tokens = memory_tokens.expand(B, -1, -1)

        # Pass 2 — full forward with memory context, gradients flow here
        logits, _ = self.reasoning_core(
            input_ids,
            memory_tokens=memory_tokens,
            return_hidden_state=True,
        )

        return logits, query_fp_np

    def _build_memory_tokens(
        self,
        query_fp_np: np.ndarray,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve relevant memories and build memory token matrix.

        Injects two types of context:
          - Projected fingerprints: encode 'what the situation looked like'
          - Answer embeddings:      encode 'what the answer was'
        """
        if self.episodic_memory.size() == 0:
            return None

        entries = self.search_layer.search(query_fp_np[0])
        if not entries:
            return None

        self.search_layer.record_coactivation(entries)

        # Fingerprint projections — (K, d_model)
        fp_array = np.stack([e.state_signature for e in entries])
        fp_tensor = torch.tensor(fp_array, dtype=torch.float32, device=device)
        mem_vectors = self.memory_injection(fp_tensor)

        # Answer hint embeddings — inject answer tokens from retrieved memories
        answer_ids = [e.answer_token for e in entries if e.answer_token >= 0]
        if answer_ids:
            ans_tensor = torch.tensor(answer_ids, dtype=torch.long, device=device)
            # Reuse the transformer's own token embedding — same semantic space
            ans_embs = self.reasoning_core.token_emb(ans_tensor)   # (num_hints, d_model)
            mem_vectors = torch.cat([mem_vectors, ans_embs], dim=0)

        return mem_vectors.unsqueeze(0)   # (1, K + num_hints, d_model)

    def remember(
        self,
        query_fp_np: np.ndarray,
        answer_token: int,
        confidence: float,
    ):
        """
        Store a (query_fingerprint, answer_token) experience in episodic memory.
        query_fp_np is the INPUT fingerprint — makes retrieval coherent with querying.
        """
        if confidence < self.config.memory.min_confidence:
            return

        entry = MemoryEntry(
            state_signature=query_fp_np[0],
            action="reasoning",
            outcome="correct" if confidence > 0.8 else "partial",
            confidence=confidence,
            answer_token=answer_token,
        )
        self.episodic_memory.write(entry)

    def retrieve(self, query_fp_np: np.ndarray) -> List[MemoryEntry]:
        entries = self.search_layer.search(query_fp_np[0])
        self.search_layer.record_coactivation(entries)
        return entries

    def step_housekeeping(self):
        self._step += 1
        self.hebbian_graph.decay()
        if self._step % 50 == 0:
            self.compressor.maybe_compress(self.episodic_memory)

    def memory_stats(self) -> dict:
        return {
            "memory_size": self.episodic_memory.size(),
            "graph_edges": self.hebbian_graph.num_edges(),
            "step": self._step,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
