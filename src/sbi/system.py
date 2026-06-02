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
        if config.memory.use_learned_fingerprint:
            self.fingerprint_layer = StateFingerprintLayer(
                hidden_dim=config.reasoning.hidden_dim,
                fingerprint_dim=config.memory.fingerprint_dim,
            )
        else:
            self.fingerprint_layer = None
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
        # Pass 1 — no grad, get query fingerprint for the current input
        with torch.no_grad():
            if self.fingerprint_layer is None:
                query_fp = self._input_fingerprint(input_ids)
            else:
                _, hidden_nograd = self.reasoning_core(input_ids, return_hidden_state=True)
                query_fp = self.fingerprint_layer(hidden_nograd)
            query_fp_np = query_fp.cpu().numpy()

        # Retrieve memories relevant to each input in the batch.
        memory_tokens = self._build_memory_tokens(query_fp_np, input_ids.device)

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

        memory_vectors = []
        any_hit = False

        for query_fp in query_fp_np:
            # top_k=1: only the single most similar entry.
            # More hints = more noise when fingerprints are imperfect.
            entries = self.episodic_memory.search(
                query_fp, top_k=1, min_similarity=0.5
            )
            if not entries or entries[0].answer_token < 0:
                memory_vectors.append(
                    torch.zeros(
                        self.config.reasoning.d_model,
                        dtype=self.reasoning_core.token_emb.weight.dtype,
                        device=device,
                    )
                )
                continue

            any_hit = True
            entry = entries[0]

            # Hebbian update only on confident retrievals (not every step)
            if entry.confidence > 0.75:
                self.search_layer.record_coactivation(entries)

            ans_tensor = torch.tensor([entry.answer_token], dtype=torch.long, device=device)
            memory_vectors.append(self.reasoning_core.token_emb(ans_tensor).squeeze(0))

        if not any_hit:
            return None

        return torch.stack(memory_vectors, dim=0).unsqueeze(1)   # (B, 1, d_model)

    def _input_fingerprint(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Deterministic diagnostic address for memory.

        This bypasses the learned fingerprint layer so memory behavior can be
        tested independently. It hashes token IDs into fingerprint_dim bins and
        L2-normalizes the bag-of-tokens vector for cosine search.
        """
        B = input_ids.shape[0]
        F = self.config.memory.fingerprint_dim
        mask = (input_ids != 0).to(dtype=self.reasoning_core.token_emb.weight.dtype)
        bins = torch.remainder(input_ids, F)
        fp = torch.zeros(B, F, dtype=mask.dtype, device=input_ids.device)
        fp.scatter_add_(1, bins, mask)
        return nn.functional.normalize(fp, p=2, dim=-1)

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
        state_signature = query_fp_np if query_fp_np.ndim == 1 else query_fp_np[0]

        entry = MemoryEntry(
            state_signature=state_signature,
            action="reasoning",
            outcome="correct" if confidence > 0.8 else "partial",
            confidence=confidence,
            answer_token=answer_token,
        )
        self.episodic_memory.write(entry)

    def remember_batch(
        self,
        query_fp_np: np.ndarray,
        answer_tokens: List[int],
        confidences: Optional[List[float]] = None,
    ):
        """Store one supervised memory entry per batch item."""
        if confidences is None:
            confidences = [1.0] * len(answer_tokens)

        for query_fp, answer_token, confidence in zip(
            query_fp_np, answer_tokens, confidences
        ):
            if answer_token < 0:
                continue
            self.remember(query_fp, answer_token, confidence)

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
