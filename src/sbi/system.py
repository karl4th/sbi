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
    """
    Projects retrieved memory fingerprints (fingerprint_dim)
    into the transformer's residual stream (d_model) so they can
    be prepended as context tokens.
    """

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

    Memory injection flow (lag-1):
      step t-1: forward → get fingerprint_t-1
      step t  : retrieve using fingerprint_t-1
               → project retrieved fingerprints → memory_tokens
               → prepend memory_tokens to input
               → forward with memory context
               → get fingerprint_t (for step t+1)
    """

    def __init__(self, config: SBIConfig):
        super().__init__()
        self.config = config

        # Trainable components (gradient descent)
        self.reasoning_core = ReasoningCore(config.reasoning)
        self.fingerprint_layer = StateFingerprintLayer(
            hidden_dim=config.reasoning.hidden_dim,
            fingerprint_dim=config.memory.fingerprint_dim,
        )
        self.memory_injection = MemoryInjectionLayer(
            fingerprint_dim=config.memory.fingerprint_dim,
            d_model=config.reasoning.d_model,
        )

        # Memory components (Hebbian updates, no backprop)
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
        prev_fingerprint: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Args:
            input_ids:        (B, T)
            prev_fingerprint: (B, fingerprint_dim) numpy — fingerprint from
                              the previous step used to retrieve memory context.
                              Pass None at the very first step.

        Returns:
            logits:      (B, T, vocab_size)
            fingerprint: (B, fingerprint_dim) numpy — use as prev_fingerprint next step
        """
        memory_tokens = self._build_memory_tokens(prev_fingerprint, input_ids.device)

        logits, hidden = self.reasoning_core(
            input_ids,
            memory_tokens=memory_tokens,
            return_hidden_state=True,
        )
        fingerprint = self.fingerprint_layer(hidden)
        fp_numpy = fingerprint.detach().cpu().numpy()
        return logits, fp_numpy

    def _build_memory_tokens(
        self,
        prev_fingerprint: Optional[np.ndarray],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Retrieve entries and project to d_model. Returns None if memory is empty."""
        if prev_fingerprint is None or self.episodic_memory.size() == 0:
            return None

        entries = self.search_layer.search(prev_fingerprint[0])
        if not entries:
            return None

        self.search_layer.record_coactivation(entries)

        fp_array = np.stack([e.state_signature for e in entries])          # (K, fp_dim)
        fp_tensor = torch.tensor(fp_array, dtype=torch.float32, device=device)
        mem_vectors = self.memory_injection(fp_tensor)                      # (K, d_model)
        return mem_vectors.unsqueeze(0)                                     # (1, K, d_model)

    def retrieve(self, fingerprint: np.ndarray) -> List[MemoryEntry]:
        entries = self.search_layer.search(fingerprint[0])
        self.search_layer.record_coactivation(entries)
        return entries

    def remember(
        self,
        fingerprint: np.ndarray,
        action: str,
        outcome: str,
        confidence: float,
    ):
        if confidence < self.config.memory.min_confidence:
            return
        entry = MemoryEntry(
            state_signature=fingerprint[0],
            action=action,
            outcome=outcome,
            confidence=confidence,
        )
        self.episodic_memory.write(entry)

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
