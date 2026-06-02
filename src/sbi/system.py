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


class SBISystem(nn.Module):
    """
    Search-Based Intelligence — full system.

    Connects the Reasoning Core, State Fingerprint Layer,
    Episodic Memory, Hebbian Graph, and Search Layer into
    one trainable + updateable system.
    """

    def __init__(self, config: SBIConfig):
        super().__init__()
        self.config = config

        # Trainable components (updated via gradient descent)
        self.reasoning_core = ReasoningCore(config.reasoning)
        self.fingerprint_layer = StateFingerprintLayer(
            hidden_dim=config.reasoning.hidden_dim,
            fingerprint_dim=config.memory.fingerprint_dim,
        )

        # Memory components (updated via Hebbian rules, no backprop)
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
        Forward pass with memory retrieval.

        Returns:
            logits:     (B, T, vocab_size)
            fingerprint: (B, fingerprint_dim) numpy array for memory operations
        """
        logits, hidden = self.reasoning_core(input_ids, return_hidden_state=True)
        fingerprint = self.fingerprint_layer(hidden)
        fp_numpy = fingerprint.detach().cpu().numpy()
        return logits, fp_numpy

    def retrieve(self, fingerprint: np.ndarray) -> List[MemoryEntry]:
        """Search memory given a fingerprint. Called during inference."""
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
        """Write a new experience to episodic memory if confidence is high enough."""
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
        """
        Called once per training step.
        Runs Hebbian decay and periodic memory compression.
        """
        self._step += 1
        self.hebbian_graph.decay()

        if self._step % 50 == 0:
            removed = self.compressor.maybe_compress(self.episodic_memory)
            if removed > 0:
                pass  # compression happened silently

    def memory_stats(self) -> dict:
        return {
            "memory_size": self.episodic_memory.size(),
            "graph_edges": self.hebbian_graph.num_edges(),
            "step": self._step,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
