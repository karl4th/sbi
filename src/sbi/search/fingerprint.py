import torch
import torch.nn as nn
import numpy as np


class StateFingerprintLayer(nn.Module):
    """
    Converts a hidden reasoning state into a compact, stable fingerprint.
    This fingerprint serves as the address space of the memory system.

    The layer is trained jointly with the Reasoning Core so that
    similar reasoning situations map to similar fingerprints.
    """

    def __init__(self, hidden_dim: int = 768, fingerprint_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fingerprint_dim = fingerprint_dim

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, fingerprint_dim),
            nn.LayerNorm(fingerprint_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (B, hidden_dim) mean-pooled hidden state
        Returns:
            fingerprint: (B, fingerprint_dim) L2-normalized fingerprint
        """
        fp = self.proj(hidden_state)
        # L2 normalize so cosine similarity = dot product (matches FAISS IndexFlatIP)
        return nn.functional.normalize(fp, p=2, dim=-1)

    def to_numpy(self, hidden_state: torch.Tensor) -> np.ndarray:
        """Convenience: run forward and detach to numpy for memory storage."""
        with torch.no_grad():
            fp = self.forward(hidden_state)
        return fp.cpu().numpy()
