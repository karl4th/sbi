from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import faiss
import time


@dataclass
class MemoryEntry:
    state_signature: np.ndarray   # fingerprint vector (128-dim)
    action: str                    # what reasoning step was taken
    outcome: str                   # result of that step
    confidence: float              # how confident the system was
    timestamp: float = field(default_factory=time.time)
    usage_count: int = 0
    entry_id: int = -1


class EpisodicMemory:
    """
    Stores reasoning experiences as (state_signature, action, outcome) tuples.
    Uses FAISS for fast approximate nearest-neighbor retrieval.
    """

    def __init__(self, fingerprint_dim: int = 128, max_size: int = 10000):
        self.fingerprint_dim = fingerprint_dim
        self.max_size = max_size

        # FAISS index — inner product on L2-normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(fingerprint_dim)
        self.entries: List[MemoryEntry] = []
        self._next_id = 0

    def write(self, entry: MemoryEntry) -> Optional[int]:
        """Store a memory entry. Returns entry_id or None if rejected."""
        if len(self.entries) >= self.max_size:
            return None

        vec = self._normalize(entry.state_signature)
        entry.entry_id = self._next_id
        self._next_id += 1

        self.index.add(vec.reshape(1, -1).astype(np.float32))
        self.entries.append(entry)
        return entry.entry_id

    def search(self, query_signature: np.ndarray, top_k: int = 8) -> List[MemoryEntry]:
        """Retrieve the top_k most similar memory entries to the query state."""
        if len(self.entries) == 0:
            return []

        k = min(top_k, len(self.entries))
        vec = self._normalize(query_signature).reshape(1, -1).astype(np.float32)
        _, indices = self.index.search(vec, k)

        results = []
        for idx in indices[0]:
            if idx >= 0:
                entry = self.entries[idx]
                entry.usage_count += 1
                results.append(entry)
        return results

    def reinforce(self, entry_id: int, delta: float = 0.05):
        """Boost confidence of a memory that proved useful."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.confidence = min(1.0, entry.confidence + delta)
                break

    def size(self) -> int:
        return len(self.entries)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return vec / norm
