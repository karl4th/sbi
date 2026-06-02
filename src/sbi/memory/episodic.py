from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import faiss
import time


@dataclass
class MemoryEntry:
    state_signature: np.ndarray   # fingerprint of the INPUT query state
    action: str                    # what reasoning step was taken
    outcome: str                   # result of that step
    confidence: float              # how confident the system was
    answer_token: int = -1         # token ID of the correct answer (used for hint injection)
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

    def search(
        self,
        query_signature: np.ndarray,
        top_k: int = 8,
        min_similarity: float = 0.5,
    ) -> List[MemoryEntry]:
        """
        Retrieve top_k most similar memory entries above min_similarity threshold.
        Returns [] when no entry is similar enough — caller must handle this.
        """
        if len(self.entries) == 0:
            return []

        k = min(top_k, len(self.entries))
        vec = self._normalize(query_signature).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_similarity:
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
