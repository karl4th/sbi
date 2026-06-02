from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from .episodic import EpisodicMemory, MemoryEntry


class MetaStateCompressor:
    """
    Prevents memory clutter by merging clusters of frequently co-occurring
    states into a single MetaState node.

    Compression rule: if a cluster of states repeatedly appears together,
    replace them with one representative MetaState.
    """

    def __init__(self, threshold: int = 20, min_cluster_size: int = 4):
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
        self._step_count = 0

    def maybe_compress(self, memory: EpisodicMemory) -> int:
        """
        Check if compression is needed and run it.
        Returns number of entries removed.
        """
        self._step_count += 1
        if memory.size() < self.threshold:
            return 0
        return self._compress(memory)

    def _compress(self, memory: EpisodicMemory) -> int:
        entries = memory.entries
        if len(entries) < self.min_cluster_size:
            return 0

        signatures = np.array([e.state_signature for e in entries], dtype=np.float32)

        n_clusters = max(1, len(entries) // self.min_cluster_size)
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        labels = kmeans.fit_predict(signatures)

        removed = 0
        new_entries: List[MemoryEntry] = []

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_entries = [e for e, m in zip(entries, cluster_mask) if m]

            if len(cluster_entries) < self.min_cluster_size:
                new_entries.extend(cluster_entries)
                continue

            # Merge cluster into one MetaState entry
            meta = self._merge(cluster_entries)
            new_entries.append(meta)
            removed += len(cluster_entries) - 1

        # Rebuild memory from compressed entries
        memory.index.reset()
        memory.entries = []

        for entry in new_entries:
            vec = memory._normalize(entry.state_signature)
            memory.index.add(vec.reshape(1, -1).astype(np.float32))
            memory.entries.append(entry)

        return removed

    def _merge(self, entries: List[MemoryEntry]) -> MemoryEntry:
        """Create a single MetaState from a cluster of entries."""
        from collections import Counter

        signatures = np.stack([e.state_signature for e in entries])
        centroid = signatures.mean(axis=0)

        best = max(entries, key=lambda e: e.confidence * (1 + e.usage_count))

        # Majority vote on answer_token — most common answer in the cluster wins.
        # Without this, answer_token is lost after compression and hints stop working.
        answer_votes = Counter(
            e.answer_token for e in entries if e.answer_token >= 0
        )
        dominant_answer = answer_votes.most_common(1)[0][0] if answer_votes else -1

        return MemoryEntry(
            state_signature=centroid,
            action=f"[MetaState] {best.action}",
            outcome=best.outcome,
            confidence=float(np.mean([e.confidence for e in entries])),
            usage_count=sum(e.usage_count for e in entries),
            entry_id=best.entry_id,
            answer_token=dominant_answer,
        )
