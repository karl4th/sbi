from typing import List, Tuple
import numpy as np
from ..memory.episodic import EpisodicMemory, MemoryEntry
from ..memory.hebbian import HebbianMemoryGraph


class SearchLayer:
    """
    Full search procedure over reasoning states.

    Steps:
    1. Similarity search — find nearest memory entries by fingerprint
    2. Graph expansion   — follow Hebbian edges to find related entries
    3. Rank and filter   — return best candidate experiences
    """

    def __init__(self, memory: EpisodicMemory, graph: HebbianMemoryGraph, top_k: int = 8):
        self.memory = memory
        self.graph = graph
        self.top_k = top_k

    def search(self, query_signature: np.ndarray) -> List[MemoryEntry]:
        """
        Retrieve relevant experiences for the current reasoning state.
        Combines similarity search with graph-based expansion.
        """
        # Step 1: direct similarity search
        direct_hits = self.memory.search(query_signature, top_k=self.top_k)
        if not direct_hits:
            return []

        seed_ids = [e.entry_id for e in direct_hits]

        # Step 2: graph expansion from seed nodes
        expanded_ids = self.graph.expand(seed_ids, depth=2, top_n=4)

        # Step 3: fetch expanded entries
        id_to_entry = {e.entry_id: e for e in self.memory.entries}
        expanded_entries = [id_to_entry[i] for i in expanded_ids if i in id_to_entry]

        # Step 4: merge and rank by confidence * usage
        all_entries = direct_hits + expanded_entries
        seen = set()
        unique: List[MemoryEntry] = []
        for e in all_entries:
            if e.entry_id not in seen:
                seen.add(e.entry_id)
                unique.append(e)

        unique.sort(key=lambda e: e.confidence * (1 + 0.1 * e.usage_count), reverse=True)
        return unique[: self.top_k]

    def format_experiences(self, entries: List[MemoryEntry]) -> str:
        """
        Render retrieved experiences as text to be injected into the
        reasoning context as additional token input.
        """
        if not entries:
            return ""
        lines = ["[Memory]"]
        for e in entries:
            lines.append(f"  situation: {e.action} -> outcome: {e.outcome} (conf={e.confidence:.2f})")
        lines.append("[/Memory]")
        return "\n".join(lines)

    def record_coactivation(self, entries: List[MemoryEntry]):
        """Update Hebbian graph based on the entries that were co-retrieved."""
        ids = [e.entry_id for e in entries if e.entry_id >= 0]
        if len(ids) > 1:
            self.graph.update(ids)
