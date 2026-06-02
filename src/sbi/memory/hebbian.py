from typing import Dict, List, Tuple
import numpy as np


class HebbianMemoryGraph:
    """
    Graph where nodes are memory entry IDs and edges represent
    co-activation strength between states.

    Hebbian rule: if two states are activated together, strengthen their edge.
    Decay rule:   all edges weaken over time to prevent stale connections.
    """

    def __init__(self, hebbian_lr: float = 0.01, decay_rate: float = 0.999):
        self.hebbian_lr = hebbian_lr
        self.decay_rate = decay_rate
        # edges[i][j] = strength of connection from node i to node j
        self.edges: Dict[int, Dict[int, float]] = {}

    def update(self, activated_ids: List[int]):
        """Strengthen all pairwise edges among co-activated nodes."""
        for i in activated_ids:
            for j in activated_ids:
                if i == j:
                    continue
                if i not in self.edges:
                    self.edges[i] = {}
                self.edges[i][j] = self.edges[i].get(j, 0.0) + self.hebbian_lr

    def decay(self):
        """Apply decay to all edges. Called once per training step."""
        for src in list(self.edges.keys()):
            for dst in list(self.edges[src].keys()):
                self.edges[src][dst] *= self.decay_rate
                if self.edges[src][dst] < 1e-4:
                    del self.edges[src][dst]
            if not self.edges[src]:
                del self.edges[src]

    def expand(self, seed_ids: List[int], depth: int = 2, top_n: int = 4) -> List[int]:
        """
        Graph expansion: starting from seed nodes, follow strongest edges
        to discover related memory nodes not found by similarity search alone.
        """
        visited = set(seed_ids)
        frontier = list(seed_ids)

        for _ in range(depth):
            candidates: Dict[int, float] = {}
            for node in frontier:
                neighbors = self.edges.get(node, {})
                for neighbor, strength in neighbors.items():
                    if neighbor not in visited:
                        candidates[neighbor] = candidates.get(neighbor, 0.0) + strength

            if not candidates:
                break

            # Pick top_n strongest candidates
            top = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
            frontier = [node for node, _ in top]
            visited.update(frontier)

        # Return only the expanded nodes (not the seeds)
        return [n for n in visited if n not in set(seed_ids)]

    def get_strength(self, src: int, dst: int) -> float:
        return self.edges.get(src, {}).get(dst, 0.0)

    def num_edges(self) -> int:
        return sum(len(v) for v in self.edges.values())
