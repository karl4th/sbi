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
        memory_tokens, memory_answer_tokens = self._build_memory_tokens(
            query_fp_np, input_ids.device
        )

        # Pass 2 — full forward with memory context, gradients flow here
        logits, _ = self.reasoning_core(
            input_ids,
            memory_tokens=memory_tokens,
            return_hidden_state=True,
        )
        logits = self._apply_memory_logit_bias(
            input_ids, logits, memory_answer_tokens
        )

        return logits, query_fp_np

    def _build_memory_tokens(
        self,
        query_fp_np: np.ndarray,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve relevant memories and build memory token matrix.

        Injects two types of context:
          - Projected fingerprints: encode 'what the situation looked like'
          - Answer embeddings:      encode 'what the answer was'
        """
        if self.episodic_memory.size() == 0:
            return None, None

        memory_vectors = []
        answer_tokens = []
        any_hit = False

        for query_fp in query_fp_np:
            # Retrieve several neighbors so Hebbian co-activation can form edges.
            # Only the best entry is injected below to keep the hint low-noise.
            entries = self.episodic_memory.search(
                query_fp,
                top_k=self.config.memory.top_k,
                min_similarity=0.5,
            )
            if not entries or entries[0].answer_token < 0:
                memory_vectors.append(
                    torch.zeros(
                        self.config.reasoning.d_model,
                        dtype=self.reasoning_core.token_emb.weight.dtype,
                        device=device,
                    )
                )
                answer_tokens.append(-1)
                continue

            any_hit = True
            entry = entries[0]
            answer_tokens.append(entry.answer_token)

            # Hebbian update only on confident retrievals (not every step)
            if entry.confidence > 0.75:
                self.search_layer.record_coactivation(
                    entries[: self.config.memory.graph_top_k]
                )

            ans_tensor = torch.tensor([entry.answer_token], dtype=torch.long, device=device)
            memory_vectors.append(self.reasoning_core.token_emb(ans_tensor).squeeze(0))

        if not any_hit:
            return None, None

        return (
            torch.stack(memory_vectors, dim=0).unsqueeze(1),   # (B, 1, d_model)
            torch.tensor(answer_tokens, dtype=torch.long, device=device),
        )

    def retrieve_answer_tokens(
        self,
        query_fp_np: np.ndarray,
        top_k: int = 1,
        min_similarity: float = 0.5,
    ) -> Tuple[List[int], List[float]]:
        """
        Diagnostic retrieval: return best answer token and cosine score per query.

        This does not mutate usage counts or the Hebbian graph, so it is safe for
        eval logging.
        """
        if self.episodic_memory.size() == 0:
            return [-1] * len(query_fp_np), [0.0] * len(query_fp_np)

        answer_tokens: List[int] = []
        similarities: List[float] = []
        for query_fp in query_fp_np:
            entries, scores = self.episodic_memory.search_with_scores(
                query_fp,
                top_k=top_k,
                min_similarity=min_similarity,
                update_usage=False,
            )
            if not entries:
                answer_tokens.append(-1)
                similarities.append(0.0)
                continue
            answer_tokens.append(entries[0].answer_token)
            similarities.append(scores[0])
        return answer_tokens, similarities

    def _apply_memory_logit_bias(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        memory_answer_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Directly bias logits toward retrieved memory answers at ANSWER positions."""
        bias = self.config.memory.memory_logit_bias
        if bias <= 0 or memory_answer_tokens is None:
            return logits

        answer_positions = input_ids == self.config.memory.answer_marker_token_id
        if not answer_positions.any():
            return logits

        biased_logits = logits.clone()
        for batch_idx, answer_token in enumerate(memory_answer_tokens.tolist()):
            if answer_token < 0:
                continue
            positions = answer_positions[batch_idx].nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue
            biased_logits[batch_idx, positions, answer_token] += bias
        return biased_logits

    def _input_fingerprint(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Deterministic diagnostic address for memory.

        This bypasses the learned fingerprint layer so memory behavior can be
        tested independently. It hashes ordered token features into
        fingerprint_dim bins and L2-normalizes the vector for cosine search.
        """
        B = input_ids.shape[0]
        F = self.config.memory.fingerprint_dim
        dtype = self.reasoning_core.token_emb.weight.dtype
        tokens = input_ids.to(torch.long)
        mask = (tokens != 0).to(dtype=dtype)
        fp = torch.zeros(B, F, dtype=dtype, device=input_ids.device)

        # Unigram identity with mild position signal.
        pos = torch.arange(tokens.shape[1], device=input_ids.device).unsqueeze(0)
        pos_weight = (1.0 + pos.to(dtype=dtype) / max(1, tokens.shape[1])).expand_as(mask)
        self._scatter_hashed_features(fp, tokens + 31 * pos, mask * pos_weight)

        # Ordered n-grams separate stories that contain the same words in
        # different reasoning chains.
        if tokens.shape[1] >= 2:
            bigram_hash = tokens[:, :-1] * 131 + tokens[:, 1:] * 17 + pos[:, :-1] * 7
            bigram_mask = mask[:, :-1] * mask[:, 1:]
            self._scatter_hashed_features(fp, bigram_hash, 0.75 * bigram_mask)

        if tokens.shape[1] >= 3:
            trigram_hash = (
                tokens[:, :-2] * 8191
                + tokens[:, 1:-1] * 131
                + tokens[:, 2:] * 17
                + pos[:, :-2] * 13
            )
            trigram_mask = mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]
            self._scatter_hashed_features(fp, trigram_hash, 0.5 * trigram_mask)

        return nn.functional.normalize(fp, p=2, dim=-1)

    def _scatter_hashed_features(
        self,
        fp: torch.Tensor,
        feature_hash: torch.Tensor,
        weights: torch.Tensor,
    ):
        bins = torch.remainder(feature_hash, fp.shape[1])
        signs = torch.where(
            torch.remainder(feature_hash // fp.shape[1], 2) == 0,
            torch.ones_like(weights),
            -torch.ones_like(weights),
        )
        fp.scatter_add_(1, bins, weights * signs)

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
