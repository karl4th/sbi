from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReasoningConfig:
    vocab_size: int = 512
    max_seq_len: int = 256
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
    # Hidden state dimension exposed to fingerprint layer
    hidden_dim: int = 768


@dataclass
class MemoryConfig:
    fingerprint_dim: int = 128
    max_memory_size: int = 10000
    # Hebbian learning rate
    hebbian_lr: float = 0.01
    # Edge strength decay per step
    decay_rate: float = 0.999
    # Min confidence to store a memory entry
    min_confidence: float = 0.6
    # Cluster size threshold to trigger meta-state compression
    compression_threshold: int = 20
    # How many neighbors to retrieve during search
    top_k: int = 8
    # How many nearest neighbors to connect in the Hebbian graph
    graph_top_k: int = 3
    # Diagnostic bias added to the retrieved answer token at ANSWER positions.
    # Set to 0.0 to disable direct logit injection.
    memory_logit_bias: float = 0.0
    # Token ID of the ANSWER separator in data.babi.tokenizer.
    answer_marker_token_id: int = 4
    # Diagnostic mode: disable learned StateFingerprintLayer and address memory
    # with a deterministic bag-of-tokens fingerprint from input_ids.
    use_learned_fingerprint: bool = True


@dataclass
class SBIConfig:
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    # Number of retrieved experiences injected into reasoning context
    num_injected_experiences: int = 4
