"""
Synthetic reasoning task generator.

Generates three task types that require reasoning, not memorized knowledge:
  1. LogicChain   — multi-step deductive reasoning
  2. Analogy      — structural pattern completion
  3. PlanSequence — reach goal state through minimal operations
"""

import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class TaskType(Enum):
    LOGIC_CHAIN = "logic_chain"
    ANALOGY = "analogy"
    PLAN_SEQUENCE = "plan_sequence"


@dataclass
class Task:
    task_type: TaskType
    input_text: str
    target_text: str


# ── Vocabulary ────────────────────────────────────────────────────────────────

SYMBOLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")

# Simple character-level tokenizer (all printable ASCII + special tokens)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
VOCAB_OFFSET = 4  # first real character id

PRINTABLE = [chr(i) for i in range(32, 127)]
CHAR_TO_ID = {c: i + VOCAB_OFFSET for i, c in enumerate(PRINTABLE)}
ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
VOCAB_SIZE = VOCAB_OFFSET + len(PRINTABLE)  # ~100 tokens


def encode(text: str, max_len: int = 256) -> List[int]:
    ids = [BOS_ID] + [CHAR_TO_ID.get(c, UNK_ID) for c in text] + [EOS_ID]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids


def decode(ids: List[int]) -> str:
    chars = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in ID_TO_CHAR:
            chars.append(ID_TO_CHAR[i])
    return "".join(chars)


# ── Task generators ───────────────────────────────────────────────────────────

def _gen_logic_chain(chain_length: int = 3) -> Task:
    """
    Example (chain=3):
      "All A are B. All B are C. All C are D. X is A. Is X D?"
      Answer: "Yes"
    """
    symbols = random.sample(SYMBOLS, chain_length + 2)
    chain = symbols[: chain_length + 1]
    subject = symbols[chain_length + 1]

    premises = " ".join(f"All {chain[i]} are {chain[i+1]}." for i in range(chain_length))
    question = f"{subject} is {chain[0]}. Is {subject} {chain[-1]}?"

    input_text = premises + " " + question
    target_text = "Yes"
    return Task(TaskType.LOGIC_CHAIN, input_text, target_text)


def _gen_analogy() -> Task:
    """
    Example:
      "A:B as C:?"   where A→B follows pattern +1 in alphabet
      Answer: "D"
    """
    idx = random.randint(0, 22)
    a, b = SYMBOLS[idx], SYMBOLS[idx + 1]
    c_idx = random.randint(0, 22)
    c = SYMBOLS[c_idx]
    d = SYMBOLS[c_idx + 1]

    input_text = f"{a}:{b} as {c}:?"
    target_text = d
    return Task(TaskType.ANALOGY, input_text, target_text)


def _gen_plan_sequence(n_steps: int = 3) -> Task:
    """
    Example (n=2):
      "Start:3 Goal:5 Ops:[+1,-1,*2] Steps:2"
      Answer: "+1 +1"
    """
    start = random.randint(0, 10)
    ops = []
    val = start
    op_names = []
    for _ in range(n_steps):
        op = random.choice(["+1", "-1"])
        op_names.append(op)
        val = val + 1 if op == "+1" else val - 1

    goal = val
    input_text = f"Start:{start} Goal:{goal} Ops:[+1,-1] Steps:{n_steps}"
    target_text = " ".join(op_names)
    return Task(TaskType.PLAN_SEQUENCE, input_text, target_text)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticReasoningDataset(Dataset):
    """
    Generates synthetic reasoning tasks on-the-fly.
    Each item is a (input_ids, target_ids) pair.
    """

    def __init__(
        self,
        size: int = 10000,
        max_seq_len: int = 256,
        task_mix: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        seed: int = 42,
    ):
        self.size = size
        self.max_seq_len = max_seq_len
        self.task_mix = task_mix

        rng = random.Random(seed)
        self.tasks = []
        for _ in range(size):
            roll = rng.random()
            if roll < task_mix[0]:
                self.tasks.append(_gen_logic_chain(chain_length=rng.randint(2, 4)))
            elif roll < task_mix[0] + task_mix[1]:
                self.tasks.append(_gen_analogy())
            else:
                self.tasks.append(_gen_plan_sequence(n_steps=rng.randint(2, 4)))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        task = self.tasks[idx]
        full_text = task.input_text + " -> " + task.target_text
        ids = encode(full_text, self.max_seq_len)

        # Pad to max_seq_len
        ids += [PAD_ID] * (self.max_seq_len - len(ids))
        ids = ids[: self.max_seq_len]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, target_ids
