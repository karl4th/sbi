"""
PyTorch Dataset for bAbI.

Training objective: given the full story + question, predict the answer word.
Loss is computed only on answer tokens (everything before ANSWER token is masked).
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset

from .loader import load_babi_tasks, flatten_example
from .tokenizer import BabiTokenizer, ANSWER_ID, PAD_ID


def build_tokenizer(examples: List[Dict]) -> BabiTokenizer:
    """Build and freeze a vocabulary from a list of bAbI examples."""
    tokenizer = BabiTokenizer()
    all_texts = []
    for ex in examples:
        input_text, answer = flatten_example(ex)
        all_texts.append(input_text)
        all_texts.append(answer)
    tokenizer.build_vocab(all_texts)
    tokenizer.freeze()
    return tokenizer


class BabiDataset(Dataset):
    """
    Args:
        task_ids:   which bAbI tasks to include
        split:      "train" or "test"
        tokenizer:  pass a pre-built tokenizer (e.g. built on train split)
        max_seq_len: max token length
        data_dir: optional local Kaggle dataset root
        allow_synthetic_fallback: whether to use generated examples if real
            Kaggle bAbI files are unavailable
    """

    def __init__(
        self,
        task_ids: List[int] = [1, 2, 3],
        split: str = "train",
        tokenizer: Optional[BabiTokenizer] = None,
        max_seq_len: int = 256,
        size_per_task: Optional[int] = None,
        data_dir: Optional[str] = None,
        allow_synthetic_fallback: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.examples = load_babi_tasks(
            task_ids,
            split=split,
            size_per_task=size_per_task,
            data_dir=data_dir,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )

        if tokenizer is None:
            self.tokenizer = build_tokenizer(self.examples)
        else:
            self.tokenizer = tokenizer

        self._encoded = [self._encode(ex) for ex in self.examples]

    def _encode(self, example: Dict) -> Tuple[List[int], List[int]]:
        input_text, answer = flatten_example(example)
        full_text = input_text + " " + answer
        ids = self.tokenizer.encode(full_text, max_len=self.max_seq_len)
        ids = self.tokenizer.pad(ids, self.max_seq_len)

        input_ids = ids[:-1]
        target_ids = ids[1:]

        # Mask loss on everything before and including ANSWER token
        # so the model is only trained to predict the answer word(s)
        target_ids = self._mask_before_answer(input_ids, target_ids)

        return input_ids, target_ids

    def _mask_before_answer(
        self, input_ids: List[int], target_ids: List[int]
    ) -> List[int]:
        """
        Mask target positions up to (but NOT including) the ANSWER token.
        Position i where input[i]=ANSWER has target[i]=answer_word — that is
        exactly what we want the model to predict, so we must NOT mask it.
        """
        masked = list(target_ids)
        found_answer = False
        for i, tok in enumerate(input_ids):
            if tok == ANSWER_ID:
                found_answer = True
                # Position i onward: keep as-is (answer word + EOS)
                break
            masked[i] = PAD_ID
        if not found_answer:
            masked = [PAD_ID] * len(masked)
        return masked

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self._encoded[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
