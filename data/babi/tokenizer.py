"""
Simple word-level tokenizer for bAbI.

bAbI has a small, closed vocabulary (~200 unique words across all tasks).
Word-level tokenization is cleaner than character-level here.
"""

import re
from typing import List, Dict


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
ANSWER_ID = 4   # special separator token before the answer
VOCAB_OFFSET = 5


def _tokenize(text: str) -> List[str]:
    """Split into word tokens, lowercase everything except the ANSWER sentinel."""
    tokens = re.findall(r"[a-zA-Z]+|[?.!,]", text)
    return ["ANSWER" if t == "ANSWER" else t.lower() for t in tokens]


class BabiTokenizer:
    """
    Build vocabulary from dataset, then encode/decode sequences.
    """

    def __init__(self):
        self.word2id: Dict[str, int] = {
            "<pad>": PAD_ID,
            "<bos>": BOS_ID,
            "<eos>": EOS_ID,
            "<unk>": UNK_ID,
            "ANSWER": ANSWER_ID,
        }
        self.id2word: Dict[int, str] = {v: k for k, v in self.word2id.items()}
        self._frozen = False

    def build_vocab(self, texts: List[str]):
        """Scan all texts and add new words to vocabulary."""
        assert not self._frozen, "Tokenizer is frozen — cannot add new words."
        for text in texts:
            for token in _tokenize(text):
                if token not in self.word2id:
                    idx = len(self.word2id)
                    self.word2id[token] = idx
                    self.id2word[idx] = token

    def freeze(self):
        self._frozen = True

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        tokens = _tokenize(text)
        ids = [BOS_ID] + [self.word2id.get(t, UNK_ID) for t in tokens] + [EOS_ID]
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            words.append(self.id2word.get(i, "<unk>"))
        return " ".join(words)

    def pad(self, ids: List[int], max_len: int) -> List[int]:
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)
