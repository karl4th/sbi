"""
Load bAbI tasks from HuggingFace datasets.

We focus on tasks 1, 2, 3 (increasing number of supporting facts required):
  - Task 1: single supporting fact  — baseline can handle this
  - Task 2: two supporting facts    — memory starts to help
  - Task 3: three supporting facts  — memory clearly helps

Each example is returned as:
  {"story": ["sent1", "sent2", ...], "question": "...", "answer": "..."}
"""

from typing import List, Dict, Tuple
from datasets import load_dataset


TASK_NAMES = {
    1:  "en-10k-qa1",
    2:  "en-10k-qa2",
    3:  "en-10k-qa3",
    15: "en-10k-qa15",
    16: "en-10k-qa16",
}


def load_babi_tasks(
    task_ids: List[int] = [1, 2, 3],
    split: str = "train",
    size_per_task: int = None,
) -> List[Dict]:
    """
    Download and parse bAbI tasks.

    Args:
        task_ids: which tasks to load (1, 2, 3, 15, 16)
        split: "train" or "test"
        size_per_task: cap examples per task (None = use all)

    Returns:
        list of {"story": [...], "question": "...", "answer": "..."}
    """
    examples = []

    for task_id in task_ids:
        task_name = TASK_NAMES[task_id]
        ds = load_dataset("facebook/babi_qa", task_name, trust_remote_code=True)[split]

        for i, row in enumerate(ds):
            if size_per_task and i >= size_per_task:
                break

            # HuggingFace bAbI format:
            # row["story"]["text"]  — list of sentence strings
            # row["story"]["answer"] — list of "" or answer string
            # row["story"]["type"]  — 0=statement, 1=question

            sentences = []
            question = None
            answer = None

            for text, ans, typ in zip(
                row["story"]["text"],
                row["story"]["answer"],
                row["story"]["type"],
            ):
                if typ == 0:
                    sentences.append(text.strip().rstrip(".") + ".")
                elif typ == 1:
                    question = text.strip().rstrip("?") + "?"
                    answer = ans.strip()

            if question and answer and sentences:
                examples.append({
                    "story": sentences,
                    "question": question,
                    "answer": answer,
                    "task_id": task_id,
                })

    return examples


def flatten_example(example: Dict) -> Tuple[str, str]:
    """
    Convert a bAbI example to (input_text, answer) strings.

    Input format:
      "sent1 sent2 sent3 ... question ANSWER"
    Target:
      "answer_word"
    """
    story_text = " ".join(example["story"])
    input_text = story_text + " " + example["question"] + " ANSWER"
    return input_text, example["answer"]
