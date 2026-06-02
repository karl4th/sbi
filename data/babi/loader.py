"""
Load bAbI tasks directly from Facebook Research servers.

Downloads the official tar.gz archive and parses the raw text format.
No HuggingFace loading scripts needed.

Raw format:
    1 Mary moved to the bathroom.
    2 John went to the hallway.
    3 Where is Mary? \tbathroom\t1
    ...
Line number resets to 1 at the start of each new story.
"""

import os
import re
import tarfile
import urllib.request
from typing import List, Dict, Tuple

BABI_URL = "https://dl.fbaipublicfiles.com/babi/tasks_1-20_v1-2.tar.gz"
BABI_DIR = os.path.join(os.path.dirname(__file__), ".cache")
ARCHIVE_PATH = os.path.join(BABI_DIR, "babi.tar.gz")
DATA_ROOT = os.path.join(BABI_DIR, "tasks_1-20_v1-2", "en-10k")

TASK_FILES = {
    1:  "qa1_single-supporting-fact",
    2:  "qa2_two-supporting-facts",
    3:  "qa3_three-supporting-facts",
    15: "qa15_basic-deduction",
    16: "qa16_basic-induction",
}


def _download():
    os.makedirs(BABI_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE_PATH):
        print("Downloading bAbI dataset (~11 MB)...")
        urllib.request.urlretrieve(BABI_URL, ARCHIVE_PATH)
        print("Download complete.")
    if not os.path.exists(DATA_ROOT):
        print("Extracting...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(BABI_DIR)
        print("Extraction complete.")


def _parse_file(path: str, task_id: int) -> List[Dict]:
    """Parse one bAbI task file into a list of examples."""
    examples = []
    story_sentences = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split off the line number
            parts = line.split(" ", 1)
            line_num = int(parts[0])
            rest = parts[1]

            # New story starts when line number resets to 1
            if line_num == 1:
                story_sentences = []

            if "\t" in rest:
                # Question line: "question\tanswer\tsupporting_ids"
                question_part, answer, *_ = rest.split("\t")
                question = question_part.strip()
                answer = answer.strip()

                if story_sentences and question and answer:
                    examples.append({
                        "story": list(story_sentences),
                        "question": question,
                        "answer": answer,
                        "task_id": task_id,
                    })
            else:
                # Statement line
                story_sentences.append(rest.strip())

    return examples


def load_babi_tasks(
    task_ids: List[int] = [1, 2, 3],
    split: str = "train",
    size_per_task: int = None,
) -> List[Dict]:
    """
    Load bAbI tasks. Downloads data on first call.

    Args:
        task_ids:      which tasks to load (1, 2, 3, 15, 16)
        split:         "train" or "test"
        size_per_task: cap examples per task (None = use all)

    Returns:
        list of {"story": [...], "question": "...", "answer": "...", "task_id": int}
    """
    _download()

    examples = []
    for task_id in task_ids:
        filename = f"{TASK_FILES[task_id]}_{split}.txt"
        path = os.path.join(DATA_ROOT, filename)
        task_examples = _parse_file(path, task_id)
        if size_per_task:
            task_examples = task_examples[:size_per_task]
        examples.extend(task_examples)

    return examples


def flatten_example(example: Dict) -> Tuple[str, str]:
    """
    Convert a bAbI example to (input_text, answer) strings.

    Input:  "sent1 sent2 ... question ANSWER"
    Target: "answer_word"
    """
    story_text = " ".join(example["story"])
    input_text = story_text + " " + example["question"] + " ANSWER"
    return input_text, example["answer"]
