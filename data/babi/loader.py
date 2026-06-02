"""
Loader for the English Facebook bAbI QA tasks from Kaggle.

The primary source is:
    roblexnana/the-babi-tasks-for-nlp-qa-system

If the Kaggle dataset is unavailable, a small synthetic generator is used only
as an explicit fallback so local smoke checks can still run.
"""

from pathlib import Path
import os
import random
from typing import Dict, List, Optional, Tuple


KAGGLE_DATASET = "roblexnana/the-babi-tasks-for-nlp-qa-system"
LANGUAGE_DIR = "en"


PEOPLE = ["Mary", "John", "Daniel", "Sandra", "Julie", "Fred", "Bill", "Joe"]
PLACES = ["bathroom", "hallway", "bedroom", "garden", "kitchen", "office"]
OBJECTS = ["football", "milk", "apple", "sandwich", "keys", "book", "bag"]

MOVE_VERBS = ["moved to", "went to", "travelled to", "journeyed to"]
PICK_VERBS = ["picked up", "grabbed", "got"]
DROP_VERBS = ["dropped", "put down", "left"]


def load_babi_tasks(
    task_ids: List[int] = [1, 2, 3],
    split: str = "train",
    size_per_task: Optional[int] = None,
    data_dir: Optional[str] = None,
    allow_synthetic_fallback: bool = False,
) -> List[Dict]:
    """
    Load real English bAbI examples from Kaggle.

    Args:
        task_ids: bAbI task numbers to load.
        split: "train" or "test".
        size_per_task: optional cap per task.
        data_dir: optional local dataset root. Defaults to BABI_DATA_DIR or
            kagglehub.dataset_download(...).
        allow_synthetic_fallback: use generated task 1/2/3 examples if Kaggle
            data is unavailable.

    Returns:
        list of {"story": [...], "question": "...", "answer": "...", "task_id": int}
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    try:
        root = _resolve_dataset_root(data_dir)
        examples = _load_real_babi(root, task_ids, split, size_per_task)
        print(f"Loaded real Kaggle bAbI ({LANGUAGE_DIR}) from {root}")
        return examples
    except Exception as exc:
        if not allow_synthetic_fallback:
            raise
        print(f"WARNING: real Kaggle bAbI unavailable ({exc}). Using synthetic fallback.")
        return _load_synthetic_babi(task_ids, split, size_per_task)


def _resolve_dataset_root(data_dir: Optional[str]) -> Path:
    explicit_dir = data_dir or os.environ.get("BABI_DATA_DIR")
    if explicit_dir:
        root = Path(explicit_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"BABI_DATA_DIR does not exist: {root}")
        return root

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Install kagglehub or set BABI_DATA_DIR") from exc

    return Path(kagglehub.dataset_download(KAGGLE_DATASET)).resolve()


def _load_real_babi(
    root: Path,
    task_ids: List[int],
    split: str,
    size_per_task: Optional[int],
) -> List[Dict]:
    examples: List[Dict] = []
    for task_id in task_ids:
        task_file = _find_task_file(root, task_id, split)
        task_examples = _parse_babi_file(task_file, task_id)
        if size_per_task is not None:
            task_examples = task_examples[:size_per_task]
        examples.extend(task_examples)

    if not examples:
        raise FileNotFoundError(f"No bAbI examples found under {root}")
    return examples


def _find_task_file(root: Path, task_id: int, split: str) -> Path:
    pattern = f"qa{task_id}_*_{split}.txt"
    candidates = [
        p for p in root.rglob(pattern)
        if p.parent.name == LANGUAGE_DIR or f"/{LANGUAGE_DIR}/" in p.as_posix()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find English bAbI file matching {pattern} under {root}"
        )
    return sorted(candidates, key=lambda p: len(p.as_posix()))[0]


def _parse_babi_file(path: Path, task_id: int) -> List[Dict]:
    examples: List[Dict] = []
    story: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            idx_text, text = line.split(" ", 1)
            if int(idx_text) == 1:
                story = []

            if "\t" in text:
                question, answer, *_supporting = text.split("\t")
                examples.append(
                    {
                        "story": list(story),
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "task_id": task_id,
                    }
                )
            else:
                story.append(text.strip())

    return examples


def flatten_example(example: Dict) -> Tuple[str, str]:
    """
    Convert a bAbI example to (input_text, answer) strings.

    Input: "sent1 sent2 ... question ANSWER"
    Target: "answer_word"
    """
    story_text = " ".join(example["story"])
    input_text = story_text + " " + example["question"] + " ANSWER"
    return input_text.strip(), example["answer"]


def _move(person: str, place: str, rng: random.Random) -> str:
    return f"{person} {rng.choice(MOVE_VERBS)} the {place}."


def _pick(person: str, obj: str, rng: random.Random) -> str:
    return f"{person} {rng.choice(PICK_VERBS)} the {obj}."


def _drop(person: str, obj: str, rng: random.Random) -> str:
    return f"{person} {rng.choice(DROP_VERBS)} the {obj}."


def _task1(rng: random.Random) -> Dict:
    people = rng.sample(PEOPLE, k=rng.randint(2, 4))
    places = rng.sample(PLACES, k=len(people))
    sentences = []
    locations = {}

    for _ in range(rng.randint(2, 5)):
        person = rng.choice(people)
        place = rng.choice(places)
        sentences.append(_move(person, place, rng))
        locations[person] = place

    target_person = rng.choice(list(locations.keys()))
    return {
        "story": sentences,
        "question": f"Where is {target_person}?",
        "answer": locations[target_person],
        "task_id": 1,
    }


def _task2(rng: random.Random) -> Dict:
    person = rng.choice(PEOPLE)
    obj = rng.choice(OBJECTS)
    place1 = rng.choice(PLACES)
    place2 = rng.choice([p for p in PLACES if p != place1])
    other = rng.choice([p for p in PEOPLE if p != person])
    dplace = rng.choice(PLACES)

    return {
        "story": [
            _move(person, place1, rng),
            _move(other, dplace, rng),
            _pick(person, obj, rng),
            _move(person, place2, rng),
        ],
        "question": f"Where is the {obj}?",
        "answer": place2,
        "task_id": 2,
    }


def _task3(rng: random.Random) -> Dict:
    person_a = rng.choice(PEOPLE)
    person_b = rng.choice([p for p in PEOPLE if p != person_a])
    obj = rng.choice(OBJECTS)
    place_a = rng.choice(PLACES)
    place_b = rng.choice([p for p in PLACES if p != place_a])
    place_c = rng.choice([p for p in PLACES if p not in (place_a, place_b)])
    other = rng.choice([p for p in PEOPLE if p not in (person_a, person_b)])
    dplace = rng.choice(PLACES)

    return {
        "story": [
            _move(person_a, place_a, rng),
            _pick(person_a, obj, rng),
            _move(person_b, place_b, rng),
            _drop(person_a, obj, rng),
            _pick(person_b, obj, rng),
            _move(other, dplace, rng),
            _move(person_b, place_c, rng),
        ],
        "question": f"Where is the {obj}?",
        "answer": place_c,
        "task_id": 3,
    }


_GENERATORS = {1: _task1, 2: _task2, 3: _task3}


def _load_synthetic_babi(
    task_ids: List[int],
    split: str,
    size_per_task: Optional[int],
) -> List[Dict]:
    base_seed = 42 if split == "train" else 9999
    default_size = 1000 if split == "train" else 200
    n = size_per_task if size_per_task is not None else default_size

    examples = []
    for task_id in task_ids:
        if task_id not in _GENERATORS:
            raise ValueError(
                f"Synthetic fallback supports tasks {list(_GENERATORS)}, got {task_id}"
            )
        rng = random.Random(base_seed + task_id)
        gen = _GENERATORS[task_id]
        for _ in range(n):
            examples.append(gen(rng))
    return examples
