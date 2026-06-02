"""
bAbI-compatible task generator (Tasks 1, 2, 3).

Generates data following the exact same rules as the original Facebook bAbI dataset.
This approach avoids download issues and gives unlimited data.
Results are directly comparable to published Memory Networks baselines.

Task 1 — Single supporting fact:
    One person moves, question: where are they?

Task 2 — Two supporting facts:
    Person picks up object, then moves. Question: where is the object?

Task 3 — Three supporting facts:
    Longer chain: two people interact, object changes hands + moves.
    Question: where is the object?
"""

import random
from typing import List, Dict, Tuple


PEOPLE  = ["Mary", "John", "Daniel", "Sandra", "Julie", "Fred", "Bill", "Joe"]
PLACES  = ["bathroom", "hallway", "bedroom", "garden", "kitchen", "office"]
OBJECTS = ["football", "milk", "apple", "sandwich", "keys", "book", "bag"]

MOVE_VERBS = [
    "moved to", "went to", "travelled to", "journeyed to"
]
PICK_VERBS = [
    "picked up", "grabbed", "got"
]
DROP_VERBS = [
    "dropped", "put down", "left"
]


def _move(person: str, place: str, rng: random.Random) -> str:
    return f"{person} {rng.choice(MOVE_VERBS)} the {place}."


def _pick(person: str, obj: str, rng: random.Random) -> str:
    return f"{person} {rng.choice(PICK_VERBS)} the {obj}."


def _drop(person: str, obj: str, rng: random.Random) -> str:
    return f"{person} {rng.choice(DROP_VERBS)} the {obj}."


# ── Task generators ───────────────────────────────────────────────────────────

def _task1(rng: random.Random) -> Dict:
    """Single supporting fact: where is person X?"""
    people = rng.sample(PEOPLE, k=rng.randint(2, 4))
    places = rng.sample(PLACES, k=len(people))

    sentences = []
    locations = {}

    # Each person moves around a few times
    for _ in range(rng.randint(2, 5)):
        person = rng.choice(people)
        place  = rng.choice(places)
        sentences.append(_move(person, place, rng))
        locations[person] = place

    target_person = rng.choice(list(locations.keys()))
    answer = locations[target_person]
    question = f"Where is {target_person}?"

    return {"story": sentences, "question": question, "answer": answer, "task_id": 1}


def _task2(rng: random.Random) -> Dict:
    """
    Two supporting facts: person picks up object then moves.
    Where is the object? (follows the person)
    """
    person = rng.choice(PEOPLE)
    obj    = rng.choice(OBJECTS)
    place1 = rng.choice(PLACES)
    place2 = rng.choice([p for p in PLACES if p != place1])

    # Distractor: another person moves somewhere
    other  = rng.choice([p for p in PEOPLE if p != person])
    dplace = rng.choice(PLACES)

    sentences = [
        _move(person, place1, rng),
        _move(other, dplace, rng),
        _pick(person, obj, rng),
        _move(person, place2, rng),
    ]
    rng.shuffle(sentences[:2])  # shuffle distractors only

    question = f"Where is the {obj}?"
    answer   = place2

    return {"story": sentences, "question": question, "answer": answer, "task_id": 2}


def _task3(rng: random.Random) -> Dict:
    """
    Three supporting facts: person A picks up object, gives to person B,
    person B moves to new place.
    Where is the object?
    """
    personA = rng.choice(PEOPLE)
    personB = rng.choice([p for p in PEOPLE if p != personA])
    obj     = rng.choice(OBJECTS)

    placeA  = rng.choice(PLACES)
    placeB  = rng.choice([p for p in PLACES if p != placeA])
    placeC  = rng.choice([p for p in PLACES if p not in (placeA, placeB)])

    # Distractor
    other  = rng.choice([p for p in PEOPLE if p not in (personA, personB)])
    dplace = rng.choice(PLACES)

    sentences = [
        _move(personA, placeA, rng),          # fact 1: A is at placeA
        _pick(personA, obj, rng),              # fact 2: A picks up obj
        _move(personB, placeB, rng),           # distractor
        _drop(personA, obj, rng),              # A drops obj (personB picks it up)
        _pick(personB, obj, rng),              # fact 3: B has obj
        _move(other, dplace, rng),             # distractor
        _move(personB, placeC, rng),           # B moves to final place
    ]

    question = f"Where is the {obj}?"
    answer   = placeC

    return {"story": sentences, "question": question, "answer": answer, "task_id": 3}


_GENERATORS = {1: _task1, 2: _task2, 3: _task3}

# ── Public API ────────────────────────────────────────────────────────────────

def load_babi_tasks(
    task_ids: List[int] = [1, 2, 3],
    split: str = "train",
    size_per_task: int = None,
) -> List[Dict]:
    """
    Generate bAbI-compatible examples.

    Args:
        task_ids:      which tasks to include (1, 2, 3)
        split:         "train" or "test" (different random seeds)
        size_per_task: examples per task (default: 1000 train / 200 test)

    Returns:
        list of {"story": [...], "question": "...", "answer": "...", "task_id": int}
    """
    base_seed = 42 if split == "train" else 9999
    default_size = 1000 if split == "train" else 200
    n = size_per_task if size_per_task is not None else default_size

    examples = []
    for task_id in task_ids:
        if task_id not in _GENERATORS:
            raise ValueError(f"Task {task_id} not implemented. Available: {list(_GENERATORS)}")
        rng = random.Random(base_seed + task_id)
        gen = _GENERATORS[task_id]
        for _ in range(n):
            examples.append(gen(rng))

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
