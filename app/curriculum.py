import json
import time
from pathlib import Path

from config import VAULT_DIR
from notebook import Notebook

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")

FIRST_WORDS = [
    "apple", "dog", "banana", "cat", "sun", "car", "ball", "book",
    "house", "tree", "water", "milk", "bread", "egg", "fish", "bird",
    "hat", "shoe", "hand", "foot", "eye", "ear", "nose", "mouth",
    "star", "moon", "rain", "cloud", "flower", "grass", "door", "window",
    "chair", "table", "cup", "spoon", "plate", "phone", "computer", "music",
]

WORD_CHUNKS = [FIRST_WORDS[i:i + 10] for i in range(0, len(FIRST_WORDS), 10)]


class Curriculum:
    """Sequential lesson plan: alphabet first, then words in chunks of 10."""

    def __init__(self, notebook: Notebook):
        self.notebook = notebook
        self.state_path = VAULT_DIR / "curriculum.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"stage": 1, "alphabet_done": [], "chunks_done": [], "started_at": time.time()}

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def next_letters(self, batch: int = 5) -> list:
        return [c for c in ALPHABET if c not in self.state["alphabet_done"]][:batch]

    def mark_letters(self, letters: list) -> None:
        for c in letters:
            if c not in self.state["alphabet_done"]:
                self.state["alphabet_done"].append(c)
        if len(self.state["alphabet_done"]) >= len(ALPHABET):
            self.state["stage"] = 2
        self.save()

    def next_chunk(self) -> list:
        for i, chunk in enumerate(WORD_CHUNKS):
            if i not in self.state["chunks_done"]:
                return chunk
        return []

    def mark_chunk(self, chunk_index: int) -> None:
        if chunk_index not in self.state["chunks_done"]:
            self.state["chunks_done"].append(chunk_index)
        self.save()

    def progress(self) -> dict:
        done = self.state["chunks_done"]
        learned = min((max(done) + 1) * 10, len(FIRST_WORDS)) if done else 0
        return {
            "stage": self.state["stage"],
            "letters": f"{len(self.state['alphabet_done'])}/26",
            "chunks": f"{len(done)}/{len(WORD_CHUNKS)}",
            "words": learned,
        }