import json
import time
from pathlib import Path


class Notebook:
    """The child's declarative memory: every word, fact and lesson learned."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"words": [], "facts": [], "lessons": []}

    def _reload(self) -> None:
        """Re-read from disk so multiple processes never clobber each other."""
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self.data = {"words": [], "facts": [], "lessons": []}

    def _add(self, key: str, entry: dict) -> dict:
        self._reload()
        self.data[key].append(entry)
        self._save()
        return entry

    def add_word(self, word: str, media_path: str = None, source: str = "parent") -> dict:
        entry = {
            "word": word.lower().strip(),
            "media": media_path,
            "source": source,
            "learned_at": time.time(),
        }
        return self._add("words", entry)

    def add_fact(self, fact: str, source: str = "parent") -> dict:
        entry = {"fact": fact, "source": source, "learned_at": time.time()}
        return self._add("facts", entry)

    def add_lesson(self, lesson: str, payload: dict) -> dict:
        entry = {"lesson": lesson, "payload": payload, "learned_at": time.time()}
        return self._add("lessons", entry)

    def words(self) -> list:
        return [w["word"] for w in self.data["words"]]

    def stats(self) -> dict:
        return {
            "words": len(self.data["words"]),
            "facts": len(self.data["facts"]),
            "lessons": len(self.data["lessons"]),
        }

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))