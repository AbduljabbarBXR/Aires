"""Train the child's own brain on the words it has actually been taught."""

import json
import logging
import time
from pathlib import Path

from brain_column import ColumnBrain, CATEGORY
from config import VAULT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train")

WORD_POOL = [w for vocab in CATEGORY.values() for w in vocab]
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def load_taught_words() -> list:
    nb = VAULT_DIR / "notebook.json"
    if not nb.exists():
        return []
    data = json.loads(nb.read_text())
    return sorted(set(w["word"].strip().lower() for w in data["words"]))


def main() -> None:
    brain = ColumnBrain()
    taught = load_taught_words()
    log.info("taught words from notebook: %d %s", len(taught), taught)
    # lessons = alphabet + taught words that are in our known vocabulary
    corpus = ALPHABET + [w for w in taught if w in WORD_POOL]
    log.info("training corpus: %d sequences", len(corpus))

    for word in corpus:
        t0 = time.time()
        loss = brain.learn(word, epochs=25)
        log.info("learned %-12s loss=%.4f (%.2fs)", word, loss, time.time() - t0)

    acc = brain.accuracy(corpus)
    log.info("next-letter accuracy on trained words: %.1f%%", acc * 100)
    brain.save()
    log.info("brain saved to %s", brain.state_dir / "brain.json")

    # babbling demo
    for prefix in ("a", "do", "ca", "b"):
        preds = brain.predict(prefix, top=3)
        log.info("babble '%s' -> %s", prefix, preds)


if __name__ == "__main__":
    main()