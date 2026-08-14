import logging
import time

from config import VAULT_DIR
from curriculum import Curriculum
from notebook import Notebook
from teacher import Teacher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("lessons")

NOTEBOOK_PATH = VAULT_DIR / "notebook.json"


def teach_letters(t: Teacher, c: Curriculum) -> None:
    batch = c.next_letters()
    for letter in batch:
        t.teach_alphabet_letter(letter)
        log.info("taught letter %s", letter)
    if batch:
        c.mark_letters(batch)
        log.info("alphabet progress: %s", c.progress()["letters"])


def teach_words(t: Teacher, c: Curriculum) -> None:
    chunk = c.next_chunk()
    if not chunk:
        return
    known = set(c.notebook.words())
    todo = [w for w in chunk if w not in known]
    if todo:
        report = t.teach_lesson(todo)
        log.info("word chunk: %s -> taught %s failed %s", todo, report["taught"], report["failed"])
    done = all(w in set(c.notebook.words()) for w in chunk)
    if done:
        c.mark_chunk(len(c.state["chunks_done"]))
        log.info("chunk complete. progress: %s", c.progress())


def main() -> None:
    n = Notebook(NOTEBOOK_PATH)
    t = Teacher(n)
    c = Curriculum(n)
    log.info("lesson runner started. stage=%s", c.state["stage"])
    while True:
        if c.state["stage"] == 1:
            teach_letters(t, c)
            c = Curriculum(n)
        else:
            teach_words(t, c)
            c = Curriculum(n)
        log.info("current progress: %s", c.progress())
        time.sleep(30)


if __name__ == "__main__":
    main()