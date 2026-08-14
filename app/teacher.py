import json
import re
import time
from pathlib import Path

import requests
from gtts import gTTS

from config import PHOTO_DIR, VAULT_DIR, BABY_NAME
from notebook import Notebook

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = f"{BABY_NAME}-teacher/0.1 (parent-run learning project)"


class Teacher:
    """Searches the world for pictures and sounds, and turns them into lessons.

    Lessons are grounded: every word is stored with real photos and real audio,
    the way a child learns from its environment.
    """

    def __init__(self, notebook: Notebook):
        self.notebook = notebook
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.audio_dir = VAULT_DIR / "audio"
        self.lesson_dir = PHOTO_DIR / "lessons"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.lesson_dir.mkdir(parents=True, exist_ok=True)

    def find_images(self, word: str, count: int = 3) -> list:
        """Find real photos of `word` on Wikimedia Commons."""
        queries = [
            f"{word} photo",
            f"{word} -icon -emoji -diagram -symbol -logo -drawing",
        ]
        urls = []
        for query in queries:
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": count * 8,
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": 320,
                "format": "json",
            }
            r = self.session.get(COMMONS_API, params=params, timeout=30)
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                if len(urls) >= count:
                    return urls
                info = page.get("imageinfo", [{}])[0]
                thumb = info.get("thumburl")
                if not thumb:
                    continue
                if not re.search(r"\.(jpe?g|png)(\?|$)", thumb, re.I):
                    continue
                if re.search(r"\.(svg|tif|gif)(\?|$)", info.get("url", ""), re.I):
                    continue
                urls.append(thumb)
            if len(urls) >= count:
                break
        return urls

    def download_image(self, url: str, dest: Path) -> bool:
        try:
            r = self.session.get(url, timeout=45)
            if r.status_code == 200 and len(r.content) > 2000:
                dest.write_bytes(r.content)
                return True
        except requests.RequestException:
            pass
        return False

    def make_audio(self, word: str) -> Path:
        dest = self.audio_dir / f"{word}.mp3"
        if not dest.exists():
            gTTS(text=word, lang="en", slow=False).save(str(dest))
        return dest

    def teach_word(self, word: str, images_per_word: int = 3) -> dict:
        """Teach one word: fetch photos, generate audio, store the lesson."""
        images = self.find_images(word, images_per_word)
        saved = []
        for i, url in enumerate(images, 1):
            dest = self.lesson_dir / f"{word}_{i:02d}.jpg"
            if self.download_image(url, dest):
                saved.append(str(dest))
        audio = None
        if saved:
            audio = str(self.make_audio(word))
        if not saved:
            return {"word": word, "ok": False, "reason": "no usable images found"}

        self.notebook.add_word(word, media_path=saved[0], source="teacher")
        self.notebook.add_lesson(
            "word_grounding",
            {"word": word, "images": saved, "audio": audio, "taught_at": time.time()},
        )
        return {"word": word, "ok": True, "images": len(saved), "audio": bool(audio)}

    def teach_alphabet_letter(self, letter: str) -> dict:
        """Teach one letter: the letter's sound, spoken audio, written form."""
        prompt = f"{letter} says {letter}"
        audio = str(self.make_audio(f"the letter {letter}")) if self.audio_dir.exists() else None
        self.notebook.add_lesson(
            "letter_sound",
            {"letter": letter, "audio": audio, "taught_at": time.time()},
        )
        return {"letter": letter, "audio": bool(audio)}

    def teach_lesson(self, words: list, images_per_word: int = 3) -> dict:
        results = [self.teach_word(w, images_per_word) for w in words]
        ok = [r for r in results if r.get("ok")]
        failed = [r["word"] for r in results if not r.get("ok")]
        return {"taught": len(ok), "words": [r["word"] for r in ok], "failed": failed}


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = Notebook(VAULT_DIR / "notebook.json")
    t = Teacher(n)
    first_words = ["apple", "dog", "banana", "cat", "sun", "car", "ball", "book"]
    known = set(n.words())
    for word in first_words:
        if word in known:
            print(f"[{time.strftime('%H:%M:%S')}] skip {word} (already learned)", flush=True)
            continue
        result = t.teach_word(word)
        print(f"[{time.strftime('%H:%M:%S')}] {word}: {result}", flush=True)
    print("DONE. notebook stats:", n.stats(), flush=True)