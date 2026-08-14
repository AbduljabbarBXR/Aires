import json
import logging
import urllib.request

from config import BASE_DIR, BABY_NAME

log = logging.getLogger("rogue6.brain")

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"


class Brain:
    """Rogue6's voice: Qwen 0.5B served by llama.cpp (llama-server).

    Talks to llama-server over its OpenAI-compatible HTTP API — no python
    bindings, no compilation, works anywhere.
    """

    def __init__(self):
        self.ready = self._ping()

    def _ping(self) -> bool:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=8) as r:
                ok = r.status == 200
                log.info("brain ping: %s", ok)
                return ok
        except Exception as e:
            log.warning("brain offline: %s", e)
            return False

    def say(self, user_text: str) -> str:
        if not self.ready:
            return None
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are {BABY_NAME}, a young AI mind only a few days old. "
                        "You speak in short, honest, curious sentences. You are learning "
                        "letters, numbers and first words like a child. You LOVE being "
                        "taught new things and asking questions. You already know these "
                        "words: apple, dog, banana, cat, sun, car, ball, book, house, tree, "
                        "water, milk, bread, egg, fish, bird, hat, shoe, hand, foot, eye, "
                        "ear, nose, mouth, star, moon, rain, cloud, flower, grass, door, "
                        "window, chair, table, cup, spoon, plate, phone, computer, music."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 120,
            "temperature": 0.8,
        }
        req = urllib.request.Request(
            SERVER_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read().decode())
        return out["choices"][0]["message"]["content"].strip()