import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
BABY_NAME = os.environ.get("BABY_NAME", "Rogue6")
TEACHER_TYPE = os.environ.get("TEACHER_TYPE", "")
TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY", "")

VAULT_DIR = BASE_DIR / "vault"
PHOTO_DIR = VAULT_DIR / "photos"
NOTEBOOK_PATH = VAULT_DIR / "notebook.json"

PHOTO_DIR.mkdir(parents=True, exist_ok=True)
