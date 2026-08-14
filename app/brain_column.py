"""The child's own brain: sparse specialist columns with a shared learning curve.

Design ideas (brain-inspired, numpy-only, no heavy deps):
- Columns: small specialist predictors (animals, food, objects, general).
  Only the column matching the current topic activates (sparse).
- Shared layer: ONE common letter-embedding matrix every column reads and
  writes to — what one column learns improves all (shared learning curve).
- Local learning: each column trains its own weights only, with its own
  learning-rate schedule (the "neuroplasticity" knobs we control).
- Predicts next-letter of words it was taught (babbling), so it learns the
  SHAPE of language before meaning.

Latent/state: activations are per-column H (tiny), shared embeddings E.
"""

import json
import time
from pathlib import Path

import numpy as np

from config import VAULT_DIR

SEED = 42
rng = np.random.default_rng(SEED)

CHARS = "abcdefghijklmnopqrstuvwxyz "  # 26 letters + space
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
C_EMB = 32  # shared embedding size
HID = 32   # per-column hidden size

CATEGORY = {
    "animals": {w for w in "dog cat fish bird elephant lion zebra monkey cow sheep horse".split()},
    "food": {w for w in "apple banana bread egg milk rice meat soup cake cheese water juice".split()},
    "objects": {w for w in "car ball book house tree door window chair table cup spoon plate phone computer shoe hat".split()},
    "general": {w for w in "sun moon rain cloud flower grass hand foot eye ear nose mouth star music happy sad big small".split()},
}
COLUMNS = list(CATEGORY.keys())


class ColumnBrain:
    def __init__(self, state_dir=None):
        self.state_dir = Path(state_dir or VAULT_DIR / "brain")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # shared layer
        self.E = rng.normal(0, 0.1, (VOCAB, C_EMB)).astype(np.float32)
        # each column: W_in (C_EMB -> HID), b_h, W_out (HID -> VOCAB)
        self.cols = {}
        for col in COLUMNS:
            self.cols[col] = {
                "W_in": rng.normal(0, 0.1, (C_EMB, HID)).astype(np.float32),
                "b_h": np.zeros(HID, dtype=np.float32),
                "W_out": rng.normal(0, 0.1, (HID, VOCAB)).astype(np.float32),
                "lr": 0.01,
            }
        self.load()

    def embed(self, char: str) -> np.ndarray:
        return self.E[CHAR_IDX[char]]

    def column_for(self, word: str) -> str:
        for col, vocab in CATEGORY.items():
            if word in vocab:
                return col
        return "general"

    # ---- forward (only one column active: sparse) ----
    def forward(self, col: str, char: str):
        x = self.embed(char)
        p = self.cols[col]
        h = np.tanh(x @ p["W_in"] + p["b_h"])
        logits = h @ p["W_out"]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return h, probs

    # ---- local learning: update ONLY the active column + shared E ----
    def learn(self, seq: str, epochs: int = 40) -> float:
        col = self.column_for(seq)
        p = self.cols[col]
        loss_sum = 0.0
        for _ in range(epochs):
            total = 0.0
            for i in range(len(seq) - 1):
                cur, nxt = seq[i], seq[i + 1]
                target = CHAR_IDX[nxt]
                x = self.embed(cur)
                h = np.tanh(x @ p["W_in"] + p["b_h"])
                logits = h @ p["W_out"]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                loss = -np.log(probs[target] + 1e-9)
                total += loss
                # gradients (local, no backprop through columns)
                d_logits = probs.copy()
                d_logits[target] -= 1.0          # dL/dlogits
                g_W_out = np.outer(h, d_logits)  # (HID, VOCAB)
                g_h = d_logits @ p["W_out"].T     # (HID,)
                g_W_in = np.outer(x, g_h * (1 - h ** 2))
                g_b_h = g_h * (1 - h ** 2)
                g_x = g_h * (1 - h ** 2) @ p["W_in"].T
                p["W_out"] -= p["lr"] * g_W_out
                p["W_in"] -= p["lr"] * g_W_in
                p["b_h"] -= p["lr"] * g_b_h
                # shared layer update (the shared learning curve)
                self.E[CHAR_IDX[cur]] -= p["lr"] * g_x
            loss_sum += total / (len(seq) - 1)
        return loss_sum / epochs

    def predict(self, prefix: str, top: int = 3) -> list:
        """Babble: predict the most likely next letters after prefix."""
        if not prefix:
            return []
        col = self.column_for(prefix + "x")
        h, probs = self.forward(col, prefix[-1])
        order = probs.argsort()[::-1][:top]
        return [(CHARS[i], float(probs[i])) for i in order]

    def accuracy(self, words: list) -> float:
        """How often it predicts the correct NEXT letter (held-out style)."""
        ok = total = 0
        for w in words:
            for i in range(len(w) - 1):
                h, probs = self.forward(self.column_for(w), w[i])
                if CHARS[probs.argmax()] == w[i + 1]:
                    ok += 1
                total += 1
        return ok / max(total, 1)

    # ---- persistence ----
    def save(self):
        state = {"E": self.E.tolist(),
                 "cols": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in self.cols.items()},
                 "saved_at": time.time()}
        (self.state_dir / "brain.json").write_text(json.dumps(state))

    def load(self):
        f = self.state_dir / "brain.json"
        if f.exists():
            d = json.loads(f.read_text())
            self.E = np.array(d["E"], dtype=np.float32)
            for col, w in d["cols"].items():
                for k in ("W_in", "b_h", "W_out"):
                    self.cols[col][k] = np.array(w[k], dtype=np.float32)