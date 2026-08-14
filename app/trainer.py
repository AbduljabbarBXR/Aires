"""The child's first real trainer: a char-level language model in pure Python.

Real training: gradients, weights, loss — no notebook, no shortcuts.
Small enough to train on a phone in one night (no numpy required).
Vocabulary: 2000+ words + 12k relationship sentences.

Neuroplasticity knobs (all explicit):
- lr: per-epoch schedule (warm up, hold, decay)
- epochs: spaced repetition (Ebbinghaus) — every epoch replays the corpus
- hidden: capacity; embed: shared letter layer (letters reused across words)
"""

import json
import logging
import math
import random
import sys
import time
from pathlib import Path

from config import VAULT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("trainer")

rng = random.Random(7)

# ---- char vocab ----
CHARS = "abcdefghijklmnopqrstuvwxyz .,!?"
C2I = {c: i for i, c in enumerate(CHARS)}
V = len(CHARS)
E_DIM = 16
HID = 64

CUR = VAULT_DIR / "curriculum"
BRAIN_DIR = VAULT_DIR / "brain"
BRAIN_DIR.mkdir(parents=True, exist_ok=True)


def clean(s: str) -> str:
    return "".join(c for c in s.lower() if c in C2I)


def load_corpus() -> list:
    seqs = []
    sentences = (CUR / "sentences.txt").read_text().splitlines()
    words = (CUR / "words_2000.txt").read_text().splitlines()
    seqs += [clean(s) + "." for s in sentences if clean(s)]
    seqs += [w for w in words if w and all(c in C2I for c in w)]
    return [s for s in seqs if len(s) >= 2]


class CharLM:
    """embed (V->E) + hidden (E->H) + out (H->V), tanh hidden, softmax out."""

    def __init__(self):
        self.embed = [[rng.uniform(-0.1, 0.1) for _ in range(E_DIM)] for _ in range(V)]
        self.W_in = [[rng.uniform(-0.1, 0.1) for _ in range(HID)] for _ in range(E_DIM)]
        self.b_in = [0.0] * HID
        self.W_out = [[rng.uniform(-0.1, 0.1) for _ in range(V)] for _ in range(HID)]
        self.b_out = [0.0] * V

    def forward(self, x_emb, params=None):
        # h = tanh(x@W_in + b_in); logits = h@W_out + b_out
        W_in, b_in, W_out, b_out = params if params else (self.W_in, self.b_in, self.W_out, self.b_out)
        h = [0.0] * HID
        for j in range(HID):
            s = b_in[j]
            for k in range(E_DIM):
                s += x_emb[k] * W_in[k][j]
            h[j] = math.tanh(s)
        logits = [0.0] * V
        for o in range(V):
            s = b_out[o]
            for j in range(HID):
                s += h[j] * W_out[j][o]
            logits[o] = s
        return h, logits

    def train_step(self, seq: str, lr: float) -> float:
        """One sequence, full sequence backprop-free (per-char local SGD)."""
        # buffers for the true gradient of each char step
        g_embed = [[0.0] * E_DIM for _ in range(V)]
        g_W_in = [[0.0] * HID for _ in range(E_DIM)]
        g_b_in = [0.0] * HID
        g_W_out = [[0.0] * V for _ in range(HID)]
        g_b_out = [0.0] * V
        total = 0.0
        n = 0
        for i in range(len(seq) - 1):
            ci, cn = C2I[seq[i]], C2I[seq[i + 1]]
            x_emb = self.embed[ci]
            h, logits = self.forward(x_emb)
            # softmax + cross-entropy
            m = max(logits)
            ex = [math.exp(l - m) for l in logits]
            s = sum(ex)
            probs = [e / s for e in ex]
            loss = -math.log(probs[cn] + 1e-12)
            total += loss
            n += 1
            # dL/dlogits = probs - onehot
            for o in range(V):
                d = probs[o] - (1.0 if o == cn else 0.0)
                g_b_out[o] += d
                for j in range(HID):
                    g_W_out[j][o] += d * h[j]
            # dL/dh
            dh = [0.0] * HID
            for j in range(HID):
                for o in range(V):
                    dh[j] += (probs[o] - (1.0 if o == cn else 0.0)) * self.W_out[j][o]
            # h = tanh(pre); dh/dpre = (1 - h^2)
            for j in range(HID):
                pre_g = dh[j] * (1.0 - h[j] * h[j])
                g_b_in[j] += pre_g
                for k in range(E_DIM):
                    g_W_in[k][j] += pre_g * x_emb[k]
                for k in range(E_DIM):
                    g_embed[ci][k] += pre_g * self.W_in[k][j]
        # apply
        for ci in range(V):
            for k in range(E_DIM):
                self.embed[ci][k] -= lr * g_embed[ci][k]
        for k in range(E_DIM):
            for j in range(HID):
                self.W_in[k][j] -= lr * g_W_in[k][j]
        for j in range(HID):
            self.b_in[j] -= lr * g_b_in[j]
            for o in range(V):
                self.W_out[j][o] -= lr * g_W_out[j][o]
        for o in range(V):
            self.b_out[o] -= lr * g_b_out[o]
        return total / max(n, 1)

    def next_probs(self, seq: str) -> list:
        x_emb = self.embed[C2I[seq[-1]]]
        h, logits = self.forward(x_emb)
        m = max(logits)
        ex = [math.exp(l - m) for l in logits]
        s = sum(ex)
        return [e / s for e in ex]

    def spell_check(self, words: list) -> tuple:
        """(strict, top3) next-char accuracy over all words."""
        ok = ok3 = total = 0
        for w in words:
            w = clean(w)
            if len(w) < 2:
                continue
            for i in range(len(w) - 1):
                probs = self.next_probs(w[:i + 1])
                order = sorted(range(V), key=lambda o: probs[o], reverse=True)
                target = C2I[w[i + 1]]
                total += 1
                if order[0] == target:
                    ok += 1
                if target in order[:3]:
                    ok3 += 1
        return ok / max(total, 1), ok3 / max(total, 1)

    def sample(self, prefix: str, length: int = 60, temp: float = 0.8) -> str:
        out = prefix
        cur = clean(prefix)[-1] or "t"
        for _ in range(length):
            probs = self.next_probs(cur)
            ps = [p ** (1.0 / temp) for p in probs]
            s = sum(ps)
            ps = [p / s for p in ps]
            r = rng.random()
            acc = 0.0
            for i, p in enumerate(ps):
                acc += p
                if r <= acc:
                    c = CHARS[i]
                    break
            else:
                c = " "
            out += c
            cur = c
        return out

    def save(self, path: Path):
        state = {"embed": self.embed, "W_in": self.W_in, "b_in": self.b_in,
                 "W_out": self.W_out, "b_out": self.b_out, "chars": CHARS}
        path.write_text(json.dumps(state))

    def load(self, path: Path):
        d = json.loads(path.read_text())
        self.embed, self.W_in = d["embed"], d["W_in"]
        self.b_in, self.W_out, self.b_out = d["b_in"], d["W_out"], d["b_out"]


def lr_schedule(epoch: int) -> float:
    if epoch < 2:
        return 0.02
    if epoch < 5:
        return 0.01
    return 0.005


def main():
    smoke = "--smoke" in sys.argv
    if "--single" in sys.argv:
        target = int(sys.argv[sys.argv.index("--single") + 1])
    else:
        target = 0

    corpus = load_corpus()
    if smoke:
        corpus = corpus[:200]
    words = (CUR / "words_2000.txt").read_text().splitlines()
    log.info("corpus: %d sequences, %.1fk chars | words: %d", len(corpus), sum(len(s) for s in corpus) / 1000, len(words))

    ckpt = BRAIN_DIR / "model.json"
    model = CharLM()
    start = 1
    if ckpt.exists() and (smoke or target > 1):
        model.load(ckpt)
        start = max(1, target)
        log.info("resumed from checkpoint, continuing at epoch %d", start)

    report = []
    ep_range = [target] if target else range(1, 9)
    for ep in ep_range:
        if target and ep != target:
            continue
        lr = lr_schedule(ep)
        rng.shuffle(corpus)
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for s in corpus:
            loss_sum += model.train_step(s, lr)
            n += 1
        loss = loss_sum / n
        acc, acc3 = model.spell_check(words[:300] if smoke else words)
        sample = model.sample("the dog ", 60)
        line = f"epoch {ep}/20 lr={lr} loss={loss:.4f} spell_acc={acc*100:.1f}% top3={acc3*100:.1f}% ({time.time()-t0:.0f}s)"
        log.info(line)
        report.append(line)
        log.info("sample: %s", sample)
        model.save(ckpt)
        log.info("checkpoint saved")
    (BRAIN_DIR / "morning_report.txt").write_text("\n".join(report))
    log.info("DONE. report -> %s", BRAIN_DIR / "morning_report.txt")


if __name__ == "__main__":
    main()