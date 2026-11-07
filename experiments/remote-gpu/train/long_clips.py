"""Runs ON Kaggle (shipped with the training data): long clips cut at sentence ends.

PARROT's median piece is 22.7 s and 83% are over max_duration_sec, so v4-v7 trained on 1 in 6 of its
reports. The stock model times each word; a clip is cut just before a sentence's first word, wherever
that word and the one before it are both heard exactly as written (a cut elsewhere could split a word).
Each piece is decoded again and kept only if it starts and ends on its own text's first and last word.
Whole recorded consultations (Fareez, 12 min) are timed in window_sec windows, which the GPU holds.
"""
import difflib
import re

from sources import SR
from text_rules import normalise

TALLY = ("long", "pieces", "edge_mismatch", "no_cut_fits", "slip")


def heard_words(res) -> list:
    """(normalised word, start second) from a timed result; a piece starting with a space starts a word."""
    words = []
    for tok, t in zip(res.tokens or [], res.timestamps or []):
        if tok.startswith((" ", "▁")) or not words:
            words.append([tok, t])
        else:
            words[-1][0] += tok
    return [(w, t) for tok, t in words for w in normalise(tok).split()[:1]]


class Cutter:
    """`load` makes the timed stock model; it is loaded on first need and dropped by free()."""

    def __init__(self, load, max_sec: float, lead_sec: float, window_sec: float):
        self._load, self._model = load, None
        self.max_sec, self.lead_sec, self.window_sec = max_sec, lead_sec, window_sec

    def free(self) -> None:
        """Give the aligner's GPU memory back once every source is cut."""
        self._model = None

    def _recognize(self, a):
        if self._model is None:
            self._model = self._load()
        return self._model.recognize(a, sample_rate=SR)

    def _heard(self, a) -> list:
        """(word, start second) over the whole clip, timed window by window; a word split by a window
        edge is misheard there, so no cut is made at it."""
        step = int(self.window_sec * SR)
        return [(w, t + i / SR) for i in range(0, len(a), step)
                for w, t in heard_words(self._recognize(a[i:i + step]))]

    def split(self, items, tally: dict, slips=None):
        """Clips within max_sec pass through; longer ones are cut at sentence ends. Counts go in `tally`.
        With `slips` (talk.Slips), a piece whose text has a transcriber's slip is dropped."""
        for k in TALLY:
            tally.setdefault(k, 0)
        for a, text in items:
            if len(a) / SR <= self.max_sec:
                pieces = [(a, text, None)]
            else:
                tally["long"] += 1
                pieces = self._cut(a, text, tally)
            for piece, words, got in pieces:
                hear = (lambda g=got: g) if got is not None else (lambda p=piece: normalise(self._recognize(p).text).split())
                if slips is not None and slips(words, hear):
                    tally["slip"] += 1
                else:
                    yield piece, words

    def _starts(self, a, sents: list) -> dict:
        """{sentence k: second a piece may start at}; sentence 0 at the start of the clip."""
        ref, first = [], []
        for x in sents:
            first.append(len(ref))
            ref += normalise(x).split()
        heard = self._heard(a)
        match = {}
        for b in difflib.SequenceMatcher(None, ref, [w for w, _ in heard], autojunk=False).get_matching_blocks():
            match.update({b.a + j: b.b + j for j in range(b.size)})
        return {0: 0.0, **{k: max(0.0, heard[match[i]][1] - self.lead_sec) for k, i in enumerate(first)
                           if k and i in match and i - 1 in match}}

    def _cut(self, a, text: str, tally: dict):
        sents = [x for x in re.split(r"(?<=[.;:?!])\s+", text) if normalise(x)]
        starts = self._starts(a, sents)
        end = len(a) / SR
        bounds = sorted(starts) + [len(sents)]
        k0 = 0
        while k0 < len(sents):
            # furthest boundary that keeps the piece within max_sec
            fits = [k for k in bounds if k > k0 and starts.get(k, end) - starts[k0] <= self.max_sec]
            if not fits:
                tally["no_cut_fits"] += 1
                nxt = [k for k in bounds if k > k0 and k in starts]
                if not nxt:
                    return
                k0 = nxt[0]
                continue
            k1 = fits[-1]
            piece, words = a[int(starts[k0] * SR): int(starts.get(k1, end) * SR)], " ".join(sents[k0:k1])
            got, want = normalise(self._recognize(piece).text).split(), normalise(words).split()
            if got and got[0] == want[0] and got[-1] == want[-1]:
                tally["pieces"] += 1
                yield piece, words, got
            else:
                tally["edge_mismatch"] += 1
            if k1 not in starts:
                return
            k0 = k1
