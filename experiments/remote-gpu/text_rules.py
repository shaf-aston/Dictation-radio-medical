"""Text rules shared by the Kaggle kernel and the local scripts, so the two cannot drift.

run_train.py ships this file with the training data; the kernel imports it from there.
Local users: data/audit_train.py (which rows train) and data/build_parrot_eval.py (leak filter).
"""
import re

NUMBER_WORDS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                          r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                          r"eighty|ninety|hundred|thousand|million|percent)\b", re.I)
ODD_CHARS = re.compile(r"[^A-Za-z0-9' ,.;:?!()/%-]")  # markup, other scripts, symbols: drop the row


def usable(text: str) -> bool:
    """A training label Parakeet would write itself: 3+ words, numbers in digits, plain characters."""
    return len(text.split()) >= 3 and not NUMBER_WORDS.search(text) and not ODD_CHARS.search(text)


def sentence_case(t: str) -> str:
    """EdAcc and LibriSpeech are UPPERCASE (EdAcc with <LAUGH>-style tags): strip tags, sentence-case it."""
    t = " ".join(re.sub(r"<[^>]*>", " ", t).split()).lower()
    t = re.sub(r"\bi\b", "I", t)
    return t[:1].upper() + t[1:]


def parrot_text(t: str) -> str:
    """PARROT marks line and paragraph breaks with the words NEWLINE and PARAGRAPH: drop them."""
    return " ".join(re.sub(r"(NEWLINE|PARAGRAPH)", " ", t).split())


FILLERS = re.compile(r"\b(?:m+-?h+m+|u+h-h+u+h+|u+h+m*|u+m+|e+r+m+|a+h+|h+m+)\b[,.]*\s*", re.I)


def talk_text(t: str) -> str:
    """Hand-made transcripts of real talk (PriMock57, Fareez) as Parakeet writes them: no um/uh
    (it does not write them, and a report must not get them), "Okay" not "OK", "Oh" not "Ohh", straight quotes."""
    t = re.sub("[‘’]", "'", t)
    t = re.sub(r"\bOK\b|\bok\b", "okay", FILLERS.sub("", t))
    t = re.sub(r"\b([Oo])hh+\b", r"\1h", t)
    t = re.sub(r"\s+([,.?!])", r"\1", re.sub(r"\.{2,}|[\"“”]", " ", t))
    t = re.sub(r"(^|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), " ".join(t.split()).strip(" ,"))
    return t


def normalise(t: str) -> str:
    """For comparing only: lowercase, no punctuation, so only the words are compared."""
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower()).split())


def sentences(t: str) -> set:
    """The sentences of four words or more, normalised (a decimal point does not end one)."""
    return {s for s in (normalise(x) for x in re.split(r"(?<!\d)[.;:?!](?!\d)", t)) if len(s.split()) >= 4}


def runs(t: str, n: int = 4) -> set:
    """Every run of `n` words in a row, normalised."""
    w = normalise(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def leak_report(train_texts: list, eval_sets: dict) -> dict:
    """How much of each eval set's text ({name: [reference, ...]}) the training text holds.
    Whole phrases should be 0; four-word runs show near-misses."""
    joined = "| " + " | ".join(normalise(t) for t in train_texts) + " |"  # spaces: match whole words only
    train_runs = set().union(*(runs(t) for t in train_texts))
    out = {}
    for name, refs in eval_sets.items():
        phrases = {normalise(t) for t in refs} - {""}
        eval_runs = set().union(*(runs(t) for t in refs))
        found = sorted(p for p in phrases if f" {p} " in joined)
        out[name] = {"phrases": len(phrases), "phrases_in_training": len(found), "found": found[:20],
                     "four_word_runs": len(eval_runs), "four_word_runs_in_training": len(eval_runs & train_runs)}
    return out
