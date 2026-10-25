"""Write many radiology-style dictation lines (no patient data, all invented).

Vocabulary comes from the app's own lexicon so the model hears every term the
correction pipeline claims to protect. Lines equal to a sentence in the test
reports are dropped, so the test set stays unseen.
"""
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEXICON = REPO / "src" / "resources" / "radiology_lexicon.txt"
TEST_CORPUS = REPO / "scripts" / "eval" / "resources" / "report_corpus.txt"

NUM = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
       "twelve", "fourteen", "fifteen", "eighteen", "twenty", "twenty two", "thirty"]
SIDE = ["right", "left", "bilateral"]
REGION = ["upper lobe", "middle lobe", "lower lobe", "basal segment", "apex", "perihilar region",
          "liver", "spleen", "right kidney", "left kidney", "pancreatic head", "sigmoid colon",
          "distal radius", "proximal femur", "lumbar spine", "cervical spine", "left knee",
          "right shoulder", "frontal lobe", "basal ganglia", "cerebellum", "thyroid gland"]
EXAM = ["Chest radiograph.", "Portable chest radiograph, supine.", "Erect chest radiograph.",
        "Computed tomography of the abdomen and pelvis with intravenous contrast.",
        "Magnetic resonance imaging of the lumbar spine without contrast.",
        "Ultrasound of the right upper quadrant.", "Radiograph of the left knee, two views.",
        "Computed tomography of the head without contrast.", "Mammogram, bilateral screening.",
        "Magnetic resonance imaging of the brain with and without gadolinium.",
        "Computed tomography pulmonary angiogram.", "Radiograph of the pelvis, anteroposterior view.",
        "Computed tomography of the chest without contrast.", "Radiograph of the right wrist, three views.",
        "Ultrasound of the pelvis, transabdominal.", "Magnetic resonance imaging of the right knee.",
        "Computed tomography urogram.", "Radiograph of the cervical spine, lateral view.",
        "Doppler ultrasound of the lower extremity veins.", "Fluoroscopic barium swallow study.",
        "Computed tomography of the neck with contrast.", "Bone density scan, dual energy x-ray absorptiometry."]
IMPRESSION = ["Impression: {t}.", "Conclusion: {t}.", "Impression: no acute abnormality.",
              "Impression: {t}. Recommend clinical correlation.",
              "Impression: {t}. Follow up imaging is advised in three months.",
              "Impression: {t}. Comparison with prior imaging is recommended.",
              "In summary, {t}.", "Overall impression: {t}, favoured over {t2}.",
              "Impression: {t}. Findings discussed with the referring physician."]
PATTERNS = [
    "There is a {n} millimetre {t} in the {s} {r}.",
    "No evidence of {t}.", "No focal {t} is identified.",
    "Findings are consistent with {t}.", "The {r} shows mild {t}.",
    "A {n} millimetre {t} is present in the {s} {r}.",
    "There is moderate {t} involving the {s} {r}.",
    "The {t} is unchanged compared with the prior study.",
    "Interval increase in the {t} of the {s} {r}.",
    "Small {s} {t} is noted. No {t2} is seen.",
    "The {s} {r} is unremarkable. There is no {t}.",
    "Differential diagnosis includes {t} and {t2}.",
    "No {t} or {t2}. The {r} is normal in appearance.",
    "Mild {t} is again seen in the {s} {r}, stable since the prior exam.",
    "There has been interval resolution of the previously noted {t}.",
    "A {n} millimetre focus of {t} is seen in the {r}, of uncertain significance.",
    "The {s} {r} demonstrates {t} without associated {t2}.",
    "Correlation with the patient's clinical history of {t} is suggested.",
    "No definite {t}, though this is limited by patient positioning.",
    "Redemonstration of {t} in the {s} {r}, similar in extent to before.",
    "New {t} is noted in the {r} when compared with the prior study.",
]


def _terms() -> list:
    out = []
    for ln in LEXICON.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and re.fullmatch(r"[a-z][a-z \-]{3,30}", ln):
            out.append(ln)
    return out


def _test_sentences() -> set:
    raw = "\n".join(ln for ln in TEST_CORPUS.read_text(encoding="utf-8").splitlines()
                    if not ln.lstrip().startswith("#"))
    return {s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", " ".join(raw.split()))}


def generate(count: int, seed: int = 1) -> list:
    rng = random.Random(seed)
    terms, banned = _terms(), _test_sentences()
    seen, lines = set(), []
    while len(lines) < count:
        parts = [rng.choice(EXAM)] if rng.random() < 0.5 else []
        for _ in range(rng.randint(2, 4)):
            parts.append(rng.choice(PATTERNS).format(
                t=rng.choice(terms), t2=rng.choice(terms), n=rng.choice(NUM),
                s=rng.choice(SIDE), r=rng.choice(REGION)))
        if rng.random() < 0.6:
            parts.append(rng.choice(IMPRESSION).format(t=rng.choice(terms), t2=rng.choice(terms)))
        parts = [p for p in parts if p.lower() not in banned]
        text = " ".join(parts)
        if len(text.split()) >= 8 and text not in seen:
            seen.add(text)
            lines.append(text)
    return lines
