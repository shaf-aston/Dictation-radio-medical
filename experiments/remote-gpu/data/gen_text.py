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


# --- v6: MRI protocol wording and spoken symbols -----------------------------
# Lines are written the way they are SPOKEN ("slash", "open bracket", "one point
# five tesla"): the model learns to hear the words, and the app's post-processing
# turns them into "/", "( )" and "1.5 T". Never write the symbols here, or the
# model and the rules would fight over who formats.
JOINT = ["elbow", "knee", "shoulder", "wrist", "ankle", "hip", "hand", "foot", "thumb"]
FIELD = ["one point five tesla", "three tesla", "one point five T", "three T"]
SEQ = ["coronal PD fat saturated", "coronal T one", "sagittal PD fat saturated",
       "axial PD fat saturated", "sagittal STIR", "axial T two", "coronal STIR",
       "sagittal T one", "axial gradient echo", "sagittal T two fat saturated",
       "axial diffusion weighted imaging"]
WEIGHT = ["T one slash T two", "PD slash T two", "T two slash STIR"]
ORIGIN = ["flexor slash pronator", "extensor slash supinator"]
PROTOCOL = [
    "Multiplanar multisequence MRI of the right slash left {j} performed on a {f} magnet.",
    "Multiplanar, multisequenced MRI of the right slash left {j} performed on a {f} magnet.",
    "MRI of the {s} {j} performed on a {f} magnet. Sequences colon {q1} comma {q2} comma {q3}.",
    "Sequences colon {q1}, {q2}, {q3} and {q4}.",
    "Imaging was performed on a {f} scanner using {q1} and {q2} sequences.",
    "The medial slash lateral compartment is preserved.",
    "Signal change is seen on {w} weighted images.",
    "Small joint effusion open bracket {n} millimetres close bracket.",
    "The {j} shows mild degenerative change open bracket see previous report close bracket.",
    "Common {o} origin is intact open bracket {q1} close bracket.",
    "There is a partial thickness tear open bracket grade two close bracket.",
    "{q1} open bracket image twelve close bracket shows no marrow oedema.",
    "Findings are as before forward slash unchanged.",
    "No fracture slash dislocation.",
    "Comparison colon MRI of the {s} {j} open bracket two years ago close bracket.",
]


def _protocol_line(rng: random.Random) -> str:
    q = rng.sample(SEQ, 4)
    return rng.choice(PROTOCOL).format(
        w=rng.choice(WEIGHT), o=rng.choice(ORIGIN), j=rng.choice(JOINT), f=rng.choice(FIELD), s=rng.choice(SIDE),
        q1=q[0], q2=q[1], q3=q[2], q4=q[3], n=rng.choice(NUM))


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


def generate(count: int, seed: int = 1, protocol_share: float = 0.15) -> list:
    """*count* lines: the original mix, then a *protocol_share* of v6 protocol lines.

    The original lines come first from their own random stream, so the first
    part of a v6 set is exactly a v5 set of the smaller size.
    """
    n_protocol = round(count * protocol_share)
    lines = _report_lines(count - n_protocol, seed)
    rng, banned, seen = random.Random(seed + 1000), _test_sentences(), set(lines)
    tries = 0
    while len(lines) < count and tries < count * 50:
        tries += 1
        parts = [_protocol_line(rng) for _ in range(rng.randint(1, 3))]
        text = " ".join(p for p in parts if p.lower() not in banned)
        if len(text.split()) >= 6 and text not in seen:
            seen.add(text)
            lines.append(text)
    rng.shuffle(lines)
    return lines


def _report_lines(count: int, seed: int) -> list:
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
