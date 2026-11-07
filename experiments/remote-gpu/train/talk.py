"""Runs ON Kaggle (shipped with the training data): real recorded clinical talk with hand-made transcripts.

PriMock57 (github babylonhealth/primock57, CC BY 4.0): 57 mock GP consultations, real clinicians and actor
patients, each side on its own channel, transcribed utterance by utterance with times (Praat TextGrid).
An utterance carrying a tag (<UNSURE>, <UNIN/>, <INAUDIBLE_SPEECH/>: the transcriber was unsure or could
not hear a word) is skipped, since its text may not be what was said.

Fareez et al. 2022 (figshare article 16550013, CC0): 272 mock OSCE interviews by residents and students,
mostly respiratory, one mixed channel each; machine transcribed, corrected by hand, checked by a second
person. No times, so the whole interview goes to long_clips.Cutter.

Both hold out every `every`-th consultation (sorted by name) whole: held=False yields the rest, to train on.
"""
import json
import re
import urllib.request
import zipfile
from pathlib import Path

from symspellpy import SymSpell, Verbosity

from sources import SR, decode
from text_rules import normalise, talk_text

PRIMOCK = "https://api.github.com/repos/babylonhealth/primock57/contents/transcripts"
PRIMOCK_AUDIO = "https://media.githubusercontent.com/media/babylonhealth/primock57/main/audio/"
INTERVAL = re.compile(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "([^"]*)"')
SPEAKER = re.compile(r"^\s*[DdPp](?:\s*[:;]+|\s+(?=[A-Z]))\s*")  # "D:", "P::", "D;", a bare "P "


def _get(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _held(names, every: int) -> set:
    return set(sorted(set(names))[::every])


def primock(cache: Path, every: int, held: bool):
    """(utterance audio, text) for every certain utterance of the held-out or the training consultations."""
    grids = sorted((f for f in json.loads(urllib.request.urlopen(PRIMOCK).read()) if f["name"].endswith(".TextGrid")),
                   key=lambda f: f["name"])
    group = {f["name"]: f["name"].rsplit("_", 1)[0] for f in grids}  # day1_consultation01_doctor.TextGrid
    out = _held(group.values(), every)
    for f in grids:
        if (group[f["name"]] in out) != held:
            continue
        channel = f["name"][: -len(".TextGrid")]
        grid = _get(f["download_url"], cache / f["name"]).read_text(encoding="utf-8")
        a = decode(_get(PRIMOCK_AUDIO + channel + ".wav", cache / f"{channel}.wav").read_bytes())
        for x0, x1, text in INTERVAL.findall(grid):
            if text.strip() and "<" not in text:
                yield a[int(float(x0) * SR): int(float(x1) * SR)], talk_text(text)


def fareez(url: str, cache: Path, every: int, held: bool):
    """(whole interview audio, its transcript as one text) for the held-out or the training interviews."""
    z = zipfile.ZipFile(_get(url, cache / "fareez.zip"))
    names = sorted(n for n in z.namelist() if n.endswith(".mp3"))
    out = _held(names, every)
    for name in names:
        if (name in out) != held:
            continue
        case = Path(name).stem
        raw = z.read(f"Data/Clean Transcripts/{case}.txt").decode("utf-8", "replace")
        turns = [talk_text(SPEAKER.sub("", t)) for t in re.split(r"\n\s*\n", raw)]
        yield decode(z.read(name)), " ".join(t if t[-1] in ".?!" else t + "." for t in turns if t)


class Slips:
    """A hand transcript's misspellings ("athsma", "ny" for any): a word outside `known` (the English
    dictionary + the app's radiology lexicon), within 2 letters of a word in it, that the stock model did not
    hear in that clip. A real drug name the model mishears ("Fexofenadine" heard "fexaphenidine") is far
    from every dictionary word, so it stays and is taught; numbers always stay."""

    def __init__(self, known: set):
        self.known = known
        self.speller = SymSpell(max_dictionary_edit_distance=2)
        for w in known:
            self.speller.create_dictionary_entry(w, 1)

    def __call__(self, text: str, hear) -> list:
        """The slips in `text`; `hear()` gives the stock model's words for the clip (decoded only if needed)."""
        odd = [w for w in normalise(text).split() if not any(c.isdigit() for c in w)
               and w not in self.known and w.split("'")[0] not in self.known]
        near = [w for w in odd if self.speller.lookup(w, Verbosity.TOP, 2)]
        if not near:
            return []
        heard = set(hear())
        return [w for w in near if w not in heard]
