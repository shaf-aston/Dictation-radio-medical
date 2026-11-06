"""Build the local `protocol` eval set: python data/build_protocol_eval.py [count]

MRI protocol dictation spoken with symbols ("right slash left", "open bracket ...
close bracket", "one point five tesla"), read only by settings.json's held-out
voice, which no training run uses. Two parts:

- HAND: lines written like real reports, not from the training templates
  (tests whether the wording generalises).
- generated: the training templates with a separate seed; any line that is in
  a training set (data/train manifest, or any models/*/train_texts.txt) is
  skipped (tests whether the trained wording is heard in an unseen voice).

The reference is what the report should finally say ("right/left", "(4 mm)",
"1.5 T"), so compare.py's "+fix" column and --features score what the user sees.
WER ignores punctuation; use compare.py --features for slash and brackets.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyttsx3  # noqa: E402
from config import MODELS, SETTINGS  # noqa: E402
from gen_text import _protocol_line, to_target  # noqa: E402
from scripts.eval.corpus import Clip, audio_duration, write_manifest  # noqa: E402
from src.core.paths import eval_set_dir  # noqa: E402

TRAIN_MANIFEST = Path(__file__).resolve().parent / "train" / "manifest.jsonl"
SEED = 4242

HAND = [
    # The user's sentence, reworded: the exact wording is a training template.
    "Multiplanar, multisequence MRI of the right slash left elbow was carried out on a one point five tesla system.",
    "Sequences colon coronal PD fat saturated, coronal T one, sagittal PD fat saturated, axial PD fat saturated.",
    "MRI of the left shoulder on a three tesla scanner.",
    "The anterior slash posterior labrum is intact.",
    "There is a full thickness tear of the supraspinatus tendon open bracket twelve millimetres retraction close bracket.",
    "Bone marrow oedema in the lateral femoral condyle open bracket image fourteen close bracket.",
    "No ligamentous injury slash tear.",
    "Mild tendinosis of the common flexor slash pronator origin.",
    "Comparison colon radiograph from last month open bracket no fracture close bracket.",
    "The ulnar collateral ligament is intact on T two fat saturated images.",
    "Examination performed on a one point five T system.",
    "Small ganglion cyst open bracket six millimetres close bracket at the dorsal wrist.",
    "The ACL slash PCL are intact.",
    "Low signal on T one slash T two weighted images is in keeping with sclerosis.",
    "Imaging of the right knee on a three T magnet, sagittal STIR and axial T two.",
    "Findings colon moderate joint effusion open bracket see sagittal STIR close bracket.",
    "Medial slash lateral menisci are intact.",
    "Grade two strain of the medial head of gastrocnemius.",
    "No fracture, dislocation or bone bruise.",
    "Sequences colon sagittal T one, sagittal STIR, axial gradient echo.",
    # 2026-10-31: 80 more, so the generalisation number rests on ~100 clips.
    "MRI lumbar spine on the three tesla unit.",
    "Disc bulge at L four slash five with mild canal narrowing.",
    "Broad based protrusion at L five slash S one contacting the traversing nerve root.",
    "Modic type one endplate change at L three slash four.",
    "Facet joint effusions bilaterally open bracket greater on the right close bracket.",
    "The conus terminates at the level of L one slash two.",
    "Cervical spine MRI shows uncovertebral hypertrophy at C five slash six.",
    "Mild foraminal stenosis on the left at C six slash seven.",
    "No cord signal abnormality on sagittal T two.",
    "Pituitary protocol on a one point five tesla system with dynamic contrast.",
    "Brain MRI on the three T scanner, axial FLAIR and diffusion.",
    "Scattered white matter hyperintensities open bracket Fazekas grade one close bracket.",
    "No restricted diffusion to suggest acute infarct.",
    "The internal auditory meati are symmetrical on thin section T two images.",
    "Small vestibular schwannoma on the right open bracket five millimetres close bracket.",
    "Hip MRI on a one point five tesla magnet, coronal STIR and axial T one.",
    "Labral tear at the anterosuperior labrum open bracket twelve to two o'clock close bracket.",
    "Cam slash pincer morphology of the right hip.",
    "Gluteus medius tendinopathy without full thickness tear.",
    "Trochanteric bursal fluid open bracket small volume close bracket.",
    "Ankle MRI shows a sprain of the anterior talofibular ligament.",
    "The peroneus brevis slash longus tendons are intact.",
    "Osteochondral lesion of the medial talar dome open bracket eight millimetres close bracket.",
    "Plantar fascia thickening at its calcaneal origin open bracket six millimetres close bracket.",
    "Achilles tendinopathy with intrasubstance signal on T two fat saturated images.",
    "Wrist MRI on a three tesla magnet with a dedicated coil.",
    "Central perforation of the triangular fibrocartilage slash TFCC.",
    "The scapholunate slash lunotriquetral ligaments are intact.",
    "Undisplaced fracture of the scaphoid waist on coronal T one.",
    "Ganglion arising from the scapholunate interval open bracket nine millimetres close bracket.",
    "Shoulder MRI shows subacromial slash subdeltoid bursitis.",
    "Partial thickness articular surface tear of supraspinatus open bracket PASTA lesion close bracket.",
    "Biceps tendon is located in the bicipital groove on axial PD.",
    "Acromioclavicular joint arthropathy with inferior osteophytes.",
    "Hill Sachs defect on the posterolateral humeral head.",
    "Knee MRI on a one point five T unit.",
    "Bucket handle tear of the medial meniscus open bracket double PCL sign close bracket.",
    "Horizontal tear of the posterior horn slash body of the lateral meniscus.",
    "Grade one MCL sprain with periligamentous oedema.",
    "Patellar tendinopathy at the proximal attachment.",
    "Chondral fissuring of the patellar apex open bracket grade two close bracket.",
    "Baker's cyst open bracket thirty millimetres close bracket without rupture.",
    "Bone bruise of the lateral femoral condyle slash posterolateral tibial plateau.",
    "Elbow MRI shows lateral epicondylitis with a partial tear.",
    "The radial collateral slash lateral ulnar collateral ligaments are intact.",
    "Distal biceps tendon is intact on FABS view.",
    "Olecranon bursitis open bracket fluid collection close bracket.",
    "Prostate MRI on the three tesla scanner without an endorectal coil.",
    "PI RADS four lesion in the left peripheral zone open bracket eleven millimetres close bracket.",
    "Liver MRI shows a T two bright lesion consistent with a haemangioma.",
    "Simple renal cyst on the left open bracket Bosniak one close bracket.",
    "MRCP shows no filling defect in the common bile duct.",
    "Pancreatic duct is not dilated open bracket two millimetres close bracket.",
    "Pelvic MRI on a one point five tesla magnet, sagittal T two and axial T one.",
    "Intramural fibroid open bracket forty millimetres close bracket.",
    "Endometrial thickness is normal for age.",
    "Sacroiliac joints show subchondral oedema on the right on STIR.",
    "No erosions at the sacroiliac joints on T one.",
    "Thoracic spine shows a haemangioma in T eight.",
    "Wedge compression fracture of T twelve with no retropulsion.",
    "Marrow signal is normal on T one and STIR.",
    "Previous right slash left comparison is not available.",
    "Image quality is degraded by motion open bracket axial series close bracket.",
    "Post contrast T one fat saturated images show no enhancement.",
    "Gadolinium was given intravenously open bracket fifteen millilitres close bracket.",
    "Diffusion weighted imaging shows no restriction.",
    "Small effusion in the radiocarpal joint.",
    "The median nerve is normal in calibre at the carpal tunnel.",
    "Tenosynovitis of the extensor carpi ulnaris.",
    "Rotator cuff muscles show no fatty atrophy open bracket Goutallier zero close bracket.",
    "Calcific tendinopathy is better assessed on radiographs.",
    "There is mild degenerative change slash spondylosis.",
    "Recommend correlation with ultrasound slash clinical findings.",
    "Findings discussed with the orthopaedic team on the day.",
    "Conclusion colon full thickness supraspinatus tear with retraction.",
    "Conclusion colon no internal derangement of the knee.",
    "Opinion colon features in keeping with plantar fasciitis.",
    "The examination was performed on the three T wide bore scanner.",
    "Follow up MRI in six months open bracket same protocol close bracket.",
    "Overall appearances are stable compared with the prior study slash report.",
]


def written(spoken: str) -> str:
    """What the finished report should say for a spoken protocol line."""
    t = to_target(spoken)
    t = re.sub(r"\s*\bopen bracket\s+(.*?)\s+close bracket\b", r" (\1)", t)
    t = re.sub(r"\s+(?:forward )?slash\s+", "/", t)
    t = re.sub(r"\s+colon\b", ":", t)
    t = re.sub(r"\s+comma\b", ",", t)
    t = re.sub(r"\b(\d+(?:\.\d+)?) (?:tesla|T)\b", r"\1 T", t)
    return t


def trained_lines() -> set:
    seen = set()
    if TRAIN_MANIFEST.exists():
        for ln in TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                seen.update({r.get("spoken", ""), r["text"]})
    for f in MODELS.glob("*/train_texts.txt"):
        seen.update(f.read_text(encoding="utf-8").splitlines())
    return {s.strip() for s in seen if s.strip()}


def _norm_sentences(line: str) -> set:
    """Each sentence of *line* as WER sees it: digits, no case, no punctuation."""
    from scripts.eval.metrics import normalize_words
    return {" ".join(normalize_words(to_target(s))) for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()}


def lines(count: int) -> list:
    """HAND, then generated lines none of whose sentences is in any training set."""
    rng, out = random.Random(SEED), list(HAND)
    trained = set().union(*(_norm_sentences(t) for t in trained_lines()))
    leaked = [h for h in HAND if _norm_sentences(h) & trained]
    if leaked:
        raise SystemExit(f"hand-written lines found in training: {leaked}")
    tries = 0
    while len(out) < count and tries < count * 200:
        tries += 1
        s = _protocol_line(rng)
        if not _norm_sentences(s) & trained and s not in out:
            out.append(s)
    return out


def main(count: int) -> None:
    out = eval_set_dir("protocol")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("protocol_*.wav"):
        old.unlink()
    eng = pyttsx3.init()
    voice = [v.id for v in eng.getProperty("voices") if SETTINGS["held_out_voice"] in v.id.lower()]
    if not voice:
        raise SystemExit(f"held-out voice {SETTINGS['held_out_voice']!r} is not installed")
    eng.setProperty("voice", voice[0])
    clips = []
    for i, spoken in enumerate(lines(count)):
        dest = out / f"protocol_{i:04d}.wav"
        eng.save_to_file(spoken, str(dest))
        eng.runAndWait()
        clips.append(Clip(set_name="protocol", audio_path=dest, reference=written(spoken),
                          source="hand" if i < len(HAND) else f"gen_text seed {SEED}",
                          licence="own", synthetic=True, duration_sec=audio_duration(dest)))
    write_manifest("protocol", clips)
    print(f"protocol set: {len(clips)} clips ({len(HAND)} hand-written), voice {voice[0].rsplit(chr(92), 1)[-1]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
