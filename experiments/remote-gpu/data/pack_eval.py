"""Upload the local eval sets to Kaggle once, so each training run scores its model on them on the GPU.

    python data/pack_eval.py            # sets listed in settings.json "eval_sets"

The kernel then decodes every clip greedy (as the app's onnx-asr engine does) and with radiology
hotwords (as the app's sherpa-onnx engine does) and returns the raw text; run_train.py files it
into decode_store, so a new version is scored locally in seconds, not hours.

Only synthetic audio, or real speech under a public licence (settings.json "public_speech_licences"),
may leave this PC: any other clip (the user's voice, a blank or mistyped licence) stops the upload,
whatever the set list says. The uploaded manifest lists only the checked clips' file names.
"""
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from config import HERE, SETTINGS, load_env  # noqa: E402
from kaggle_io import push_dataset  # noqa: E402
from scripts.eval.corpus import load_set  # noqa: E402
from src.dictation.asr.engines.parakeet_hotword_engine import write_bpe_vocab, write_hotwords  # noqa: E402
from src.medical.medical_dict import get_correction_targets  # noqa: E402

PACK = HERE / "train" / "_eval_pack"


def uploadable(name: str, clips: list) -> list:
    """`clips` if every one is synthetic or real speech under a public licence; stops otherwise."""
    for c in clips:
        if not (c.synthetic or c.licence in SETTINGS["public_speech_licences"]):
            raise SystemExit(f"{name}/{c.audio_path.name} is real speech under licence {c.licence!r}: not uploading")
    return clips


def add_set(tf: tarfile.TarFile, name: str) -> None:
    """The set's checked clips and a manifest naming only them (the kernel reads nothing else)."""
    clips = uploadable(name, load_set(name))
    if len({c.audio_path.name for c in clips}) != len(clips):  # flattened names would overwrite each other
        raise SystemExit(f"{name}: two clips share a file name")
    rows = "".join(json.dumps({"audio": c.audio_path.name}) + "\n" for c in clips).encode()
    info = tarfile.TarInfo(f"{name}/manifest.jsonl")
    info.size = len(rows)
    tf.addfile(info, io.BytesIO(rows))
    for c in clips:
        tf.add(c.audio_path, arcname=f"{name}/{c.audio_path.name}")


def main() -> None:
    shutil.rmtree(PACK, ignore_errors=True)
    PACK.mkdir(parents=True)
    with tarfile.open(PACK / "eval.tar", "w") as tf:
        for name in SETTINGS["eval_sets"]:
            add_set(tf, name)
    # What the app's hotword engine builds at load time, built by the app's own code.
    write_hotwords(get_correction_targets(), PACK / "hotwords.txt")
    # Every Parakeet version keeps the published token list (the kernel refuses otherwise), so any one serves.
    vocab = next(p for p in sorted((HERE / "models").glob("*/model/vocab.txt")) if (p.parents[1] / "parakeet.json").exists())
    write_bpe_vocab(vocab, PACK / "bpe.vocab")
    (PACK / "eval_sets.json").write_text(json.dumps(SETTINGS["eval_sets"]))
    load_env()
    push_dataset(PACK, SETTINGS["eval_dataset"])
    print("uploaded", SETTINGS["eval_sets"])


if __name__ == "__main__":
    main()
