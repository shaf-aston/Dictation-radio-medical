"""Upload the local test sets to Kaggle once, so each training run scores its model on them on the GPU.

    python data/pack_eval.py            # sets listed in settings.json "eval_sets"

The kernel then decodes every clip greedy (as the app's onnx-asr engine does) and with radiology
hotwords (as the app's sherpa-onnx engine does) and returns the raw text; run_train.py files it
into compare.py's decode cache, so a new version is scored locally in seconds, not hours.

Only public or synthetic audio may leave this PC: a clip that is real speech under the "own"
licence (the user's voice) stops the upload, whatever the set list says.
"""
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from config import HERE, REPO, SETTINGS, load_env  # noqa: E402
from kaggle_io import push_dataset  # noqa: E402

sys.path.insert(0, str(REPO))
from scripts.eval.corpus import load_set  # noqa: E402
from src.core.paths import eval_set_dir  # noqa: E402
from src.dictation.asr.engines.parakeet_hotword_engine import write_bpe_vocab, write_hotwords  # noqa: E402
from src.medical.medical_dict import get_correction_targets  # noqa: E402

PACK = HERE / "train" / "_eval_pack"


def main() -> None:
    shutil.rmtree(PACK, ignore_errors=True)
    PACK.mkdir(parents=True)
    with tarfile.open(PACK / "eval.tar", "w") as tf:
        for name in SETTINGS["eval_sets"]:
            for c in load_set(name):
                if not c.synthetic and c.licence == "own":
                    raise SystemExit(f"{name}/{c.audio_path.name} is real speech under licence 'own': not uploading")
            folder = eval_set_dir(name)
            tf.add(folder / "manifest.jsonl", arcname=f"{name}/manifest.jsonl")
            for c in load_set(name):
                tf.add(c.audio_path, arcname=f"{name}/{c.audio_path.name}")
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
