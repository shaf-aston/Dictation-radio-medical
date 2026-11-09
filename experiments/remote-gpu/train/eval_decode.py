"""Runs ON Kaggle (shipped with the training data): the local eval sets (rd-eval-sets, uploaded by
data/pack_eval.py) decoded here, so the PC only scores text. Two ways, as the app decodes:
greedy (onnx-asr, GPU) and with radiology hotwords (sherpa-onnx beam search, CPU)."""
import json
import os
import tarfile
from pathlib import Path

import onnx_asr
import sherpa_onnx
import soundfile as sf
from nemo.collections.asr.metrics.wer import word_error_rate

import sherpa_view
from text_rules import normalise


def _eval_sets(tmp: Path):
    """(the uploaded folder, the folder holding one sub-folder per set), or None when none was attached."""
    marker = next(Path("/kaggle/input").rglob("eval_sets.json"), None)
    if marker is None:
        return None
    root = marker.parent
    if (root / "eval.tar").exists():
        tarfile.open(root / "eval.tar").extractall(tmp / "eval_sets")
        return root, tmp / "eval_sets"
    return root, next(p.parent.parent for p in root.rglob("manifest.jsonl"))  # Kaggle unpacked the archive


def _sherpa(view: Path, **how):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(view / "encoder.onnx"), decoder=str(view / "decoder.onnx"), joiner=str(view / "joiner.onnx"),
        tokens=str(view / "tokens.txt"), model_type="nemo_transducer", num_threads=os.cpu_count(), **how)


def _text(rec, a, sr: int) -> str:
    st = rec.create_stream()
    st.accept_waveform(sr, a)
    rec.decode_stream(st)
    return (st.result.text or "").strip()


class ViewCheck:
    """The sherpa view must be the same network: sherpa's own greedy search on it should write nearly what
    onnx-asr writes. Not exactly: sherpa computes its own audio features, and on hard Eka clips writes "40"
    where onnx-asr writes "forty": 14% of words differed for the untouched model (smoke, 2026-11-06, the
    same 13.6% when rerun locally), 5.5% for v8. A wrongly cut network writes gibberish (near 100%), so
    settings.json view_check_max_word_gap (0.3) sits between the two."""

    def __init__(self, plain, clips: int):
        self.plain, self.clips, self.pairs = plain, clips, []
        self.log = {"same": 0, "clips": 0, "differ": []}

    def add(self, a, sr: int, greedy: str) -> None:
        if self.log["clips"] >= self.clips:
            return
        sherpa = _text(self.plain, a, sr)
        self.pairs.append((normalise(sherpa), normalise(greedy)))
        same = self.pairs[-1][0] == self.pairs[-1][1]
        self.log["clips"] += 1
        self.log["same"] += same
        self.log["word_gap"] = round(word_error_rate(*zip(*self.pairs)), 4)  # onnx-asr's text as the reference
        if not same and len(self.log["differ"]) < 5:
            self.log["differ"].append([greedy, sherpa])


def decode_eval_sets(folder: Path, cfg: dict, providers: list, tmp: Path, out: Path, save_log) -> None:
    """decodes/<set>.json = {"greedy": {audio: text}, "hotwords": {...}} for the model in `folder`."""
    found = _eval_sets(tmp)
    if found is None:
        return
    root, sets = found
    view = sherpa_view.build(folder, tmp / "sherpa_view")
    # The same arguments as the app's ParakeetHotwordEngine: hotword_beam and hotword_score in parakeet.json
    # copy its _BEAM and its hotword_score setting; keep them equal.
    hot = _sherpa(view, decoding_method="modified_beam_search", max_active_paths=cfg["hotword_beam"],
                  hotwords_file=str(root / "hotwords.txt"), hotwords_score=cfg["hotword_score"],
                  modeling_unit="bpe", bpe_vocab=str(root / "bpe.vocab"))
    greedy = onnx_asr.load_model("nemo-conformer-tdt", str(folder), quantization=cfg["quantization"],
                                 providers=providers).with_timestamps()
    check = ViewCheck(_sherpa(view, decoding_method="greedy_search"), cfg["view_check_clips"])
    (out / "decodes").mkdir(exist_ok=True)
    decoded = {}
    for name in json.loads((root / "eval_sets.json").read_text()):
        rows = [json.loads(ln) for ln in (sets / name / "manifest.jsonl").read_text().splitlines() if ln.strip()]
        rows = rows[: cfg.get("eval_set_limit") or None]  # smoke: a few clips prove the path
        texts = {"greedy": {}, "hotwords": {}}
        for r in rows:
            a, sr = sf.read(sets / name / r["audio"], dtype="float32")
            a = a.mean(axis=1) if a.ndim > 1 else a
            texts["greedy"][r["audio"]] = greedy.recognize(a, sample_rate=sr).text
            texts["hotwords"][r["audio"]] = _text(hot, a, sr)
            check.add(a, sr, texts["greedy"][r["audio"]])
        (out / "decodes" / f"{name}.json").write_text(json.dumps(texts), encoding="utf-8")
        decoded[name] = len(rows)
        save_log(decoded=decoded, view_check=check.log)
