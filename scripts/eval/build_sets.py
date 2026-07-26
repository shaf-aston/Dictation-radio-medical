"""Build the dictation evaluation gold sets.

    python -m scripts.eval.build_sets --set tts
    python -m scripts.eval.build_sets --set own
    python -m scripts.eval.build_sets --set bench
    python -m scripts.eval.build_sets --set libri     # ~350 MB one-time download

Each set writes ``data/eval/<name>/`` with audio plus a ``manifest.jsonl``. All
of it is local: the ``tts`` set is synthesised on-device with the Windows speech
engine, and ``own``/``bench`` never leave the machine at all. Only ``libri``
touches the network, once, and only when explicitly asked for.

The ``bench`` set writes *drafts* transcribed by the current engine and marks
them ``[UNREVIEWED]``. A draft is a machine guess, not ground truth, so
``corpus.load_set`` refuses to score against one until a human has corrected it
and deleted the marker — scoring a model against its own output would report a
flattering number that means nothing.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Sequence

from scripts.eval.corpus import Clip, KNOWN_SETS, audio_duration, write_manifest
from src.core.logging_setup import setup_logging
from src.features.file_manager import bench_audio_dir, eval_set_dir

logger = logging.getLogger(__name__)

_CORPUS_PATH = Path(__file__).resolve().parent / "resources" / "report_corpus.txt"

#: Reports the user reads aloud for the `own` set. Five is enough to expose a
#: consistent accent/microphone problem without being a chore to record.
_OWN_SET_SIZE = 5

#: LibriSpeech test-clean: public domain read speech, the standard neutral
#: yardstick. Only a slice is kept — the full archive is 346 MB and the extra
#: utterances add nothing but runtime.
_LIBRI_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
_LIBRI_CLIPS = 100


# ---------------------------------------------------------------------------
# Shared corpus reader
# ---------------------------------------------------------------------------

def load_report_corpus() -> List[str]:
    """Read the bundled radiology reports, one string per report."""
    raw = _CORPUS_PATH.read_text(encoding="utf-8")
    body = "\n".join(
        ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")
    )
    reports = [" ".join(block.split()) for block in body.split("\n\n")]
    reports = [r for r in reports if r]
    if not reports:
        raise RuntimeError(f"No reports parsed from {_CORPUS_PATH}")
    return reports


# ---------------------------------------------------------------------------
# tts — synthesised radiology reports
# ---------------------------------------------------------------------------

def build_tts(limit: int = 0) -> int:
    """Synthesise the report corpus with the offline Windows speech engine.

    Voices are cycled across reports so the set carries at least a little
    acoustic variety. This audio is far cleaner than a real dictation, which is
    exactly why the manifest flags it ``synthetic`` — it measures how the
    pipeline handles medical *vocabulary*, and says nothing about microphones.
    """
    try:
        import pyttsx3
    except ImportError:
        logger.error(
            "pyttsx3 is not installed - needed only to build this set. "
            "Install it with: pip install pyttsx3"
        )
        return 1

    reports = load_report_corpus()
    if limit:
        reports = reports[:limit]
    out_dir = eval_set_dir("tts")

    engine = pyttsx3.init()
    voices = [v.id for v in engine.getProperty("voices")] or [None]
    base_rate = engine.getProperty("rate")

    clips: List[Clip] = []
    for i, text in enumerate(reports):
        path = out_dir / f"report_{i:03d}.wav"
        voice = voices[i % len(voices)]
        if voice:
            engine.setProperty("voice", voice)
        # Small rate jitter so every clip is not metronomically identical.
        engine.setProperty("rate", base_rate + (i % 3 - 1) * 15)

        engine.save_to_file(text, str(path))
        engine.runAndWait()

        if not path.exists() or path.stat().st_size < 1024:
            logger.error("Synthesis produced no audio for report %d - aborting", i)
            return 1

        clips.append(Clip(
            set_name="tts",
            audio_path=path,
            reference=text,
            source="scripts/eval/resources/report_corpus.txt",
            licence="project-authored",
            synthetic=True,
            duration_sec=audio_duration(path),
        ))
        logger.info("Synthesised %s (%.1fs)", path.name, clips[-1].duration_sec)

    engine.stop()
    write_manifest("tts", clips)
    total = sum(c.duration_sec for c in clips)
    logger.info("tts set: %d clips, %.1f minutes of audio", len(clips), total / 60)
    return 0


# ---------------------------------------------------------------------------
# own — the user's own voice
# ---------------------------------------------------------------------------

def build_own() -> int:
    """Write the reading scripts and a manifest awaiting the user's recordings.

    The reference is the script itself — the user reads it verbatim, so no
    hand-correction step is needed and no machine guess ever enters the truth.
    """
    reports = load_report_corpus()
    out_dir = eval_set_dir("own")

    # Spread the picks across the corpus so the five reports span modalities
    # rather than being five consecutive chest radiographs.
    step = max(1, len(reports) // _OWN_SET_SIZE)
    picks = reports[::step][:_OWN_SET_SIZE]

    clips: List[Clip] = []
    for i, text in enumerate(picks):
        script = out_dir / f"own_{i:02d}.script.txt"
        script.write_text(text + "\n", encoding="utf-8")
        clips.append(Clip(
            set_name="own",
            audio_path=out_dir / f"own_{i:02d}.wav",
            reference=text,
            source="read aloud by the user",
            licence="local-only, never uploaded",
            synthetic=False,
        ))

    write_manifest("own", clips)
    print(f"""
Reading scripts written to: {out_dir}

  1. Open each own_NN.script.txt and read it aloud, recording as own_NN.wav
     in the same folder. Use the same microphone, room, and speaking pace you
     actually dictate with - that realism is the entire point of this set.
  2. Read it verbatim. If you misspeak, re-record; do not edit the script,
     because the script IS the ground truth.
  3. Any format soundfile can read is fine (16 kHz mono WAV is ideal).

Missing recordings are reported and skipped, so you can record them one at a
time and re-run the evaluation as you go.
""")
    return 0


# ---------------------------------------------------------------------------
# bench — the pre-existing clips, once hand-corrected
# ---------------------------------------------------------------------------

def build_bench() -> int:
    """Draft references for ``data/bench_audio/`` using the current engine.

    The drafts are marked ``[UNREVIEWED]`` and are unusable until corrected.
    """
    src_dir = bench_audio_dir()
    wavs = sorted(src_dir.glob("*.wav")) if src_dir.exists() else []
    if not wavs:
        logger.error("No .wav files found in %s", src_dir)
        return 1

    out_dir = eval_set_dir("bench")
    from src.core.settings import Settings
    from src.dictation.transcriber import Transcriber

    settings = Settings()
    transcriber = Transcriber(model_size=settings.get("model_size", "base"))

    clips: List[Clip] = []
    for wav in wavs:
        dest = out_dir / wav.name
        if not dest.exists():
            shutil.copy2(wav, dest)

        draft_path = out_dir / f"{wav.stem}.reference.txt"
        if draft_path.exists():
            reference = draft_path.read_text(encoding="utf-8").strip()
            logger.info("Keeping existing reference for %s", wav.name)
        else:
            logger.info("Transcribing %s to draft a reference...", wav.name)
            text, _ = transcriber.transcribe(str(dest))
            reference = f"[UNREVIEWED] {text}"
            draft_path.write_text(reference + "\n", encoding="utf-8")

        clips.append(Clip(
            set_name="bench",
            audio_path=dest,
            reference=reference,
            source=f"data/bench_audio/{wav.name}",
            licence="project-local",
            synthetic=False,
            duration_sec=audio_duration(dest),
        ))

    write_manifest("bench", clips)
    print(f"""
Drafts written to: {out_dir}

Each *.reference.txt starts with [UNREVIEWED] and holds what the CURRENT model
heard - a machine guess, not truth. Listen to the clip, correct the text, and
delete the [UNREVIEWED] marker. The evaluator refuses to score against a draft
that still carries it, so there is no way to accidentally grade the model
against its own output.
""")
    return 0


# ---------------------------------------------------------------------------
# libri — general-English regression baseline
# ---------------------------------------------------------------------------

def build_libri(clip_count: int = _LIBRI_CLIPS) -> int:
    """Download a slice of LibriSpeech test-clean (public domain read speech).

    The only set that touches the network, and only when explicitly requested.
    It is not medical; its job is to catch "did this change break ordinary
    English" and to give each engine a like-for-like real-time factor.
    """
    import tarfile
    import tempfile
    import urllib.request

    out_dir = eval_set_dir("libri")
    archive = out_dir / "test-clean.tar.gz"

    if not archive.exists():
        logger.info("Downloading LibriSpeech test-clean (~346 MB) from %s", _LIBRI_URL)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
            urllib.request.urlretrieve(_LIBRI_URL, tmp.name)  # noqa: S310 - fixed https URL
            shutil.move(tmp.name, archive)
    logger.info("Extracting %d utterances", clip_count)

    import soundfile as sf

    clips: List[Clip] = []
    with tarfile.open(archive, "r:gz") as tar:
        transcripts: dict = {}
        members = tar.getmembers()
        for m in members:
            if m.name.endswith(".trans.txt"):
                fh = tar.extractfile(m)
                if fh is None:
                    continue
                for line in fh.read().decode("utf-8").splitlines():
                    utt_id, _, text = line.partition(" ")
                    transcripts[utt_id] = text

        for m in members:
            if len(clips) >= clip_count or not m.name.endswith(".flac"):
                continue
            utt_id = Path(m.name).stem
            reference = transcripts.get(utt_id)
            if not reference:
                continue
            fh = tar.extractfile(m)
            if fh is None:
                continue
            dest = out_dir / f"{utt_id}.wav"
            data, rate = sf.read(fh)
            sf.write(dest, data, rate)
            clips.append(Clip(
                set_name="libri",
                audio_path=dest,
                reference=reference,
                source=f"LibriSpeech test-clean/{utt_id}",
                licence="CC BY 4.0",
                synthetic=False,
                duration_sec=audio_duration(dest),
            ))

    if not clips:
        logger.error("Extracted no usable utterances from %s", archive)
        return 1

    write_manifest("libri", clips)
    logger.info("libri set: %d clips", len(clips))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BUILDERS = {
    "tts": lambda a: build_tts(limit=a.limit),
    "own": lambda a: build_own(),
    "bench": lambda a: build_bench(),
    "libri": lambda a: build_libri(clip_count=a.limit or _LIBRI_CLIPS),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build dictation evaluation gold sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<7} {v}" for k, v in KNOWN_SETS.items()),
    )
    parser.add_argument(
        "--set", dest="sets", action="append", required=True, choices=list(_BUILDERS),
        help="gold set to build (repeatable)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="cap the number of clips (tts, libri) - useful for a quick smoke run",
    )
    args = parser.parse_args(argv)

    setup_logging()
    for name in args.sets:
        logger.info("Building set '%s' - %s", name, KNOWN_SETS[name])
        rc = _BUILDERS[name](args)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
