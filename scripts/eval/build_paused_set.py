"""Derive a pause-heavy gold set from an existing one, references intact.

Every gold set in this project is continuous speech, because a text-to-speech
engine reading a report does not stop to think. That makes all four of them
blind to the thing a radiologist actually does: dictate a sentence, pause to
read the film, dictate the next one. A chunk policy is judged almost entirely
on how it behaves in those pauses, so measuring one on continuous speech scores
it as zero by construction.

Inserting silence adds no words, so the source set's reference text stays
exactly correct. That is the whole trick: a real reference and a realistic
pause shape at the same time, without recording anything.

This is NOT a substitute for the `own` set. It still uses synthesised voice
under unrealistically clean acoustics, so it measures chunk POLICY, never real
acoustics or a real accent.

Run:
    python -m scripts.eval.build_paused_set --from tts --gap 6.0
    python -m scripts.eval.replay --set tts_paused --limit 6
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

from scripts.eval import corpus
from src.dictation.stream.vad import SAMPLE_RATE as SR
from src.dictation.stream.vad import detect_speech


def insert_pauses(audio: np.ndarray, gap_sec: float, every_sec: float) -> np.ndarray:
    """Put ``gap_sec`` of true silence at real speech boundaries.

    The gaps go at VAD-confirmed mark ends, never at an arbitrary offset, so no
    word is ever cut in half: the clip stays honest audio of the same sentences
    with longer thinking time between them.
    """
    marks = detect_speech(audio)
    if not marks:
        return audio

    silence = np.zeros(int(gap_sec * SR), dtype=np.float32)
    out, cursor, spoken_since = [], 0, 0.0
    for mark in marks:
        spoken_since += (mark.end_sample - mark.start_sample) / SR
        if spoken_since >= every_sec and mark.end_sample > cursor:
            out.append(audio[cursor : mark.end_sample])
            out.append(silence)
            cursor = mark.end_sample
            spoken_since = 0.0
    out.append(audio[cursor:])
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="source", default="tts", help="set to derive from")
    ap.add_argument("--name", default="", help="new set name (default <source>_paused)")
    ap.add_argument("--gap", type=float, default=6.0, help="seconds of silence to insert")
    ap.add_argument("--every", type=float, default=8.0,
                    help="insert a gap after this many seconds of speech")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    name = args.name or f"{args.source}_paused"
    clips = corpus.load_set(args.source)
    if args.limit:
        clips = clips[: args.limit]

    dest_dir = corpus.eval_set_dir(name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    from faster_whisper.audio import decode_audio

    built = []
    for clip in clips:
        audio = decode_audio(str(clip.audio_path), sampling_rate=SR)
        paused = insert_pauses(audio, args.gap, args.every)
        out_path = dest_dir / clip.audio_path.name
        sf.write(out_path, paused, SR)
        built.append(corpus.Clip(
            set_name=name,
            audio_path=out_path,
            reference=clip.reference,      # unchanged: silence carries no words
            source=f"{clip.source} + {args.gap}s pauses every {args.every}s of speech",
            licence=clip.licence,
            synthetic=True,
            duration_sec=len(paused) / SR,
        ))
        print(f"{clip.audio_path.name}: {len(audio)/SR:.1f}s -> {len(paused)/SR:.1f}s")

    corpus.write_manifest(name, built)
    total_in = sum(c.duration_sec for c in clips)
    total_out = sum(c.duration_sec for c in built)
    print(f"\nwrote {len(built)} clips to {dest_dir}")
    print(f"{total_in/60:.1f} min of speech -> {total_out/60:.1f} min with pauses")


if __name__ == "__main__":
    main()
