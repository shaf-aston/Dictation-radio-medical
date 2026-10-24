"""PHI de-identification: the shared safety gate before data leaves the device.

This lives in the medical (core-domain) layer, not under ``src.cloud``, because
three callers need it and must NOT all depend on the optional cloud subsystem:
the training collector (the cloud bridge), the optional Groq cleanup add-on in
dictation, and the cloud uploader itself. Keeping it here lets each import it
*downward* and preserves the invariant that the dictation pipeline never imports
``src.cloud.*``.

Nothing is uploaded until it has passed through :class:`DeIdentifier` and
``validate_clean`` has confirmed no known patient identifier survives. The design
is deliberately conservative: when in doubt, a record is dropped rather than risk
a breach.

Two surfaces are scrubbed:
  * **Text**: the correction strings and any context (names, IDs, dates).
  * **Audio**: segments whose transcript contains PHI are silenced, using the
    timestamps already produced by ``Transcriber.transcribe``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.patient_schema import IDENTIFYING_KEYS

logger = logging.getLogger(__name__)


class PrivacyError(Exception):
    """De-identification failed validation: data must NOT leave the device.

    Re-exported from :mod:`src.cloud.exceptions` for back-compat with existing
    cloud-side callers.
    """


# Generic PHI patterns applied regardless of the known patient_info fields.
# Every way a date is written, not a list of formats: numbers in any order
# and separator ("1960-02-01", "01.02.1960", "3/4/2024"), and a month name
# with a day and a year in either order ("1 Feb 1960", "February 1st, 1960").
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b"),
    re.compile(
        rf"\b(?:{_DAY}\s+(?:of\s+)?)?{_MONTH}\s+(?:{_DAY},?\s+)?\d{{4}}\b", re.IGNORECASE
    ),
]
# Accession / MRN / NHS-style numbers: an optional letter prefix, then digit
# groups that may be split by a space, hyphen or slash ("ACC2024-5566",
# "943 476 5919"). Counted as an identifier at six digits or more, so a
# measurement ("12 mm", "15 20 mm") never is.
_ID_PATTERN = re.compile(r"\b[A-Z]{0,5}[-/]?\d{2,}(?:[ \-/]\d{2,})*\b")
_ID_MIN_DIGITS = 6

#: What the scrubbers put in place of what they remove. Never itself a leak,
#: so validation ignores it: otherwise a patient called "Ted" fails every
#: check on the "ted" inside "[REDACTED]" and nothing can ever be sent.
_PLACEHOLDER = re.compile(r"\[(?:REDACTED|DATE|ID|PHI-\d+)\]")
_TOKEN = "[PHI-{}]"
_LABEL = {"identifier": "[REDACTED]", "date": "[DATE]", "id": "[ID]"}

# Fields of patient_info that are treated as direct identifiers. Derived from
# the shared schema so a new patient field is scrubbed by default, rather than
# silently sailing through to an upload because this list wasn't updated.
_PHI_FIELDS = IDENTIFYING_KEYS
#: A name part shorter than this ("Li", "Jo") is matched only as the whole
#: value: two letters as a free word would redact every "jo" and "li".
_MIN_NAME_PART = 3
#: An identifier this short, with separators removed, cannot be searched for
#: on its own without matching ordinary numbers.
_MIN_ID_CHARS = 5


def _digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def _value_pattern(value: str) -> Optional[re.Pattern]:
    """How one patient-info value is found in text, decided by its shape.

    With digits it is an identifier: matched whatever separators the text puts
    between its characters ("MRN12345678" = "MRN-1234-5678"). Without, it is a
    name: each part matched as a whole word in any order or case ("Jane Doe"
    finds "Doe, Jane", "Mrs Doe", "Jane"), never inside another word ("Ann"
    is not in "anular").
    """
    if any(ch.isdigit() for ch in value):
        chars = re.findall(r"[A-Za-z0-9]", value)
        if len(chars) < 2:
            return None
        body = r"[\s\-./]*".join(re.escape(ch) for ch in chars)
        return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)
    parts = [p for p in re.findall(r"[^\W\d_]+", value) if len(p) >= _MIN_NAME_PART]
    if not parts:
        return re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE) if len(value) >= 2 else None
    alternatives = "|".join(re.escape(p) for p in sorted(set(parts), key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


class DeIdentifier:
    """Removes patient identifiers from training text and audio."""

    def __init__(self, patient_info: Optional[Dict[str, str]] = None) -> None:
        self._patient_info = patient_info or {}
        values = [(self._patient_info.get(f) or "").strip() for f in _PHI_FIELDS]
        self._identifier_patterns: List[re.Pattern] = [
            p for p in (_value_pattern(v) for v in values if len(v) >= 2) if p is not None
        ]
        # What validate_clean looks for, independently of the patterns above:
        # name words, and identifiers with every separator removed.
        self._name_words = {
            p.casefold() for v in values if not any(ch.isdigit() for ch in v)
            for p in re.findall(r"[^\W\d_]+", v) if len(p) >= _MIN_NAME_PART
        }
        self._compact_ids = {
            c for v in values if any(ch.isdigit() for ch in v)
            if len(c := re.sub(r"[^a-z0-9]", "", v.casefold())) >= _MIN_ID_CHARS
        }

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def _spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Every identifier in *text*: ``(start, end, kind)``, in order, not
        overlapping. The one place that decides what is PHI; both scrubs use it.
        """
        found: List[Tuple[int, int, str]] = []
        for kind, patterns in (
            ("identifier", self._identifier_patterns),
            ("date", _DATE_PATTERNS),
            ("id", (_ID_PATTERN,)),
        ):
            for pattern in patterns:
                for m in pattern.finditer(text):
                    if kind == "id" and _digits(m.group(0)) < _ID_MIN_DIGITS:
                        continue
                    if not any(m.start() < e and s < m.end() for s, e, _ in found):
                        found.append((m.start(), m.end(), kind))
        return sorted(found)

    def _replace_spans(self, text: str, label) -> str:
        out, pos = [], 0
        for start, end, kind in self._spans(text):
            out.append(text[pos:start])
            out.append(label(kind, text[start:end]))
            pos = end
        out.append(text[pos:])
        return "".join(out)

    def deidentify_text(self, text: str) -> str:
        """Return *text* with patient identifiers, dates, and IDs redacted."""
        if not text:
            return text
        return self._replace_spans(text, lambda kind, _original: _LABEL[kind])

    def validate_clean(self, text: str) -> bool:
        """Return True only if no known patient identifier remains, in any form.

        Checked a different way from the scrub, so it is a second opinion and
        not the scrub's patterns run twice: every word of the text against the
        patient's name words, and the text with all separators removed against
        each identifier with its separators removed.

        Raises :class:`PrivacyError` if a known identifier survived: callers
        should treat the record as un-uploadable.
        """
        text = _PLACEHOLDER.sub(" ", text).casefold()
        words = set(re.findall(r"[^\W\d_]+", text))
        compact = re.sub(r"[^a-z0-9]", "", text)
        if words & self._name_words or any(i in compact for i in self._compact_ids):
            raise PrivacyError(
                "Patient identifier survived de-identification; record dropped."
            )
        return True

    def pseudonymise(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Scrub *text* reversibly, for a round trip that must come back whole.

        The same spans :meth:`deidentify_text` removes become numbered tokens
        (``[PHI-1]``), and the map from token to original is returned so
        :func:`restore` can put the radiologist's own text back.
        """
        originals: Dict[str, str] = {}

        def keep(_kind: str, original: str) -> str:
            token = _TOKEN.format(len(originals) + 1)
            originals[token] = original
            return token

        return self._replace_spans(text, keep), originals

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def deidentify_audio(
        self,
        wav_path: str,
        segments: List[dict],
        out_dir: Path,
        session_id: str,
    ) -> Optional[str]:
        """Silence any audio segments whose transcript contains PHI.

        Returns the path to the scrubbed WAV clip, or None if the audio could
        not be processed (the caller then falls back to a text-only record).
        Requires soundfile + numpy, which are already core dependencies.
        """
        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover - deps always present in app
            logger.warning("Audio de-identification skipped (deps missing): %s", exc)
            return None

        try:
            audio, sr = sf.read(wav_path, dtype="float32")
        except Exception as exc:
            logger.warning("Could not read audio for de-identification: %s", exc)
            return None

        if audio.ndim > 1:
            audio = audio[:, 0]

        pad = int(0.5 * sr)  # silence ±0.5s around a flagged segment
        for seg in segments:
            seg_text = (seg.get("text") or "")
            if self._contains_phi(seg_text):
                start = max(0, int(float(seg.get("start", 0)) * sr) - pad)
                end = min(len(audio), int(float(seg.get("end", 0)) * sr) + pad)
                audio[start:end] = 0.0
                logger.info("Silenced PHI audio segment %.1f–%.1fs",
                            seg.get("start", 0), seg.get("end", 0))

        out_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(wav_path.encode("utf-8")).hexdigest()[:8]
        out_path = out_dir / f"{session_id}_{digest}.wav"
        try:
            sf.write(str(out_path), audio, sr)
        except Exception as exc:
            logger.warning("Could not write de-identified audio: %s", exc)
            return None
        return str(out_path)

    def _contains_phi(self, text: str) -> bool:
        """The audio is silenced exactly where the text would be scrubbed."""
        return bool(text) and bool(self._spans(text))


def restore(text: str, originals: Dict[str, str]) -> Optional[str]:
    """Put back what :meth:`DeIdentifier.pseudonymise` took out, or None.

    None unless every token comes back exactly once and no unknown token
    appears: a token dropped, doubled or invented means the text that came
    back cannot be mapped onto the report safely.
    """
    found = re.findall(r"\[PHI-\d+\]", text)
    if sorted(found) != sorted(originals):
        return None
    return re.sub(r"\[PHI-\d+\]", lambda m: originals[m.group(0)], text)
