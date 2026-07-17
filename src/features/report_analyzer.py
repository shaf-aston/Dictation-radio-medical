"""Local statistical analysis of completed reports — no cloud, no second model.

Mines the plain-text reports in ``data/autosave/`` for the patterns the user
asked about: most-frequent terminology, common phrasing per section, repeated
sentence templates, and body-region distribution. This is deliberately
rule/statistics based (``collections.Counter`` + ``rapidfuzz`` dedup) rather than
a trained model: the signal is frequency, the corpus is small, and the result
must be transparent to a clinician.

Runs in a background QThread on startup when ``report_analysis_enabled`` is set,
writing ``data/analysis/report_analytics.json`` for the UI to surface.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.json_store import write_json
from src.features.report_manager import _SECTION_HEADERS

logger = logging.getLogger(__name__)

# Words too common to be informative; medical stop-list on top of generic English.
_STOPWORDS = {
    "the", "and", "is", "are", "with", "without", "of", "to", "in", "on", "no",
    "a", "an", "as", "at", "or", "be", "was", "were", "this", "that", "there",
    "seen", "noted", "demonstrates", "demonstrated", "appears", "appearance",
    "normal", "unremarkable", "within", "limits", "left", "right", "patient",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Body-region keywords for the distribution count.
_BODY_REGIONS = {
    "knee": ["knee", "meniscus", "acl", "pcl", "patella"],
    "shoulder": ["shoulder", "rotator", "supraspinatus", "glenoid", "labrum"],
    "hip": ["hip", "acetabul", "femoral head", "labral"],
    "spine": ["spine", "vertebra", "disc", "lumbar", "cervical", "thoracic"],
    "ankle": ["ankle", "talus", "achilles", "calcaneus"],
    "wrist": ["wrist", "carpal", "scaphoid", "triquetral"],
    "elbow": ["elbow", "olecranon", "epicondyle"],
    "chest": ["chest", "lung", "pulmonary", "pleural", "mediastin"],
}


@dataclass
class ReportAnalytics:
    """Aggregated analysis result, serialisable to JSON for the UI."""

    generated_at: str
    report_count: int
    top_terms: List[Tuple[str, int]] = field(default_factory=list)
    section_phrases: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    template_suggestions: List[Tuple[str, int]] = field(default_factory=list)
    body_region_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "report_count": self.report_count,
            "top_terms": self.top_terms,
            "section_phrases": self.section_phrases,
            "template_suggestions": self.template_suggestions,
            "body_region_distribution": self.body_region_distribution,
        }


class ReportAnalyzer:
    """Scans autosaved reports and produces a :class:`ReportAnalytics`."""

    def __init__(self) -> None:
        from src.features.file_manager import autosave_dir
        self._autosave_dir = autosave_dir()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def analyze_all_reports(self) -> ReportAnalytics:
        texts = self._load_report_bodies()
        analytics = ReportAnalytics(
            generated_at=datetime.now(timezone.utc).isoformat(),
            report_count=len(texts),
        )
        if not texts:
            return analytics

        analytics.top_terms = self._top_terms(texts, n=50)
        analytics.section_phrases = self._section_phrases(texts)
        analytics.template_suggestions = self._template_suggestions(texts)
        analytics.body_region_distribution = self._body_regions(texts)
        return analytics

    def analyze_and_save(self) -> Path:
        """Run analysis and persist the result; return the output path."""
        from src.features.file_manager import analysis_dir
        analytics = self.analyze_all_reports()
        out = analysis_dir() / "report_analytics.json"
        write_json(out, analytics.to_dict())
        logger.info("Report analytics written to %s (%d reports)",
                    out, analytics.report_count)
        return out

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_report_bodies(self) -> List[str]:
        bodies: List[str] = []
        if not self._autosave_dir.is_dir():
            return bodies
        for f in self._autosave_dir.glob("*.txt"):
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            bodies.append(self._strip_header(raw))
        return bodies

    @staticmethod
    def _strip_header(raw: str) -> str:
        """Return just the report body — no patient header, no footer.

        ``report_manager`` formats reports with full-width '=' rule lines: a
        title, rule, patient-info block, rule, the body, then a final rule and
        footer (report date / radiologist line). The body is what sits between
        the *second* rule and the *last* rule. We locate the rule lines rather
        than splitting on the rule text, so neither patient identifiers nor the
        boilerplate footer leak into the term/phrase statistics.
        """
        lines = raw.splitlines()
        rules = [
            i for i, ln in enumerate(lines)
            if set(ln.strip()) == {"="} and len(ln.strip()) >= 10
        ]
        if len(rules) >= 3:
            body = lines[rules[1] + 1: rules[-1]]
        elif len(rules) >= 2:
            body = lines[rules[1] + 1:]
        elif len(rules) == 1:
            body = lines[rules[0] + 1:]
        else:
            body = lines
        return "\n".join(body).strip()

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------

    def _top_terms(self, texts: List[str], n: int) -> List[Tuple[str, int]]:
        counter: Counter = Counter()
        for text in texts:
            for w in _WORD_RE.findall(text.lower()):
                if w not in _STOPWORDS:
                    counter[w] += 1
        return counter.most_common(n)

    def _section_phrases(self, texts: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """Group sentences by the section header they fall under."""
        section_sentences: Dict[str, Counter] = {}
        for text in texts:
            current = "GENERAL"
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                upper = stripped.upper().rstrip(":")
                if upper in {h.rstrip(":") for h in _SECTION_HEADERS}:
                    current = upper
                    continue
                for sent in _SENTENCE_RE.split(stripped):
                    sent = sent.strip()
                    if len(sent) >= 12:
                        section_sentences.setdefault(current, Counter())[sent.lower()] += 1
        return {
            sec: counter.most_common(10)
            for sec, counter in section_sentences.items()
        }

    def _template_suggestions(self, texts: List[str]) -> List[Tuple[str, int]]:
        """Repeated full sentences are candidate macros/templates."""
        counter: Counter = Counter()
        for text in texts:
            for sent in _SENTENCE_RE.split(text):
                sent = " ".join(sent.split())
                if len(sent) >= 20:
                    counter[sent] += 1
        # Only sentences that recur are worth suggesting; dedup near-duplicates.
        recurring = [(s, c) for s, c in counter.most_common(50) if c >= 2]
        return self._dedup_similar(recurring)

    @staticmethod
    def _dedup_similar(phrases: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Collapse near-identical phrases using rapidfuzz when available."""
        try:
            from rapidfuzz import fuzz
        except Exception:
            return phrases
        kept: List[Tuple[str, int]] = []
        for phrase, count in phrases:
            if all(fuzz.ratio(phrase, k) <= 90 for k, _ in kept):
                kept.append((phrase, count))
        return kept

    def _body_regions(self, texts: List[str]) -> Dict[str, int]:
        dist: Counter = Counter()
        for text in texts:
            lower = text.lower()
            for region, keywords in _BODY_REGIONS.items():
                if any(k in lower for k in keywords):
                    dist[region] += 1
        return dict(dist)
