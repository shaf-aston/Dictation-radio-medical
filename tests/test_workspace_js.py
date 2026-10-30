"""Section detection for the report outline (src/ui/frontends/workspace.js), run in Node.

The outline is read from the report text, never drawn into it, so what counts
as a heading is the whole contract: known report headings in any case are top
level, other ALL-CAPS labels are sub-sections, and Title-case labels such as
"Trachea:" stay body text.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "src" / "ui" / "frontends" / "workspace.js"
TEMPLATES = ROOT / "src" / "templates"

_SCRIPT = """
const { parseSections, sectionAt } = require(%(path)s);
const text = %(text)s;
const s = parseSections(text);
const at = %(at)s;
const where = at === null ? null : sectionAt(s, at);
console.log(JSON.stringify({
  sections: s.map(x => ({title: x.title, level: x.level, body: x.body, end: x.end})),
  here: where && where.here ? where.here.title : null,
  parent: where && where.parent ? where.parent.title : null,
}));
"""


def _run(text, at=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _SCRIPT % {"path": json.dumps(str(WORKSPACE)), "text": json.dumps(text), "at": json.dumps(at)}
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_headings_levels_and_bodies():
    text = (
        "EXAMINATION: CT Head\n"
        "Clinical history: fall\n\n"
        "FINDINGS:\n\n"
        "HAEMORRHAGE:\n"
        "Subdural haematoma: None identified.\n\n"
        "IMPRESSION:\n"
        "1. No acute abnormality.\n"
    )
    out = _run(text)
    got = [(s["title"], s["level"]) for s in out["sections"]]
    assert got == [("EXAMINATION", 1), ("Clinical history", 1), ("FINDINGS", 1),
                   ("HAEMORRHAGE", 2), ("IMPRESSION", 1)]
    bodies = {s["title"]: s["body"] for s in out["sections"]}
    assert bodies["EXAMINATION"] == "CT Head"
    # A top-level section owns its sub-sections' text (for pins and summaries).
    assert bodies["FINDINGS"].startswith("HAEMORRHAGE:")
    assert bodies["HAEMORRHAGE"] == "Subdural haematoma: None identified."
    assert bodies["IMPRESSION"] == "1. No acute abnormality."


def test_title_case_labels_and_times_are_body_text():
    out = _run("FINDINGS:\nTrachea: Patent.\nScanned at 10:30 today.\nRatio 1:2.\n")
    assert [s["title"] for s in out["sections"]] == ["FINDINGS"]


def test_caret_section_and_parent():
    text = "FINDINGS:\nLUNGS:\nClear.\n\nIMPRESSION:\nNormal."
    out = _run(text, at=text.index("Clear"))
    assert (out["here"], out["parent"]) == ("LUNGS", "FINDINGS")
    out = _run(text, at=len(text))
    assert (out["here"], out["parent"]) == ("IMPRESSION", None)


def test_section_end_excludes_trailing_blank_lines():
    text = "FINDINGS:\nClear.\n\n\nIMPRESSION:\n"
    out = _run(text)
    assert out["sections"][0]["end"] == text.index("Clear.") + len("Clear.")


@pytest.mark.parametrize("name", sorted(p.name for p in TEMPLATES.glob("*.txt")))
def test_every_template_has_findings_and_a_conclusion(name):
    out = _run((TEMPLATES / name).read_text(encoding="utf-8"))
    tops = {s["title"].upper() for s in out["sections"] if s["level"] == 1}
    assert tops & {"FINDINGS", "RESULTS"}, tops
    assert tops & {"IMPRESSION", "CONCLUSION", "OPINION", "SUMMARY"}, tops
