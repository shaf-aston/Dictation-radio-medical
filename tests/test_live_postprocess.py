"""Live post-processing gives the same report as processing the whole text.

The live path processes only the part of the transcript that is new; the text
handed back at Stop comes from it. Any difference from the whole-document
pass is a report the final pass would not have written: lost paragraph
breaks, a doubled full stop, dropped words (all seen, 2026-10-24).
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.dictation.postprocess.incremental import IncrementalPostprocessor
from src.dictation.postprocess.pipeline import postprocess_transcript

WORDS = ["no", "acute", "fracture", "the", "spine", "is", "intact", "small", "effusion",
         "dr.", "e.g.", "period", "comma", "new", "paragraph", "line", "fracture.",
         "clear.", "noted,", "thank", "you", "colon", "findings", "question", "mark",
         "scratch", "that", "menscus", "correct", "word", "meniscus", "2.5", "mm"]
chunk = st.lists(st.sampled_from(WORDS), min_size=1, max_size=6).map(" ".join)
separator = st.sampled_from([" ", " ", ". ", ".\n", "\n\n", "\n"])


@settings(max_examples=200, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.lists(st.tuples(chunk, separator), min_size=1, max_size=7), st.data())
def test_live_result_equals_the_whole_document_pass(parts, data):
    live = IncrementalPostprocessor()
    raw = ""
    for text, sep in parts:
        raw = text if not raw else raw + sep + text
        committed = data.draw(st.integers(0, len(raw)))
        assert live.process(raw, committed)[0] == postprocess_transcript(raw), raw


def test_paragraph_breaks_survive_the_live_path():
    live = IncrementalPostprocessor()
    raw = "no acute fracture. new paragraph the spine is intact."
    live.process("no acute fracture.", 18)
    assert live.process(raw, len(raw))[0] == "No acute fracture.\n\nThe spine is intact."
