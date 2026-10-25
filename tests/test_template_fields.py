"""One definition of a template field, checked against every shipped template.

The release gate used to see 1 of the 148 fields the templates contain, so a
report could leave with ``[Right / Left]`` still in it. The first test walks
the real template folder: a new template with a new field spelling is covered
without anyone remembering to add a case.
"""

import re

import pytest

from src.core.paths import templates_dir
from src.features.report_release import unfilled_fields
from src.features.template_fields import (
    find_fields,
    next_field,
    previous_field,
    resolve_choice,
)


def _templates():
    return sorted(templates_dir().glob("*.txt"))


def test_every_bracket_in_every_template_is_a_field():
    paths = _templates()
    assert paths, "no templates found"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        brackets = re.findall(r"\[[^\]\n]*\]", text)
        assert len(find_fields(text)) == len(brackets), path.name


def test_release_gate_lists_unfilled_fields_once_each():
    text = "MRI [Right / Left] knee. [X]T magnet. The [right/left] knee. [X] mm."
    assert unfilled_fields(text) == ("Right / Left", "X", "right/left")


@pytest.mark.parametrize("text", ["Measures [5 mm].", "Lesion [2.5 cm] long.", "No brackets."])
def test_ordinary_text_is_not_a_field(text):
    assert unfilled_fields(text) == ()


def test_placeholder_braces_are_a_field():
    assert unfilled_fields("Dear {{referrer}},") == ("referrer",)


def test_tab_moves_through_fields_in_order_and_stops_at_the_end():
    text = "MRI [Right / Left] Knee on a [X]T magnet."
    first = next_field(text, 0)
    second = next_field(text, first.end)
    assert (first.text, second.text) == ("[Right / Left]", "[X]")
    assert next_field(text, second.end) is None
    assert previous_field(text, second.start).text == "[Right / Left]"
    assert previous_field(text, first.start) is None


@pytest.mark.parametrize(
    "field, spoken, expected",
    [
        ("[Right / Left]", "left", "Left"),
        ("[Right / Left]", "Left.", "Left"),
        ("[right/left]", "Right", "right"),
        ("[Right / Left / Bilateral]", "bilateral", "Bilateral"),
        ("[IV contrast administered: yes / no]", "no", "IV contrast administered: no"),
        # A colon with no choice after it is part of an option, not a prefix.
        ("[Right / Left: lateral]", "right", "Right"),
        ("[Right / Left]", "left sided", None),
        ("[X]", "3", None),
        ("[Right / Left]", "", None),
    ],
)
def test_saying_an_option_answers_the_pick_list(field, spoken, expected):
    assert resolve_choice(field, spoken) == expected
