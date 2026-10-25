# How the safety code is tested

Three kinds of test, each catching what the others miss.

| Kind | Where | Answers |
|---|---|---|
| Example tables | `tests/test_*.py` | Does this sentence give this result? |
| Properties (Hypothesis) | `tests/test_properties.py` | Does a rule hold for **every** input? |
| Mutation testing (mutmut) | `scripts/mutation_test.sh` | Would the tests notice if the code were wrong? |

## Example tables

One row per clinical sentence and the answer it must get. Clinical lists
(critical terms, denial and hedge wording, US→UK pairs) are **written out
again in the test**, not read from the module: a test that loops over the
module's own list still passes when an entry is deleted or misspelt.

## Properties

Hypothesis generates hundreds of inputs per rule and shrinks any failure to
the smallest example. The rules:

- every field `find_fields` returns is where it says, and Tab/Shift+Tab land on it;
- saying any option of a pick-list (any case, trailing punctuation) answers it;
- no patient identifier survives outside a placeholder, and the reversible
  scrub round-trips exactly;
- an AI cleanup that changes nothing gives back the same report;
- UK spelling is idempotent; "scratch that" only removes text;
- the pipeline never loses or invents a number.

The cleanup property found a real bug on its first run: an accepted cleanup
turned every date in the report into `[DATE]`.

## Mutation testing

`scripts/mutation_test.sh` (needs `pip install "mutmut<3"`) makes hundreds of
small deliberate bugs in the seven safety modules (`>=` to `>`, a word
deleted from a list, `continue` to `break`) and runs the tests against each.
A bug no test notices ("survived") is either a gap in the tests or a change
with no visible effect.

| Module | Caught, 2026-10-24 before | after |
|---|---|---|
| `medical/critical_findings.py` | 75/190 (39%) | 188/196 (96%) |
| `medical/spelling.py` | 47/138 (34%) | 130/134 (97%) |
| `dictation/postprocess/voice_commands.py` | 55/135 (41%) | 124/135 (92%) |
| `features/report_checks.py` | 73/96 (76%) | 86/96 (90%) |
| `features/template_fields.py` | 36/44 (82%) | 39/44 (89%) |
| `medical/deid.py` | 14/71 (20%) | 73/96 (76%) |
| `features/report_release.py` | 53/87 (61%) | 66/87 (76%) |
| **all seven** | **353/761 (46%)** | **706/788 (90%)** |

What still survives was read one by one and left on purpose: log message
wording, default values every caller overrides, `frozen=True` on dataclasses
nothing mutates, a regex group number where the pattern has only one group,
and the exact 70-character negation window (70 vs 71 is not a clinical
rule).

Reading the survivors found four real defects, all fixed with tests:

- **"Pneumothorax is not excluded" raised no alert.** "is not" matched a
  denial; hedges after the term now count, and a hedge beats a denial.
- **An accepted AI cleanup replaced dates and IDs with placeholders.**
- **"semi-colon" became "semi-:"**: the colon rule ran first.
- **"new line" / "new paragraph" left stray spaces** and an uncapitalised
  line; "open bracket" left a space; "hyphen" did not join the words.

And untested code that now is: audio de-identification (silences around a
spoken name), month-name dates, every critical term and every denial/hedge
cue, every communication-statement and audit shape.
