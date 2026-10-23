# Plan: reporting release (October 2026)

Requirements: https://claude.ai/artifact/AfxwyS8MycYDHW5WsfkhSv. Each R/E number
below is a requirement on that page. Commits name the number they deliver, so
a requirement can be followed from page → plan → commit → test.

Calls the user made before planning (2026-10-23):

1. Tests come back, for the safety modules only (R7).
2. Patient sex becomes an optional field (R4). The exam side is read from the
   report's own `EXAMINATION:` line, so it needs no field.
3. The user is one radiologist on their own machine. HL7/FHIR, sign/addendum
   and multi-user stay cut (X4, X5, X9).
4. **Approved 2026-10-23**: R6a (hard AI cleanup asks first, passes
   identifiers; the pipeline itself no longer calls the AI) and R6b (drop
   "tendonitis → tendinopathy", and the identical terminology rule for
   "tendinitis"). Locked by `tests/test_no_unseen_rewrites.py`.

## Order, and why

Each step leaves the app working and ships on its own.

| # | Step | Delivers | Why here |
|---|---|---|---|
| 1 | Safety net | R7, R6c | Every later step edits these modules. Tests first. |
| 2 | One definition of a template field | R3 | Smallest change with the largest effect: 1 of 148 fields was visible to the gate. Step 4 reuses it. |
| 3 | Dictate at the cursor | R1 | Templates are useless until dictation lands where you are. Riskiest change, so it goes in alone. |
| 4 | Fields, voice commands, keys | R2, R8, R9, R10 | Built on 2 and 3. |
| 5 | Consistency checks | E3, R4 | Corpus before rules, so precision is measured, not hoped. |
| 6 | Communication record, web audit | R5, R11 | Extends the release gate after step 5 has reshaped it. |
| 7 | Docs | E2 | Last, so they describe what shipped. |

## Design decisions

**A field is any square-bracketed span on one line**
(`template_fields.FIELD_PATTERN`). The old rule only matched SHOUTED names and
argued `[5 mm]` and `[Findings]` were ordinary text. But dictation never makes
a square bracket ("open bracket" gives `(`), and the shipped templates use 12
different field spellings, from `[X]` to `[Right / Left]` to whole sentences
like `[With gadolinium contrast if indicated.]`. A number with a unit (`[5 mm]`)
is still treated as text. The pattern is a plain string that is valid in both
Python and JavaScript, and the web page receives it in its bootstrap payload,
so the two front-ends cannot define a field differently.

**Dictation region has two ends, not one.** Today the dictated region runs from
an anchor to the end of the document, and every cycle rewrites everything after
the anchor. Dictating at the cursor needs an end anchor too: the desktop keeps
two `QTextCursor`s (start stays put on insert, end moves with it); the web keeps
the text before and after the caret. A selected field is replaced. The
"decode once" ledger is untouched: only where its text is drawn changes.

**A pick-list is answered by saying one of its options.** If dictation into
`[Right / Left]` is just "left" (case, spaces and punctuation ignored), the
field becomes `Left`, the option's own spelling. Decided by one pure function
(`template_fields.resolve_choice`), run by the desktop renderer and by the web
server before it sends text, so there is no JavaScript copy of the rule.

**Commands that act on text run in the pipeline; commands that move the cursor
run in the front-end.** "Scratch that" and "insert <phrase>" are text edits, so
they are pipeline stages and both front-ends get them for free. "Next field" is
a key (Tab), bindable to a dictation-mic button (R10). Saying "next field"
*while recording* would need the dictation region to split in two mid-session,
which touches the live loop; it is **deferred** to a later cycle, with the mic
button covering the same action now.

**Checks warn, never block.** Unfilled fields and consistency issues are asked
together, once, before the critical-findings question, and the answer can be
"go back". Critical findings are asked last and never withhold the report, as
today.

**The side check compares against the exam, not against every side word.**
Minn 2015 found side flags 26% precise when every opposite-side word was
flagged. Here a side is checked only when the `EXAMINATION:` line names exactly
one side, and sentences about comparison or the contralateral side are skipped.
Scored on a labelled corpus (E3) before it ships.

**Sex is not an identifier for de-identification.** Adding it to the schema
would make the de-identifier scrub the word "female" from text. The schema now
says which fields identify a person; the de-identifier reads that list.

**The communication record is built by one function.** `report_release`
writes the sentence ("… communicated to Dr X by telephone at 14:20 on
23/10/2026 by Dr Y."), both front-ends insert it and send the same fields to
the audit log.

## How each step is checked

* `pytest` (table-driven: a clinical sentence in, an expected answer out).
* `ruff check src tests` with the rule set pinned in `pyproject.toml`
  (ruff's defaults changed between versions; an unpinned check is not a check).
* Both apps started and the changed path exercised by hand.
* CI runs ruff and the tests on every push (`.github/workflows/ci.yml`).
