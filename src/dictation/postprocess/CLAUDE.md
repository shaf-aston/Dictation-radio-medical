# src/dictation/postprocess: the correction pipeline

Loads only when work touches `src/dictation/postprocess/`. The always-loaded
rules for this project are in the root [CLAUDE.md](../../../CLAUDE.md).
Any accuracy change is judged by the harness in `scripts/eval/`, never by
reading a sample.

The pipeline runs `pipeline.pipeline_stages()` in order: hallucinations →
correction commands → spoken punctuation → scratch-that → whitespace →
measurements → terminology → UK spelling → accent → split-compound rejoin →
fuzzy dictionary → context real-word fix → learned corrections → insert-phrase
→ capitalisation. The pipeline only sequences them; the stages live in their
own modules (some outside this folder: `medical/spelling.py`,
`features/adaptive_learning.py`). Cleanup level "soft" skips the guessing
stages (`_SOFT_SKIP_STAGES`).

The fuzzy stage (`medical_dict_match.py`) is where mis-transcribed medical terms
get fixed, and it leans on **two** wordlists with distinct jobs (`medical_dict.py`):
the broad generic list answers *"is this already a real word? leave it alone"*
(membership). A confirmed non-word is snapped to the nearest term in a SymSpell
index over that whole set, within a length-scaled edit distance (`_max_edits`),
with the **curated `radiology_lexicon.txt`** boosted so a tie prefers the
radiology spelling ("efusion" → "effusion", not "fusion"); a tie the boost
cannot break is left alone. The English-word guard (`is_english_word`) and the
inflection, derivation and British-spelling guards are what stop junk snaps. To
improve correction of a term, add it to the lexicon (the spelling authority) or
add a precise rule to `corrections.yaml`.
Never feed PDF/OCR-extracted text into the lexicon: extraction noise (ligature
splits, hyphenation artefacts) pollutes the snap targets.
