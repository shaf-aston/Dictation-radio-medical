# src/dictation/postprocess: the correction pipeline

Loads only when work touches `src/dictation/postprocess/`. The always-loaded
rules for this project are in the root [CLAUDE.md](../../../CLAUDE.md).
Any accuracy change is judged by the harness in `scripts/eval/`, never by
reading a sample.

The pipeline (`dictation/postprocess/pipeline.py`) runs, in order: hallucination
removal → voice commands → punctuation → measurements → terminology →
accent-specific → fuzzy medical-dictionary match → learned corrections →
capitalization. Each stage owns one file; the pipeline only sequences them.

The fuzzy stage (`medical_dict_match.py`) is where mis-transcribed medical terms
get fixed, and it leans on **two** wordlists with distinct jobs (`medical_dict.py`):
the broad generic list answers *"is this already a real word? leave it alone"*
(membership), while the **curated `radiology_lexicon.txt`** is the only thing a
typo is *snapped to* (correction targets). Keeping snap targets radiology-only is
what stops a misspelling from being pulled toward the generic list's chemistry /
drug / obscure-procedure junk. To improve correction of a term, add it to the
lexicon (the spelling authority) or add a precise rule to `corrections.yaml`.
Never feed PDF/OCR-extracted text into the lexicon: extraction noise (ligature
splits, hyphenation artefacts) pollutes the snap targets.
