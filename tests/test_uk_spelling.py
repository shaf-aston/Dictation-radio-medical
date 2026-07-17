"""Regression net for British/American medical-spelling handling.

Two directions, both safety-relevant because radiology reports must use one
consistent spelling convention:

* Stage 5 (``terminology``) normalises American-spelled Whisper output
  ("hemoglobin", "esophageal") to the project's British convention
  ("haemoglobin", "oesophageal") before the fuzzy stage runs.
* Stage 7 (``medical_dict_match``) must never *reverse* that — a correctly
  spelled British term must not be fuzzy-"corrected" to its American
  counterpart just because the American spelling is the one present in the
  ~98k-term wordlist.
"""

from __future__ import annotations

from src.dictation.postprocess.terminology import apply_terminology


# American -> expected British spelling after Stage 5.
_US_TO_UK = {
    "esophageal": "oesophageal",
    "esophagus": "oesophagus",
    "esophagitis": "oesophagitis",
    "anesthetic": "anaesthetic",
    "anesthesia": "anaesthesia",
    "leukemia": "leukaemia",
    "leukemic": "leukaemic",
    "diarrhea": "diarrhoea",
    "fetus": "foetus",
    "fetal": "foetal",
    "hemoglobin": "haemoglobin",
    "cesarean": "caesarean",
    "cesarian": "caesarean",
    "gynecology": "gynaecology",
    "gynecological": "gynaecological",
    "estrogen": "oestrogen",
    "hemodynamic": "haemodynamic",
    "hemodynamically": "haemodynamically",
    "hemothorax": "haemothorax",
    "hemoptysis": "haemoptysis",
    "fecal": "faecal",
    "feces": "faeces",
    "ischemia": "ischaemia",
    "ischemic": "ischaemic",
    "edematous": "oedematous",
    "anemic": "anaemic",
    "hemorrhage": "haemorrhage",
    "hemorrhages": "haemorrhages",
    "hemorrhagic": "haemorrhagic",
    "hematological": "haematological",
    "hematology": "haematology",
    "hemostasis": "haemostasis",
    "hemostatic": "haemostatic",
}


def test_american_spelling_normalised_to_british():
    for us, uk in _US_TO_UK.items():
        out = apply_terminology(f"Findings show {us} change.")
        assert uk in out, f"{us!r} -> expected {uk!r}, got {out!r}"


def test_british_spelling_left_unchanged():
    for uk in set(_US_TO_UK.values()):
        sentence = f"Findings show {uk} change."
        assert apply_terminology(sentence) == sentence
