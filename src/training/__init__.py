"""Local training-data capture and staging for Lightning AI fine-tuning.

This package collects the (audio, wrong, correct) correction triples produced
during normal dictation, de-identifies them, and stages them in a local SQLite
database until enough have accumulated to justify a cloud training run.

The package is import-safe with no cloud dependency: if the user never enables
cloud training, the collector is a no-op and nothing is persisted.
"""
