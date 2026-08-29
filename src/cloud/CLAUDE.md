# src/cloud — Cloud fine-tuning path (opt-in, off by default)

Follows the root [`CLAUDE.md`](../../CLAUDE.md) — this covers only this subsystem.

Inert unless **both** `cloud_enabled` and `cloud_training_consent` are true in
settings. With default settings nothing is retained, staged, or uploaded.

```
dictation corrections → training/collector.py (consent-gated capture)
   → medical/deid.py  de-identify TEXT + AUDIO, validate_clean()
   → training/staging_db.py  (SQLite: data/training/staging.db)
   → cloud/tasks/<task>.build_archive()  tar.gz batch (manifest + de-id clips)
   → cloud/framework/client.py  upload + submit Lightning AI job (key from keychain)
   → cloud/framework/job_monitor.py polls → framework/sync.py downloads + extracts
   → cloud/framework/registry.py registers the version (data/models/registry.json)
   → user activates it; the consuming model loads that artifact directory
```

`collector.py` is the **only** bridge from dictation into the cloud subsystem,
and the dependency is one-directional: dictation/features call into the
collector but never import `src.cloud.*`.

**Multi-task framework.** The framework is task-agnostic; a `TrainingTask`
(`cloud/tasks/`) supplies only what differs per model — how to bundle its batch
and how to describe its Lightning job (`JobSpec`). Three tasks exist: voice
(Whisper LoRA → CT2), text-correction (small seq2seq), and scan-classifier
(DenseNet head fine-tune). Each is keyed by `task_type`; the registry holds one
active model **per task**. Every training script enforces a *do-not-regress*
gate on a held-out split (WER for voice, exact-match for text, mean-AUC for
scans) and rejects a run that doesn't beat its validated baseline.
