# src/imaging: Scan assistant path (opt-in, local inference)

Follows the root [`CLAUDE.md`](../../CLAUDE.md): this covers only this subsystem.

`src/imaging/` analyses a chest X-ray locally and surfaces findings **only when
confident and localisable**. Pipeline: `classifier` (TorchXRayVision DenseNet121)
→ `abstention` (drop untrained heads + anything below the calibrated
per-pathology threshold) → `localization` (Grad-CAM region per kept finding) →
`analyzer` (withhold any finding it can't point to; render an overlay; attach
the non-diagnostic disclaimer). Needs the optional imaging extra
(`scripts/lightning/requirements_imaging.txt`); without it the feature degrades
gracefully and the rest of the app is unaffected.

**Specialty focus: chest trauma.** Rib/clavicle fractures correlate clinically
with pneumothorax and haemothorax, and plain films are documented to miss a
large share of rib fractures: the highest-leverage gap for this assistant to
close. The local fine-tune (`scan_classifier` task, `ScanClassifierTask` in
`cloud/tasks/scan_finetune.py`) oversamples `Fracture`, `Pneumothorax`, and
`Effusion` positives (`_TRAUMA_FOCUS_LABELS`) when building a batch, so a site's
fine-tune sharpens on trauma cases rather than diluting evenly across all 18
baseline labels. This changes *what the model trains on*, not the safety gate:
per-label abstention thresholds remain the calibration lever for a site's own
validated data (`data/imaging/thresholds.json`).
