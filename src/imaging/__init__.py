"""Local scan-assistant subsystem — offline chest X-ray analysis.

Given a chest X-ray, this subsystem surfaces candidate findings **only when the
model is confident**, and always shows *where* it is looking (a Grad-CAM region
overlay) so the radiologist can judge the evidence. It is deliberately
conservative: below a calibrated per-pathology threshold the tool says nothing
rather than risk an over-confident, misleading suggestion.

Mirrors the ``src.dictation`` shape: a local pipeline with NO hard cloud
dependency. Inference runs on validated pretrained weights (TorchXRayVision
DenseNet121); cloud fine-tuning is a separate, opt-in extension. Heavy ML deps
(torch, torchxrayvision, skimage) are lazy-imported so the rest of the app stays
light and importable without them.

Pipeline (orchestrated by :mod:`src.imaging.analyzer`):
    image → classifier (probabilities) → abstention (keep only confident,
    trained labels) → localization (Grad-CAM region per kept finding) → result.

SAFETY: outputs are assistive suggestions for radiologist review, never a
diagnosis. Every surfaced finding carries a region overlay; if localization
fails, the finding is withheld.
"""

from __future__ import annotations
