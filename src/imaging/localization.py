"""Grad-CAM region localization for surfaced findings.

Interpretability is mandatory for the scan assistant: every finding shown to the
radiologist must come with *where* the model looked, so they can confirm or
dismiss it. This module computes a Grad-CAM heatmap for a given pathology against
the DenseNet's final convolutional block, reduces it to a bounding region, and
renders an overlay PNG.

Grad-CAM is hand-rolled (~one hook + a weighted sum) rather than pulling in
another dependency, keeping the imaging extra lean. Torch/skimage are imported
lazily. If localization fails for a finding, the caller (the analyzer) withholds
that finding — a finding with no shown evidence is worse than no finding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.imaging.schemas import ImagingFinding, Region

logger = logging.getLogger(__name__)

# Fraction of the heatmap's max activation that counts as "inside" the region.
_REGION_ACTIVATION_FRAC = 0.5


def localize(
    classifier, image_path: str, findings: List[ImagingFinding]
) -> Dict[str, Region]:
    """Compute a Grad-CAM bounding region (fractional coords) per finding label.

    Returns ``{label: (x0, y0, x1, y1)}`` for labels that localised successfully;
    labels that fail are simply absent from the map. Best-effort — never raises,
    so a localization failure degrades to "finding withheld", not a crash.
    """
    if not findings:
        return {}
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        logger.warning("Localization skipped (deps missing): %s", exc)
        return {}

    try:
        model = classifier.model
        pathologies = classifier.pathologies
        tensor = classifier._preprocess(image_path)
    except Exception as exc:
        logger.warning("Localization setup failed: %s", exc)
        return {}

    target_layer = _final_conv_layer(model)
    if target_layer is None:
        logger.warning("No conv layer found for Grad-CAM; skipping localization")
        return {}

    regions: Dict[str, Region] = {}
    for finding in findings:
        try:
            idx = pathologies.index(finding.label)
        except ValueError:
            continue
        region = _gradcam_region(model, target_layer, tensor, idx, np, torch)
        if region is not None:
            regions[finding.label] = region
    return regions


def _final_conv_layer(model):
    """Return the last Conv2d module in *model*, the standard Grad-CAM target."""
    import torch.nn as nn
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    return last


def _gradcam_region(model, target_layer, tensor, class_idx, np, torch) -> Optional[Region]:
    """Run Grad-CAM for *class_idx* and reduce the heatmap to a bounding box."""
    activations: list = []
    gradients: list = []

    fwd = target_layer.register_forward_hook(
        lambda _m, _i, out: activations.append(out.detach())
    )
    bwd = target_layer.register_full_backward_hook(
        lambda _m, _gi, go: gradients.append(go[0].detach())
    )
    try:
        model.zero_grad()
        output = model(tensor)
        score = output[0, class_idx]
        score.backward()
        if not activations or not gradients:
            return None
        acts = activations[0][0]          # (C, H, W)
        grads = gradients[0][0]           # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # global-average-pooled gradients
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
        cam_np = cam.cpu().numpy()
    except Exception as exc:
        logger.warning("Grad-CAM failed for class %d: %s", class_idx, exc)
        return None
    finally:
        fwd.remove()
        bwd.remove()

    return _heatmap_to_region(cam_np, np)


def _heatmap_to_region(cam, np) -> Optional[Region]:
    """Reduce a 2-D activation map to a fractional bounding box, or None."""
    if cam.size == 0 or float(cam.max()) <= 0.0:
        return None
    cam = cam / cam.max()
    ys, xs = np.where(cam >= _REGION_ACTIVATION_FRAC)
    if xs.size == 0 or ys.size == 0:
        return None
    h, w = cam.shape
    x0, x1 = xs.min() / w, (xs.max() + 1) / w
    y0, y1 = ys.min() / h, (ys.max() + 1) / h
    return (float(x0), float(y0), float(x1), float(y1))


def render_overlay(
    image_path: str, findings: List[ImagingFinding], out_path: Path
) -> Optional[str]:
    """Draw the finding regions over the image and save a PNG; return its path.

    Best-effort: returns None if rendering deps are missing or drawing fails (the
    analyzer still returns the findings, just without an overlay image).
    """
    boxed = [f for f in findings if f.region is not None]
    if not boxed:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        import skimage.io
    except ImportError as exc:
        logger.warning("Overlay rendering skipped (deps missing): %s", exc)
        return None

    try:
        img = skimage.io.imread(image_path)
        h = img.shape[0]
        w = img.shape[1]
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        for f in boxed:
            x0, y0, x1, y1 = f.region  # type: ignore[misc]
            rect = patches.Rectangle(
                (x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x0 * w, max(0, y0 * h - 5),
                    f"{f.label} {f.probability:.0%}",
                    color="red", fontsize=8, weight="bold")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not render overlay: %s", exc)
        return None
    return str(out_path)
