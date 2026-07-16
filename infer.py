#!/usr/bin/env python3
"""
Inference for hip bone 2D nnU-Net (Dataset802 ensemble).

Accepts a case ID from imagesTs/Tr, a NIfTI, or a DICOM. Writes prediction
NIfTI (+ optional overlay PNG). Ensembles available fold checkpoints
(folds 0–3 with checkpoint_best.pth by default).

Usage (medcv env):
  python infer.py --case EE4559F3
  python infer.py --input path/to/img_0000.nii.gz --output pred.nii.gz
  python infer.py --input path/to/scan.dcm --output pred.nii.gz --overlay out.png
  python infer.py --case EE4559F3 --gt   # also score vs labelsTs if present
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "rawdata"
WS = ROOT / "nnunet_workspace_v2"
RAW = WS / "nnUNet_raw" / "Dataset802_HipBones_Aug"
MODEL = (
    WS
    / "nnUNet_results/Dataset802_HipBones_Aug/nnUNetTrainerNoMirrorEarlyStop__nnUNetPlans__2d"
)
CHECKPOINT = "checkpoint_best.pth"
# Trained / kept folds (fold 4 removed)
DEFAULT_FOLDS = (0, 1, 2, 3)

os.environ["nnUNet_raw"] = str(WS / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(WS / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(WS / "nnUNet_results")

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

COLORS = {
    1: (1.0, 0.2, 0.2),
    2: (0.2, 0.55, 1.0),
    3: (0.15, 0.85, 0.3),
    4: (1.0, 0.75, 0.1),
}
NAMES = {1: "L innominate", 2: "R innominate", 3: "L femur", 4: "R femur"}


def _find_us(raw: bytes, tag: bytes) -> int:
    i = 0
    while True:
        i = raw.find(tag, i)
        if i < 0:
            raise KeyError(tag)
        if raw[i + 4 : i + 6] == b"US":
            return int.from_bytes(raw[i + 8 : i + 10], "little")
        i += 1


def read_dicom_pixels(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    rows = _find_us(raw, b"\x28\x00\x10\x00")
    cols = _find_us(raw, b"\x28\x00\x11\x00")
    bits = _find_us(raw, b"\x28\x00\x00\x01")
    pixrep = _find_us(raw, b"\x28\x00\x03\x01")
    expected = rows * cols * (bits // 8)
    tag = b"\xe0\x7f\x10\x00"
    start = 0
    data = None
    while True:
        i = raw.find(tag, start)
        if i < 0:
            break
        vr = raw[i + 4 : i + 6]
        if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
            length = int.from_bytes(raw[i + 8 : i + 12], "little")
            payload = raw[i + 12 : i + 12 + length]
        else:
            length = int.from_bytes(raw[i + 6 : i + 8], "little")
            payload = raw[i + 8 : i + 8 + length]
        if length == expected:
            data = payload
            break
        start = i + 1
    if data is None:
        raise RuntimeError(f"PixelData not found: {path}")
    dtype = "<u2" if (bits == 16 and pixrep == 0) else ("<i2" if bits == 16 else "u1")
    return np.frombuffer(data, dtype=dtype).reshape(rows, cols).astype(np.float32)


def to_display(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        hi = lo + 1
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay(gray: np.ndarray, lab: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], -1).copy()
    for c, col in COLORS.items():
        m = lab == c
        if m.any():
            rgb[m] = (1 - alpha) * rgb[m] + alpha * np.asarray(col, np.float32)
    return rgb


def dice(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p, g = pred == c, gt == c
    if p.sum() + g.sum() == 0:
        return float("nan")
    return float(2 * np.logical_and(p, g).sum() / (p.sum() + g.sum()))


def resolve_input(args) -> tuple[np.ndarray, dict, Path | None]:
    """Return (image HxW float32, nnU-Net props, optional gt path)."""
    rw = SimpleITKIO()
    gt_path = None

    if args.case:
        stem = args.case
        for split in ("Ts", "Tr"):
            img_p = RAW / f"images{split}" / f"{stem}_0000.nii.gz"
            if img_p.exists():
                data, props = rw.read_images([[str(img_p)]])
                img = data.squeeze().astype(np.float32)
                cand = RAW / f"labels{split}" / f"{stem}.nii.gz"
                if cand.exists():
                    gt_path = cand
                return img, props, gt_path
        dcm = RAW_DIR / f"{stem}.dcm"
        if dcm.exists():
            img = read_dicom_pixels(dcm)
            props = {
                "sitk_stuff": {
                    "spacing": (1.0, 1.0, 1.0),
                    "origin": (0.0, 0.0, 0.0),
                    "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                },
                "spacing": [1.0, 1.0, 1.0],
            }
            return img, props, None
        raise FileNotFoundError(
            f"Case not found in Dataset802 or as rawdata/{stem}.dcm: {stem}"
        )

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".dcm" or path.name.lower().endswith(".dcm"):
        img = read_dicom_pixels(path)
        props = {
            "sitk_stuff": {
                "spacing": (1.0, 1.0, 1.0),
                "origin": (0.0, 0.0, 0.0),
                "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            },
            "spacing": [1.0, 1.0, 1.0],
        }
        return img, props, None

    data, props = rw.read_images([[str(path)]])
    img = data.squeeze().astype(np.float32)
    return img, props, None


def build_predictor(folds: list[int] | None = None) -> nnUNetPredictor:
    if folds is None:
        folds = [f for f in DEFAULT_FOLDS if (MODEL / f"fold_{f}" / CHECKPOINT).exists()]
    else:
        missing = [f for f in folds if not (MODEL / f"fold_{f}" / CHECKPOINT).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {CHECKPOINT} for folds {missing} under {MODEL}"
            )
    if not folds:
        raise FileNotFoundError(f"No fold checkpoints ({CHECKPOINT}) under {MODEL}")
    print("Ensemble folds:", folds, f"({CHECKPOINT})")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL), use_folds=tuple(folds), checkpoint_name=CHECKPOINT
    )
    return predictor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--case",
        help="Case stem, e.g. EE4559F3 (looks in imagesTs/Tr or rawdata/*.dcm)",
    )
    ap.add_argument("--input", help="Path to NIfTI (*_0000.nii.gz) or DICOM")
    ap.add_argument("--output", help="Output prediction NIfTI path")
    ap.add_argument("--overlay", help="Optional overlay PNG path")
    ap.add_argument("--gt", action="store_true", help="If labels exist for --case, print Dice")
    ap.add_argument(
        "--folds",
        default="",
        help="Comma-separated folds (default: 0,1,2,3 if present)",
    )
    args = ap.parse_args()

    if not args.case and not args.input:
        ap.error("Provide --case or --input")

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""] or None
    img, props, gt_path = resolve_input(args)
    predictor = build_predictor(folds)

    data = img[None, None]  # (C, Z, H, W)
    pred = np.asarray(predictor.predict_single_npy_array(data, props, None, None, False)).squeeze()

    out = Path(args.output) if args.output else (
        ROOT / "predictions" / f"{args.case or Path(args.input).stem}_pred.nii.gz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(pred.astype(np.uint8)), str(out))
    print("Wrote", out)

    metrics = None
    if args.gt and gt_path is not None:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).squeeze()
        dices = {NAMES[c]: dice(pred, gt, c) for c in [1, 2, 3, 4]}
        mean_fg = float(np.nanmean(list(dices.values())))
        metrics = {"mean_fg": mean_fg, "dice": dices}
        print(json.dumps(metrics, indent=2))
    elif args.gt:
        print("No GT found for this input; skipped scoring")

    overlay_path = Path(args.overlay) if args.overlay else None
    if overlay_path is None and args.case:
        overlay_path = out.with_suffix("").with_suffix(".png")  # strip .nii.gz → .png awkwardly
        # fix: pred.nii.gz → pred_overlay.png
        overlay_path = out.parent / (out.name.replace(".nii.gz", "") + "_overlay.png")

    if overlay_path is not None:
        gray = to_display(img)
        n = 3 if (args.gt and gt_path is not None) else 2
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
        if n == 2:
            axes = list(axes)
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("input")
        axes[1].imshow(overlay(gray, pred))
        title = "prediction"
        if metrics:
            title += f"\nmean Dice={metrics['mean_fg']:.3f}"
        axes[1].set_title(title)
        if n == 3:
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).squeeze()
            axes[2].imshow(overlay(gray, gt))
            axes[2].set_title("ground truth")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", overlay_path)


if __name__ == "__main__":
    main()
