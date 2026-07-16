#!/usr/bin/env python3
"""16:9 presentation summary — absolute layout, no overlaps."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
WS = ROOT / "nnunet_workspace_v2"
RAW = WS / "nnUNet_raw" / "Dataset802_HipBones_Aug"
MODEL = (
    WS
    / "nnUNet_results/Dataset802_HipBones_Aug/nnUNetTrainerNoMirrorEarlyStop__nnUNetPlans__2d"
)
OUT = ROOT / "visualizations" / "summary_16x9_slide.png"
OUT2 = ROOT / "visualizations" / "summary_16x9.png"
CURVES = Path("/tmp/hip_summary_curves.json")
CACHE = Path("/tmp/hip_summary_viz_cache.npz")

os.environ["nnUNet_raw"] = str(WS / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(WS / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(WS / "nnUNet_results")

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

BG = "#F2F1EE"
PANEL = "#FFFFFF"
INK = "#181818"
MUTED = "#5A5A5A"
RULE = "#C8C4BC"
ACCENT = "#1F6F8B"
FOLD_C = {0: "#1F6F8B", 1: "#C46B2B", 2: "#2F7D4A", 3: "#6B5B7A"}
CLS_C = {
    1: np.array([0.85, 0.22, 0.22]),
    2: np.array([0.18, 0.45, 0.80]),
    3: np.array([0.18, 0.65, 0.30]),
    4: np.array([0.90, 0.68, 0.12]),
}
CLS_N = {1: "L Innominate", 2: "R Innominate", 3: "L Femur", 4: "R Femur"}
CASES = ["EECC14AD", "EE4559F3", "EEEAA8F5"]


def display(img):
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay(gray, lab, a=0.45):
    rgb = np.stack([gray, gray, gray], -1).copy()
    for c, col in CLS_C.items():
        m = lab == c
        if m.any():
            rgb[m] = (1 - a) * rgb[m] + a * col
    return np.clip(rgb, 0, 1)


def panel(fig, rect, title=None):
    """Create a white panel with optional title. rect = [l,b,w,h] in fig coords."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(RULE)
        s.set_linewidth(0.9)
    if title:
        ax.set_title(title, loc="left", fontsize=10, fontweight="600", color=INK, pad=4)
    return ax


def load_viz(metrics):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if set(z["stems"].tolist()) == set(CASES):
            return {
                s: {
                    "img": z[f"{s}_img"],
                    "pred": z[f"{s}_pred"],
                    "gt": z[f"{s}_gt"],
                    "mean": float(z[f"{s}_mean"]),
                }
                for s in CASES
            }
    pred_model = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False, allow_tqdm=False,
    )
    pred_model.initialize_from_trained_model_folder(
        str(MODEL), use_folds=(0, 1, 2, 3), checkpoint_name="checkpoint_best.pth"
    )
    rw = SimpleITKIO()
    viz, save = {}, {"stems": np.array(CASES, dtype=object)}
    for stem in CASES:
        data, props = rw.read_images([[str(RAW / "imagesTs" / f"{stem}_0000.nii.gz")]])
        pr = np.asarray(pred_model.predict_single_npy_array(data, props, None, None, False)).squeeze()
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / "imagesTs" / f"{stem}_0000.nii.gz"))).squeeze().astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / "labelsTs" / f"{stem}.nii.gz"))).squeeze().astype(np.uint8)
        mean = next(c["mean_fg"] for c in metrics["cases"] if c["case"] == stem)
        viz[stem] = {"img": img, "pred": pr, "gt": gt, "mean": mean}
        save[f"{stem}_img"] = img
        save[f"{stem}_pred"] = pr
        save[f"{stem}_gt"] = gt
        save[f"{stem}_mean"] = mean
    np.savez_compressed(CACHE, **save)
    return viz


def main():
    metrics = json.loads((ROOT / "visualizations" / "test_metrics_clean.json").read_text())
    curves = json.loads(CURVES.read_text())
    viz = load_viz(metrics)
    mean_fg = float(metrics["mean_fg_dice"])

    fig = plt.figure(figsize=(16, 9), dpi=240, facecolor=BG)

    # ----------------------------------------------------------
    # Absolute layout (figure fractions). Gaps are intentional.
    # Header  y: 0.915–0.985
    # Mid     y: 0.575–0.895   (4 panels, gap 0.012)
    # Viz     y: 0.055–0.545
    # Footer  y: 0.010–0.045
    # ----------------------------------------------------------

    # ===== HEADER =====
    ax_h = fig.add_axes([0.02, 0.915, 0.96, 0.07])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1)
    ax_h.text(0, 0.70, "Hip Bone Segmentation", fontsize=20, fontweight="700", color=INK, va="center")
    ax_h.text(
        0, 0.18,
        "2D nnU-Net · Dataset802 · Exclusive multi-class · Ensemble folds 0–3 (checkpoint_best)",
        fontsize=9, color=MUTED, va="center",
    )
    kpis = [("26", "studies"), ("20 / 6", "train / test"), ("40", "train + aug"), (f"{mean_fg:.3f}", "test Dice")]
    for i, (v, lab) in enumerate(kpis):
        x = 0.58 + i * 0.105
        if i:
            ax_h.axvline(x - 0.012, ymin=0.12, ymax=0.88, color=RULE, lw=0.8)
        ax_h.text(x, 0.70, v, fontsize=15, fontweight="700", color=ACCENT, va="center")
        ax_h.text(x, 0.18, lab, fontsize=8, color=MUTED, va="center")

    # ===== MID PANELS (explicit x positions — no shared overflow) =====
    y_m, h_m = 0.568, 0.320
    # tighter gaps, fuller width: 0.008 between panels
    panels_x = [
        (0.018, 0.235),  # raw
        (0.261, 0.210),  # aug
        (0.479, 0.312),  # train
        (0.799, 0.183),  # test
    ]

    # --- RAW ---
    ax_r = panel(fig, [panels_x[0][0], y_m, panels_x[0][1], h_m], "Raw data")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)
    ax_r.axis("off")
    for i, t in enumerate([
        "26 DICOM · COCO RLE",
        "4 classes · femur > innominate",
        "H 1409–3032 · W 1205–3408",
        "Seed 42 → 20 train / 6 test",
    ]):
        ax_r.text(0.05, 0.88 - i * 0.11, t, fontsize=8, color=INK, va="center")

    # bar chart inset fully inside raw panel (dedicated lower half)
    ax_rb = fig.add_axes([
        panels_x[0][0] + 0.012,
        y_m + 0.018,
        panels_x[0][1] - 0.024,
        h_m * 0.42,
    ])
    ax_rb.set_facecolor(PANEL)
    for s in ax_rb.spines.values():
        s.set_color(RULE)
        s.set_linewidth(0.6)
    xs = np.arange(4)
    counts = [25, 26, 18, 19]
    ax_rb.bar(xs, counts, color=[CLS_C[c] for c in [1, 2, 3, 4]], width=0.68, zorder=2)
    ax_rb.set_xticks(xs)
    ax_rb.set_xticklabels(["LI", "RI", "LF", "RF"], fontsize=7)
    ax_rb.set_ylim(0, 34)
    ax_rb.set_yticks([0, 15, 30])
    ax_rb.tick_params(labelsize=6.5, colors=MUTED, length=2, pad=1)
    ax_rb.set_title("Label presence", fontsize=8, color=MUTED, loc="left", pad=2)
    ax_rb.yaxis.grid(True, color=RULE, lw=0.5, zorder=0)
    ax_rb.set_axisbelow(True)
    for x, v in zip(xs, counts):
        ax_rb.text(x, v + 1.2, str(v), ha="center", va="bottom", fontsize=7, color=INK, fontweight="600")

    # --- AUG ---
    ax_a = panel(fig, [panels_x[1][0], y_m, panels_x[1][1], h_m], "After augmentation")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    rows = [
        ("Offline", "H-flip + L↔R swap"),
        ("Swap", "1↔2 · 3↔4"),
        ("Train", "20 → 40 volumes"),
        ("Test", "6 · unaugmented"),
        ("Online", "nnU-Net, no mirror"),
        ("Trainer", "NoMirrorEarlyStop"),
        ("Kept", "folds 0–3 · best"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.90 - i * 0.105
        ax_a.text(0.05, y, k, fontsize=7.5, color=MUTED, va="center")
        ax_a.text(0.34, y, v, fontsize=8, color=INK, va="center", fontweight="500")

    # --- TRAIN CURVES ---
    # leave bottom strip inside panel for fold keys (not a separate overlapping axes)
    ax_t = panel(fig, [panels_x[2][0], y_m, panels_x[2][1], h_m], "Training · val mean Dice")
    for f in [0, 1, 2, 3]:
        cv = curves[str(f)]
        ax_t.plot(cv["epochs"], cv["mean_dice"], color=FOLD_C[f], lw=1.5, zorder=3)
    ax_t.set_xlim(0, 199)
    ax_t.set_ylim(0.72, 1.01)
    ax_t.set_xlabel("Epoch", fontsize=7.5, color=MUTED, labelpad=2)
    ax_t.set_ylabel("Dice", fontsize=7.5, color=MUTED, labelpad=4)
    ax_t.tick_params(labelsize=6.5, colors=MUTED, length=2, pad=1)
    ax_t.yaxis.grid(True, color=RULE, lw=0.5, zorder=0)
    ax_t.set_axisbelow(True)
    ax_t.set_position([
        panels_x[2][0] + 0.008,
        y_m + 0.068,
        panels_x[2][1] - 0.016,
        h_m - 0.092,
    ])
    # fold key strip under plot (clear of xlabel)
    ax_tk = fig.add_axes([panels_x[2][0] + 0.008, y_m + 0.010, panels_x[2][1] - 0.016, 0.036])
    ax_tk.set_facecolor(PANEL)
    ax_tk.set_xlim(0, 4)
    ax_tk.set_ylim(0, 1)
    ax_tk.axis("off")
    for s in ax_tk.spines.values():
        s.set_visible(False)
    for i, f in enumerate([0, 1, 2, 3]):
        x = i + 0.08
        ax_tk.plot([x, x + 0.14], [0.5, 0.5], color=FOLD_C[f], lw=2.2, solid_capstyle="round")
        ax_tk.text(x + 0.18, 0.5, f"F{f} {curves[str(f)]['final_best_ema']:.3f}",
                   fontsize=7.2, color=INK, va="center", ha="left", fontweight="500")

    # --- TEST BARS ---
    ax_e = panel(fig, [panels_x[3][0], y_m, panels_x[3][1], h_m], f"Held-out test · mean {mean_fg:.3f}")
    pc = metrics["per_class_mean_dice"]
    keys = ["L innominate", "R innominate", "L femur", "R femur"]
    labs = ["L Innominate", "R Innominate", "L Femur", "R Femur"]
    vals = [pc[k] for k in keys]
    yp = np.arange(4)[::-1]
    ax_e.set_position([
        panels_x[3][0] + 0.008,
        y_m + 0.040,
        panels_x[3][1] - 0.052,
        h_m - 0.058,
    ])
    ax_e.barh(yp, vals, color=[CLS_C[c] for c in [1, 2, 3, 4]], height=0.55, zorder=2)
    ax_e.set_yticks(yp)
    ax_e.set_yticklabels(labs, fontsize=7.5)
    ax_e.set_xlim(0.97, 1.0)
    ax_e.set_xticks([0.97, 0.98, 0.99, 1.00])
    ax_e.set_xlabel("Mean Dice", fontsize=7.5, color=MUTED, labelpad=1)
    ax_e.tick_params(labelsize=6.5, colors=MUTED, length=2, pad=1)
    ax_e.axvline(mean_fg, color=INK, ls="--", lw=0.85, zorder=3)
    ax_e.xaxis.grid(True, color=RULE, lw=0.5, zorder=0)
    ax_e.set_axisbelow(True)
    ax_ev = fig.add_axes([
        panels_x[3][0] + panels_x[3][1] - 0.042,
        y_m + 0.040,
        0.038,
        h_m - 0.058,
    ])
    ax_ev.set_facecolor(PANEL)
    ax_ev.set_xlim(0, 1)
    ax_ev.set_ylim(-0.5, 3.5)
    ax_ev.axis("off")
    for y, v in zip(yp, vals):
        ax_ev.text(0.0, y, f"{v:.3f}", fontsize=7.5, color=INK, va="center", ha="left", fontweight="600")

    # ===== EXAMPLES =====
    ax_vs = fig.add_axes([0.018, 0.538, 0.25, 0.020])
    ax_vs.set_facecolor(BG)
    ax_vs.axis("off")
    ax_vs.text(0, 0.4, "Held-out examples", fontsize=10, fontweight="600", color=INK, va="center")

    y_top, y_bot = 0.528, 0.052
    gap_y = 0.006
    row_h = (y_top - y_bot - 2 * gap_y) / 3
    x_lab, w_lab = 0.016, 0.078
    x0, x1, gap_x = 0.100, 0.985, 0.006
    col_w = (x1 - x0 - 2 * gap_x) / 3
    titles = ["Input", "Prediction", "Ground truth"]

    for i, stem in enumerate(CASES):
        y = y_top - (i + 1) * row_h - i * gap_y
        ax_l = fig.add_axes([x_lab, y, w_lab, row_h])
        ax_l.set_facecolor(BG)
        ax_l.axis("off")
        ax_l.text(0.9, 0.62, stem, fontsize=8.5, fontweight="700", color=INK, ha="right", va="center")
        ax_l.text(0.9, 0.28, f"{viz[stem]['mean']:.3f}", fontsize=8.5, color=ACCENT, ha="right", va="center", fontweight="700")

        g = display(viz[stem]["img"])
        imgs = [g, overlay(g, viz[stem]["pred"]), overlay(g, viz[stem]["gt"])]
        for j, im in enumerate(imgs):
            x = x0 + j * (col_w + gap_x)
            axp = fig.add_axes([x, y, col_w, row_h])
            if im.ndim == 2:
                axp.imshow(im, cmap="gray", aspect="auto", interpolation="bilinear")
            else:
                axp.imshow(im, aspect="auto", interpolation="bilinear")
            axp.set_xticks([])
            axp.set_yticks([])
            for s in axp.spines.values():
                s.set_color(RULE)
                s.set_linewidth(0.7)
            if i == 0:
                axp.set_title(titles[j], fontsize=9, fontweight="600", color=INK, pad=1.5)

    # ===== FOOTER =====
    ax_f = fig.add_axes([0.02, 0.008, 0.96, 0.038])
    ax_f.set_facecolor(BG)
    ax_f.axis("off")
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CLS_C[c], markersize=7, label=CLS_N[c])
        for c in [1, 2, 3, 4]
    ]
    ax_f.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=8.5, columnspacing=1.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=240, facecolor=BG)
    fig.savefig(OUT2, dpi=240, facecolor=BG)
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
