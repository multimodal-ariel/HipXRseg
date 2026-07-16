#!/usr/bin/env python3
"""
Train 2D nnU-Net hip bone segmentation (Dataset802).

Pipeline:
  1) prepare    — DICOM + COCO → exclusive multi-class NIfTI + offline hflip/L-R swap
  2) preprocess — nnU-Net plan_and_preprocess (batch_size=8)
  3) train      — folds 0–3 CV with nnUNetTrainerNoMirrorEarlyStop (no online mirroring)
                 Inference uses checkpoint_best.pth only.

Does not modify raw DICOMs / instances_default.json under rawdata/.

Usage (medcv env):
  python train.py --stage prepare
  python train.py --stage preprocess
  python train.py --stage train --waves
  python train.py --stage all --waves
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "rawdata"
WS = ROOT / "nnunet_workspace_v2"
NNUNET_RAW = WS / "nnUNet_raw"
NNUNET_PRE = WS / "nnUNet_preprocessed"
NNUNET_RES = WS / "nnUNet_results"
DATASET_ID = 802
DATASET_NAME = f"Dataset{DATASET_ID:03d}_HipBones_Aug"
DATASET_RAW = NNUNET_RAW / DATASET_NAME
CONFIG = "2d"
TRAINER = "nnUNetTrainerNoMirrorEarlyStop"
PLANS = "nnUNetPlans"
PY = sys.executable
SEED = 42
N_TEST = 6

LABEL_NAMES = {
    0: "background",
    1: "Left Innominate Bone",
    2: "Right Innominate Bone",
    3: "Left Femur",
    4: "Right Femur",
}
CAT_ORDER = [1, 2, 3, 4]
LR_SWAP = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}

SITE_TRAINER = (
    Path(PY).resolve().parent.parent
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages/nnunetv2/training/nnUNetTrainer/variants/training_length"
    / "nnUNetTrainerNoMirrorEarlyStop.py"
)


DEFAULT_FOLDS = (0, 1, 2, 3)


def set_env():
    for p in (NNUNET_RAW, NNUNET_PRE, NNUNET_RES):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PRE)
    os.environ["nnUNet_results"] = str(NNUNET_RES)
    os.environ.setdefault("nnUNet_n_proc_DA", "4")
    os.environ.setdefault("nnUNet_compile", "f")


def install_trainer():
    src = ROOT / "nnUNetTrainerNoMirrorEarlyStop.py"
    SITE_TRAINER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, SITE_TRAINER)
    print("Installed", SITE_TRAINER)


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
    return np.frombuffer(data, dtype=dtype).reshape(rows, cols)


def decode_rle(segmentation: dict) -> np.ndarray:
    h, w = segmentation["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for count in segmentation["counts"]:
        flat[idx : idx + count] = val
        idx += count
        val ^= 1
    return flat.reshape((h, w), order="F")


def build_label_map(anns: list, height: int, width: int) -> np.ndarray:
    lab = np.zeros((height, width), dtype=np.uint8)
    by_cat = {a["category_id"]: a for a in anns}
    for cid in CAT_ORDER:
        if cid not in by_cat:
            continue
        m = decode_rle(by_cat[cid]["segmentation"]).astype(bool)
        lab[m] = cid
    return lab


def save_nifti_2d(arr: np.ndarray, path: Path, dtype=None) -> None:
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    img.SetSpacing((1.0, 1.0))
    sitk.WriteImage(img, str(path))


def hflip_swap(img: np.ndarray, lab: np.ndarray):
    img_f = np.ascontiguousarray(img[:, ::-1])
    lab_f = np.ascontiguousarray(lab[:, ::-1])
    lab_s = np.zeros_like(lab_f)
    for a, b in LR_SWAP.items():
        lab_s[lab_f == a] = b
    return img_f, lab_s


def stage_prepare():
    """Build Dataset802 from raw DICOMs + COCO, with offline LR-safe flip aug."""
    set_env()
    install_trainer()

    coco = json.loads((RAW_DIR / "instances_default.json").read_text())
    images = sorted(coco["images"], key=lambda x: x["file_name"])
    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    stems = [Path(im["file_name"]).stem for im in images]
    assert len(stems) == 26, len(stems)

    # Prefer existing held-out split for reproducibility
    split_path = WS / "split_aug.json"
    if split_path.exists():
        meta_old = json.loads(split_path.read_text())
        train, test = meta_old["base_train"], meta_old["test"]
        print("Reusing split from", split_path)
    else:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(stems))
        test_idx = set(perm[:N_TEST].tolist())
        test = [stems[i] for i in sorted(test_idx)]
        train = [s for i, s in enumerate(stems) if i not in test_idx]

    if DATASET_RAW.exists():
        shutil.rmtree(DATASET_RAW)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (DATASET_RAW / sub).mkdir(parents=True)

    id_by_stem = {Path(im["file_name"]).stem: im for im in images}

    def convert_one(stem: str, split_tag: str, out_stem: str | None = None):
        out_stem = out_stem or stem
        im = id_by_stem[stem]
        dcm = RAW_DIR / im["file_name"]
        if not dcm.exists():
            dcm = RAW_DIR / f"{stem}.dcm"
        pix = read_dicom_pixels(dcm).astype(np.float32)
        lab = build_label_map(anns_by_image[im["id"]], im["height"], im["width"])
        assert pix.shape == lab.shape, (stem, pix.shape, lab.shape)
        save_nifti_2d(pix, DATASET_RAW / f"images{split_tag}" / f"{out_stem}_0000.nii.gz")
        save_nifti_2d(lab, DATASET_RAW / f"labels{split_tag}" / f"{out_stem}.nii.gz", np.uint8)
        return pix, lab

    print("Converting train/test from DICOM + COCO...")
    for s in train:
        convert_one(s, "Tr")
    for s in test:
        convert_one(s, "Ts")

    print("Offline horizontal-flip + L/R label-swap (train only)...")
    aug_stems = []
    for s in train:
        img = sitk.GetArrayFromImage(
            sitk.ReadImage(str(DATASET_RAW / "imagesTr" / f"{s}_0000.nii.gz"))
        ).squeeze()
        lab = sitk.GetArrayFromImage(
            sitk.ReadImage(str(DATASET_RAW / "labelsTr" / f"{s}.nii.gz"))
        ).squeeze()
        img_f, lab_f = hflip_swap(img, lab)
        out = f"{s}_hflip"
        save_nifti_2d(img_f.astype(np.float32), DATASET_RAW / "imagesTr" / f"{out}_0000.nii.gz")
        save_nifti_2d(lab_f.astype(np.uint8), DATASET_RAW / "labelsTr" / f"{out}.nii.gz", np.uint8)
        aug_stems.append(out)

    n_train = len(train) + len(aug_stems)
    dataset_json = {
        "channel_names": {"0": "Xray"},
        "labels": {LABEL_NAMES[i]: i for i in range(5)},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (DATASET_RAW / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    meta = {
        "seed": SEED,
        "base_train": train,
        "test": test,
        "offline_aug": aug_stems,
        "n_train_total": n_train,
        "notes": [
            "Offline aug: horizontal flip + label swap 1↔2, 3↔4",
            "Online: nnU-Net DA without mirroring (nnUNetTrainerNoMirrorEarlyStop)",
            "Exclusive multi-class labels; femur overwrites innominate at overlaps",
        ],
    }
    (WS / "split_aug.json").write_text(json.dumps(meta, indent=2))
    print(f"Dataset802 ready: {n_train} train, {len(test)} test → {DATASET_RAW}")


def stage_preprocess():
    set_env()
    install_trainer()
    cmd = [
        PY,
        "-m",
        "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "-d",
        str(DATASET_ID),
        "-c",
        CONFIG,
        "--verify_dataset_integrity",
    ]
    print("Running", " ".join(cmd))
    subprocess.check_call(cmd, env=os.environ.copy())
    plans_path = NNUNET_PRE / DATASET_NAME / "nnUNetPlans.json"
    plans = json.loads(plans_path.read_text())
    plans["configurations"]["2d"]["batch_size"] = 8
    plans_path.write_text(json.dumps(plans, indent=2))
    print("Set 2d batch_size=8")


def stage_train(folds=None, sequential_waves=False):
    set_env()
    install_trainer()
    folds = list(DEFAULT_FOLDS) if folds is None else folds

    from nnunetv2.training.dataloading.utils import unpack_dataset

    pre = NNUNET_PRE / DATASET_NAME / f"{PLANS}_{CONFIG}"
    if len(list(pre.glob("*_seg.npy"))) < 10:
        unpack_dataset(str(pre), unpack_segmentation=True, overwrite_existing=False, num_processes=1)

    log_dir = WS / "train_logs"
    log_dir.mkdir(exist_ok=True)

    def launch(fold):
        log = log_dir / f"fold_{fold}.log"
        cmd = [
            PY,
            "-m",
            "nnunetv2.run.run_training",
            str(DATASET_ID),
            CONFIG,
            str(fold),
            "-tr",
            TRAINER,
            "-p",
            PLANS,
            "-device",
            "cuda",
        ]
        print("Launch", " ".join(cmd))
        fh = open(log, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh, log

    if sequential_waves:
        for i in range(0, len(folds), 2):
            wave = folds[i : i + 2]
            handles = [launch(f) for f in wave]
            for p, fh, lp in handles:
                r = p.wait()
                fh.close()
                if r:
                    raise RuntimeError(lp)
    else:
        handles = [launch(f) for f in folds]
        rc = 0
        for p, fh, lp in handles:
            r = p.wait()
            fh.close()
            print(lp.name, "exit", r)
            rc = rc or r
        if rc:
            raise RuntimeError("fold failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "preprocess", "train"])
    ap.add_argument("--folds", default="0,1,2,3", help="folds to train (fold 4 dropped)")
    ap.add_argument("--waves", action="store_true", help="train 2 folds at a time")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    t0 = time.time()
    if args.stage in ("all", "prepare"):
        stage_prepare()
    if args.stage in ("all", "preprocess"):
        stage_preprocess()
    if args.stage in ("all", "train"):
        stage_train(folds=folds, sequential_waves=args.waves)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
