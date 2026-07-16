## Custom segmentation model for bone segmentation from 2D Hip X-rays

![Summary slide](visualizations/summary_16x9_slide.png)

### Setup

```bash
pip install -r requirements.txt
```

Place raw DICOMs + `instances_default.json` under `rawdata/` (gitignored).

### Checkpoint

Download [`nnUNet_results.zip`](https://github.com/multimodal-ariel/HipXRseg/releases/tag/v1.0.0), then unzip it into `nnunet_workspace_v2/`:

```bash
# from repo root
cd nnunet_workspace_v2
# download nnUNet_results.zip from the release page, then:
unzip nnUNet_results.zip
```

You should end up with:

```text
nnunet_workspace_v2/nnUNet_results/Dataset802_HipBones_Aug/
  nnUNetTrainerNoMirrorEarlyStop__nnUNetPlans__2d/
    fold_0/checkpoint_best.pth
    fold_1/checkpoint_best.pth
    fold_2/checkpoint_best.pth
    fold_3/checkpoint_best.pth
    plans.json
    dataset.json
```

### Inference

```bash
python infer.py --case EE4559F3 --gt
python infer.py --input path/to/scan.dcm --output pred.nii.gz --overlay out.png
python infer.py --input path/to/img_0000.nii.gz --output pred.nii.gz
```

### Training (on a single GPU)

```bash
python train.py --stage prepare
python train.py --stage preprocess
python train.py --stage train --waves        
# or: python train.py --stage all --waves
```

