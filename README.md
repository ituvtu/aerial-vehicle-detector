# Aerial Vehicle Detector

Sub-500k-parameter single-class (`vehicle`) detector for drone footage, evaluated separately on the
`0–200 m` and `200–400 m` distance bands (test assignment for thefourthlaw.ai).

> Status: Stage 1 (frame extraction + auto-labeling) ready; training/evaluation pending.

## 1. Data

The dataset consists of 6 aerial drone video clips from Pexels sampled at 1.0 FPS. Four clips are used for model training and validation (chronological 80/20 train/val split per video to avoid temporal leakage, following **Rule 1**), and two clips are strictly reserved as held-out evaluation clips (`split: test`):

| Video ID | Role | Scene Description | Resolution | Native FPS | Duration | Sampled (1 FPS) | Split Assignment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `8968356` | `train_A` | Highway interchange | 1920×1080 (16:9) | 29.97 | 19.2s | 19 | train: 15, val: 4 |
| `5382494` | `train_B` | Rural highway, light traffic | 3840×2160 (16:9) | 24.00 | 31.2s | 31 | train: 25, val: 6 |
| `8457857` | `train_C` | Simple highway, top-down | 3840×2160 (16:9) | 23.98 | 16.9s | 17 | train: 14, val: 3 |
| `3405804` | `train_D` | Urban intersection | 3840×2160 (16:9) | 29.97 | 24.5s | 24 | train: 19, val: 5 |
| `32179597` | `eval` | City highway, daytime | 2160×3840 (9:16) | 30.00 | 30.6s | 31 | test: 31 |
| `13763017` | `eval_B` | Highway traffic, high-angle oblique | 3840×2160 (16:9) | 59.94 | 32.5s | 33 | test: 33 |
| **Total** | | | | | | **155** | **train: 73, val: 18, test: 64** |

### Rule 4 Justification (Addition of `eval_B` / `13763017.mp4`)
Task specification **Rule 4** states:
> *"Both distance bands need data. If the eval clip has no vehicles at 200–400 m, add another clip and say which one."*

Optical distance auditing via the Pinhole camera model ($f_{\text{px}} \approx 2132.4\text{ px}$, $L_{\text{car}} = 4.5\text{ m}$, $\text{FOV}_{\text{long}} = 84^\circ$) of the default held-out evaluation clip `32179597.mp4` (`eval`) revealed it is a low-altitude (~30 m) vertical nadir shot containing 83 vehicles in the `0–200 m` band, but **0 vehicles in the `200–400 m` band**.

In strict adherence to Rule 4, we added `13763017.mp4` (`https://www.pexels.com/video/13763017`, `eval_B`, 3840×2160 @ 59.94 FPS, 33 frames at 1.0 FPS) as a second evaluation clip. It captures high-altitude oblique highway traffic where vehicle bounding boxes range between $12\text{ px}$ and $24\text{ px}$, directly populating the `200–400 m` evaluation band. Both clips remain 100% isolated inside the held-out `test` split (`split: test`) per Rule 1.

## 2. Labels

TODO: YOLOE-26x auto-labeling (`data/labels_raw/`), manual correction (`data/labels/`), Added/Removed/Fixed audit.

## 3. Training

TODO: architecture (`configs/yolo26_pico_p2.yaml`), parameter count (< 500 000), augmentation, hyper-parameters.

## 4. Eval setup

TODO: pinhole distance model (FOV 84° along the long side, car 4.5 × 1.8 m), metrics, matching rules.

## 5. Results

TODO: Recall, Precision, false alarms/min, time to first detection, mAP@0.5 for both bands.

## 6. What next

TODO: limitations and next steps.

