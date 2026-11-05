# Experiments, Ablations, Architecture Breakdown, and Data Audit Logs

This document contains the complete experimental logs, per-band and per-clip validation/test evaluation tables, layer-by-layer architecture parameters, and data curation provenance for the **YOLO26-Pico-P2** (`< 500,000` parameters) aerial vehicle detector. Per **Rule 6**, detailed logs and side experiments are maintained here to keep [`README.md`](../README.md) focused and within the 4-page ceiling.

---

## 1. Complete Ablation Matrix (`m001` through `m005`)

All models were trained for `150` epochs at `imgsz=1024`, `batch=8`, `seed=42`, `cos_lr=True` (`lr0=0.01`, `lrf=0.01`), `warmup_epochs=5.0`, `patience=40`, `degrees=10.0`, `translate=0.1`, `fliplr=0.5`, `flipud=0.5`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `mixup=0.0`, `copy_paste=0.0` on an NVIDIA A100-SXM4-40GB GPU (`Ultralytics 8.4.202`, `PyTorch 2.11.0+cu130`).

In strict compliance with **Rule 1**, confidence thresholds $\tau^*$ were calibrated exclusively on the 18-frame `val` split by maximizing $\text{Macro-F1} = 0.5 \cdot (\text{F1}_{0\text{--}200\text{m}} + \text{F1}_{200\text{--}400\text{m}})$ over $\tau \in [0.10, 0.85]$ (step `0.05`) and frozen prior to test evaluation. Model selection in [`src/model_registry.py`](../src/model_registry.py) uses the composite validation score $J = 0.5 \cdot \text{Macro-F1}_{\text{val}} + 0.5 \cdot \text{mAP@0.5}_{\text{val}}$.

### 1.1 Validation Split Progression (`val`: 18 Frames, 361 GT Vehicles in `0–400 m` [230 Near, 131 Far])

| Registry ID | Experiment Name | Weights SHA-8 | Init Source & Stem Tensors | Train `rect` / `mosaic` | Train Images | Calibrated $\tau^*$ | Val `0–200m` R / P / F1 | Val `200–400m` R / P / F1 | Val Macro-F1 | Val `0–400m` mAP@0.5 | Val Selection Score $J$ | Status |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`m001`** | `exp1_mosaic` | `ec0238fb` | `.yaml` + `model.load("yolo26n.pt")` | `False` / `1.0` | 73 | `0.25` | 71.7% / 83.3% / **77.1%** | 68.7% / 75.0% / **71.7%** | 0.7440 | 0.7416 | 0.7428 | Baseline |
| **`m002`** | `exp2_rect_nomosaic` | `bdf9970a` | `.yaml` + `model.load("yolo26n.pt")` | `True` / `0.0` | 73 | `0.20` | 70.9% / 75.8% / **73.3%** | 74.1% / 68.3% / **71.1%** | 0.7216 | 0.7305 | 0.7261 | Rejected on `val` (Rule 1) |
| **`m003`** | `exp3_pretrained` | `3af3f707` | Persisted `.pt` (`46/66` stem tensors) | `False` / `1.0` | 73 | `0.25` | 73.9% / 81.7% / **77.6%** | 67.9% / 77.4% / **72.4%** | **0.7500** | 0.7432 | 0.7466 | **Long-Range / Recall ★** |
| **`m004`** | `exp4_pretrained_negatives` | `7085ceb1` | Persisted `.pt` (`46/66` stem tensors) + 24 Hard Negatives | `False` / `1.0` | 97 (73+24) | `0.35` | 70.9% / 87.2% / **78.2%** | 67.2% / 75.2% / **71.0%** | 0.7458 | **0.7576** | **0.7517** | **Active Champion ★** |
| **`m005`** | `exp3_pretrained_v1_scratch` | `b6ebd73d` | Scratch control (`0/66` random init) | `False` / `1.0` | 73 | `0.20` | 67.8% / 78.0% / **72.6%** | 45.0% / 57.3% / **50.4%** | 0.6149 | 0.6368 | 0.6259 | Scratch Control Ablation |

#### Detailed Validation Counts at $\tau^*$ (`18` Frames = `0.30 min` @ `1.0 FPS`)

| Registry ID | `0–200 m` TP / FP / FN | `0–200 m` FA/min | `0–200 m` mAP@0.5 | `200–400 m` TP / FP / FN | `200–400 m` FA/min | `200–400 m` mAP@0.5 | `0–400 m` TP / FP / FN | `0–400 m` Recall / Precision / F1 | `0–400 m` FA/min |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m001`** (`ec0238fb`) | 165 / 33 / 65 | 110.00 | 0.7654 | 90 / 30 / 41 | 100.00 | 0.7007 | 255 / 63 / 106 | 70.64% / 80.19% / 75.11% | 210.00 |
| **`m002`** (`bdf9970a`) | 163 / 52 / 67 | 173.33 | 0.7361 | 97 / 45 / 34 | 150.00 | 0.7247 | 260 / 97 / 101 | 72.02% / 72.83% / 72.42% | 323.33 |
| **`m003`** (`3af3f707`) | 170 / 38 / 60 | 126.67 | **0.7797** | 89 / 26 / 42 | **86.67** | 0.6845 | **259** / 64 / 102 | **71.75%** / 80.19% / **75.73%** | 213.33 |
| **`m004`** (`7085ceb1`) | 163 / **24** / 67 | **80.00** | 0.7776 | 88 / 29 / 43 | 96.67 | **0.7115** | 251 / **53** / 110 | 69.53% / **82.57%** / 75.49% | **176.67** |
| **`m005`** (`b6ebd73d`) | 156 / 44 / 74 | 146.67 | 0.7239 | 59 / 44 / 72 | 146.67 | 0.4627 | 215 / 88 / 146 | 59.56% / 70.96% / 64.76% | 293.33 |

---

### 1.2 Held-Out Test Split Evaluation (`test`: 64 Frames = `1.0667 min`, 839 Total GT Vehicles)

Ground-truth breakdown on `test`: **`441` vehicles in `0–200 m`**, **`352` vehicles in `200–400 m`** (`793` target vehicles in `0–400 m`), and **`46` ambient background vehicles in `>400 m`**. Note that `m002` (`bdf9970a`) was rejected on `val` per Rule 1 and never evaluated on `test`.

#### Combined Test Split (`eval` + `eval_B`, 64 Frames)

| Model (`SHA-8`, $\tau^*$) | Distance Band | GT | TP | FP | FN | Detection Rate (Recall) | Precision | F1 Score | False Alarms / min | Time to First Detection (s) | mAP@0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m001` (`ec0238fb`, $\tau^*=0.25$)** | **`0–200 m`** | 441 | 25 | 115 | 416 | 5.67% | 17.86% | 8.61% | 107.81 | 2.520 s | 0.1488 |
| *Baseline `.yaml` + `model.load`* | **`200–400 m`** | 352 | 52 | 82 | 300 | 14.77% | 38.81% | 21.40% | 76.88 | 0.020 s | 0.1672 |
| | **`overall_0_400m`** | 793 | 77 | 197 | 716 | 9.71% | 28.10% | 14.43% | 184.69 | 2.520 s | 0.1396 |
| | `>400 m` | 46 | 1 | 13 | 45 | 2.17% | 7.14% | 3.33% | 12.19 | 18.021 s | 0.0337 |
| | `overall_all` | 839 | 78 | 210 | 761 | 9.30% | 27.08% | 13.84% | 196.88 | 2.520 s | 0.1201 |
| **`m003` (`3af3f707`, $\tau^*=0.25$)** | **`0–200 m`** | 441 | 108 | 220 | 333 | 24.49% | 32.93% | 28.09% | 206.25 | 1.019 s | 0.2457 |
| **Long-Range / Recall Champion ★** | **`200–400 m`** | 352 | **132** | 120 | 220 | **37.50%** | **52.38%** | **43.71%** | 112.50 | **0.019 s** | **0.3596** |
| | **`overall_0_400m`** | 793 | **240** | 340 | 553 | **30.26%** | 41.38% | **34.96%** | 318.75 | 1.019 s | 0.2688 |
| | `>400 m` | 46 | 3 | 25 | 43 | 6.52% | 10.71% | 8.11% | 23.44 | 17.019 s | 0.0301 |
| | `overall_all` | 839 | 243 | 365 | 596 | 28.96% | 39.97% | 33.59% | 342.19 | 1.019 s | 0.2436 |
| **`m004` (`7085ceb1`, $\tau^*=0.35$)** | **`0–200 m`** | 441 | **114** | **105** | 327 | **25.85%** | **52.05%** | **34.55%** | **98.44** | **0.519 s** | **0.3852** |
| **High-Precision / Urban Champion ★** | **`200–400 m`** | 352 | 26 | **46** | 326 | 7.39% | 36.11% | 12.26% | **43.12** | 9.011 s | 0.1411 |
| | **`overall_0_400m`** | 793 | 140 | **151** | 653 | 17.65% | **48.11%** | 25.83% | **141.56** | **0.519 s** | **0.2765** |
| | `>400 m` | 46 | 0 | 9 | 46 | 0.00% | 0.00% | 0.00% | 8.44 | N/A | 0.0000 |
| | `overall_all` | 839 | 140 | **160** | 699 | 16.69% | **46.67%** | 24.58% | **150.00** | **0.519 s** | **0.2457** |
| **`m005` (`b6ebd73d`, $\tau^*=0.20$)** | **`0–200 m`** | 441 | 22 | 222 | 419 | 4.99% | 9.02% | 6.42% | 208.12 | 2.519 s | 0.0760 |
| *Scratch Control Ablation (`0/66`)* | **`200–400 m`** | 352 | 7 | 36 | 345 | 1.99% | 16.28% | 3.54% | 33.75 | 5.024 s | 0.0420 |
| | **`overall_0_400m`** | 793 | 29 | 258 | 764 | 3.66% | 10.10% | 5.37% | 241.88 | 2.519 s | 0.0601 |
| | `>400 m` | 46 | 0 | 10 | 46 | 0.00% | 0.00% | 0.00% | 9.38 | N/A | 0.0008 |
| | `overall_all` | 839 | 29 | 268 | 810 | 3.46% | 9.76% | 5.11% | 251.25 | 2.519 s | 0.0518 |

---

### 1.3 Per-Clip Breakdown on Held-Out Test Split (`eval` vs. `eval_B`)

#### Primary Evaluation Clip `eval` (`32179597.mp4` — Vertical 4K `2160×3840`, 31 Frames = `0.5167 min`, Low-Altitude ~30 m Nadir City Highway)
*Ground Truth: `85` vehicles in `0–200 m`, `0` vehicles in `200–400 m`, `0` vehicles in `>400 m`.* Because `num_gt = 0` in `200–400 m`, small false alarms on road markings or shadows (~14–28 px in 4K) project mathematically via the pinhole model into the `200–400 m` or `>400 m` buckets.

| Model (`SHA-8`, $\tau^*$) | Band | GT | TP | FP | FN | Detection Rate | Precision | F1 Score | False Alarms / min | TTFD (s) | mAP@0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m001` (`ec0238fb`, $\tau^*=0.25$)** | `0–200 m` | 85 | 7 | 112 | 78 | 8.24% | 5.88% | 6.86% | 216.77 | 5.020 s | 0.0668 |
| | `200–400 m` | 0 | 0 | 63 | 0 | — (0 GT) | 0.00% | 0.00% | 121.94 | N/A | 0.0000 |
| | `overall_0_400m` | 85 | 7 | 175 | 78 | 8.24% | 3.85% | 5.24% | 338.71 | 5.020 s | 0.0348 |
| | `overall_all` (incl. `>400m` 13 FP) | 85 | 7 | 188 | 78 | 8.24% | 3.59% | 5.00% | 363.87 | 5.020 s | 0.0286 |
| **`m003` (`3af3f707`, $\tau^*=0.25$)** | `0–200 m` | 85 | 28 | 179 | 57 | 32.94% | 13.53% | 19.18% | 346.45 | 2.019 s | 0.1281 |
| | `200–400 m` | 0 | 0 | 69 | 0 | — (0 GT) | 0.00% | 0.00% | 133.55 | N/A | 0.0000 |
| | `overall_0_400m` | 85 | 28 | 248 | 57 | 32.94% | 10.14% | 15.51% | 480.00 | 2.019 s | 0.0908 |
| | `overall_all` (incl. `>400m` 25 FP) | 85 | 28 | 273 | 57 | 32.94% | 9.30% | 14.51% | 528.39 | 2.019 s | 0.0806 |
| **`m004` (`7085ceb1`, $\tau^*=0.35$)** | **`0–200 m`** | 85 | **57** | **71** | **28** | **67.06%** | **44.53%** | **53.52%** | **137.42** | **1.019 s** | **0.4927** |
| | `200–400 m` | 0 | 0 | **35** | 0 | — (0 GT) | 0.00% | 0.00% | **67.74** | N/A | 0.0000 |
| | **`overall_0_400m`** | 85 | **57** | **106** | **28** | **67.06%** | **34.97%** | **45.97%** | **205.16** | **1.019 s** | **0.4417** |
| | `overall_all` (incl. `>400m` 9 FP) | 85 | **57** | **115** | **28** | **67.06%** | **33.14%** | **44.36%** | **222.58** | **1.019 s** | **0.4337** |
| **`m005` (`b6ebd73d`, $\tau^*=0.20$)** | `0–200 m` | 85 | 5 | 201 | 80 | 5.88% | 2.43% | 3.44% | 389.03 | 5.019 s | 0.0497 |
| | `200–400 m` | 0 | 0 | 30 | 0 | — (0 GT) | 0.00% | 0.00% | 58.06 | N/A | 0.0000 |
| | `overall_0_400m` | 85 | 5 | 231 | 80 | 5.88% | 2.12% | 3.12% | 447.10 | 5.019 s | 0.0390 |
| | `overall_all` (incl. `>400m` 10 FP) | 85 | 5 | 241 | 80 | 5.88% | 2.03% | 3.02% | 466.45 | 5.019 s | 0.0364 |

#### Supplementary Evaluation Clip `eval_B` (`13763017.mp4` — Horizontal 4K `3840×2160`, 33 Frames = `0.5500 min`, High-Angle Oblique Highway Interchange, Rule 4)
*Ground Truth: `356` vehicles in `0–200 m`, `352` vehicles in `200–400 m` (`708` target vehicles in `0–400 m`), and `46` vehicles in `>400 m` (`754` total).*

| Model (`SHA-8`, $\tau^*$) | Band | GT | TP | FP | FN | Detection Rate | Precision | F1 Score | False Alarms / min | TTFD (s) | mAP@0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m001` (`ec0238fb`, $\tau^*=0.25$)** | `0–200 m` | 356 | 18 | 3 | 338 | 5.06% | 85.71% | 9.55% | 5.45 | 0.020 s | 0.3728 |
| | `200–400 m` | 352 | 52 | 19 | 300 | 14.77% | 73.24% | 24.59% | 34.55 | 0.020 s | 0.3837 |
| | `overall_0_400m` | 708 | 70 | 22 | 638 | 9.89% | 76.09% | 17.50% | 40.00 | 0.020 s | 0.3806 |
| **`m003` (`3af3f707`, $\tau^*=0.25$)** | **`0–200 m`** | 356 | **80** | 41 | 276 | **22.47%** | 66.12% | **33.54%** | 74.55 | **0.019 s** | 0.4405 |
| | **`200–400 m`** | 352 | **132** | 51 | 220 | **37.50%** | 72.13% | **49.35%** | 92.73 | **0.019 s** | **0.4614** |
| | **`overall_0_400m`** | 708 | **212** | 92 | 496 | **29.94%** | 69.74% | **41.90%** | 167.27 | **0.019 s** | **0.4454** |
| **`m004` (`7085ceb1`, $\tau^*=0.35$)** | `0–200 m` | 356 | 57 | **34** | 299 | 16.01% | **62.64%** | 25.50% | **61.82** | **0.019 s** | **0.4532** |
| | `200–400 m` | 352 | 26 | **11** | 326 | 7.39% | **70.27%** | 13.37% | **20.00** | 9.011 s | 0.3303 |
| | `overall_0_400m` | 708 | 83 | **45** | 625 | 11.72% | **64.84%** | 19.86% | **81.82** | **0.019 s** | 0.3803 |
| **`m005` (`b6ebd73d`, $\tau^*=0.20$)** | `0–200 m` | 356 | 17 | 21 | 339 | 4.78% | 44.74% | 8.63% | 38.18 | 0.019 s | 0.2290 |
| | `200–400 m` | 352 | 7 | 6 | 345 | 1.99% | 53.85% | 3.84% | 10.91 | 5.024 s | 0.1324 |
| | `overall_0_400m` | 708 | 24 | 27 | 684 | 3.39% | 47.06% | 6.32% | 49.09 | 0.019 s | 0.1751 |

---

## 2. Detailed Engineering Narrative of Each Experiment

### 2.1 `m001` (`exp1_mosaic`, SHA `ec0238fb`): Baseline `.yaml` Config + `model.load("yolo26n.pt")`
- **Configuration:** Initialized via `model = YOLO("configs/yolo26_pico_p2.yaml")` followed by `model.load("yolo26n.pt")`, trained with `rect=False, mosaic=1.0, close_mosaic=25, scale=0.5` on the 73 active training frames (`v002_curated`).
- **Initialization subtleties discovered during audit:**
  1. `inspect_and_verify_architecture()` in [`src/metrics.py`](../src/metrics.py) attempted to capture `"Transferred X/Y items"` by redirecting `sys.stdout` around `model.load("yolo26n.pt")`. However, Ultralytics logs transfer messages via its `LOGGER` handler rather than `sys.stdout`, causing `model_arch_report.json` in `m001` and `m002` to record `"transferred_items": 0, "total_items": 0`.
  2. When calling `model.load("yolo26n.pt")` on a `YOLO(yaml)` instance, Ultralytics matches 58 tensors across the model and sets `self.ckpt`. During `model.train()`, Ultralytics still passes `cfg=self.model.yaml` into `DetectionTrainer`, which runs YAML-based initialization settings before applying `self.model`.
- **Results & Failure Analysis:** On `val`, `m001` achieved `val_selection_score = 0.7428` (`Macro-F1 = 0.7440`, `mAP@0.5 = 0.7416` at $\tau^* = 0.25$). On `test`, it achieved decent precision on the high-angle highway clip `eval_B` (`70 TP`, `22 FP`, `76.1%` Precision, `mAP@0.5 = 0.3806`), but collapsed on the vertical low-altitude nadir clip `eval` (`32179597`): only `7 TP` out of `85` GT vehicles (`8.24%` Recall) and `175 FP` in `0–400 m` (`188 FP` total including `13 FP` in `>400 m`, `mAP@0.5 = 0.0348`). Root cause: without pure background negative supervision and robust checkpoint-mode fine-tuning, the detector fired heavily on dashed white lane markings, crosswalk stripes, and building roof edges in `32179597`, while missing large ~250–320 px nadir vehicles due to the lacking `P5/32` receptive field (`max stride = 16`).

### 2.2 `m002` (`exp2_rect_nomosaic`, SHA `bdf9970a`): Rectangular Training without Mosaic (`rect=True, mosaic=0.0`)
- **Configuration:** Trained with `rect=True, mosaic=0.0, close_mosaic=0, scale=0.25` on the 73 training frames to test whether training at the native `16:9` aspect ratio (`1024×576`) without 4-image Mosaic composition improves small-object localization.
- **Validation Result & Rule 1 Rejection:** On the 18-frame `val` split ($\tau^* = 0.20$), `m002` degraded across all primary metrics relative to `m001`:
  - `val_macro_f1` dropped from `0.7440` to **`0.7216`** (`-2.24%`).
  - `val_map50` dropped from `0.7416` to **`0.7305`**.
  - `val_selection_score` dropped from `0.7428` to **`0.7261`**, so the `Champion Guard` in [`src/model_registry.py`](../src/model_registry.py) archived `m002` without overwriting `weights/best.pt`.
  - Validation false positives in `0–400 m` jumped by **+54.0%** (`63 FP` in `m001` $\to$ `97 FP` in `m002`, increasing false alarms from `210.0` to `323.3 FA/min`).
- **Engineering Takeaway:** Disabling Mosaic (`mosaic=0.0`) and reducing scale jitter (`scale=0.25`) on a small 73-frame dataset caused rapid overfitting to the fixed background layouts of the 4 training clips and deprived the network of synthetic multi-scale context. Per **Rule 1** (*"Decide all of that on training data"*), `exp2_rect_nomosaic` was rejected on validation evidence alone and never evaluated on `test`. Instead, we locked in `rect=False, mosaic=1.0` for training while keeping `rect=True` strictly for validation and test inference (where it eliminates `43.75%` of letterbox padding FLOPs).

### 2.3 `m003` (`exp3_pretrained`, SHA `3af3f707` — Long-Range / High-Recall Champion ★)
- **Deterministic Checkpoint Persistence Fix:** To prevent `DetectionTrainer` from re-initializing the network from YAML, we implemented `create_pretrained_pico_checkpoint()` in [`src/model_registry.py`](../src/model_registry.py) (tested by [`tests/test_pretrained_checkpoint.py`](../tests/test_pretrained_checkpoint.py)):
  1. Instantiates `target_model = YOLO("configs/yolo26_pico_p2.yaml")`.
  2. Loads the raw state dictionary from local `yolo26n.pt` and inspects all `66` parameter/buffer tensors belonging to backbone layers `model.0.` through `model.4.` (the `P1/2` and `P2/4` stem `Conv` layers, the `P2/4` `C3k2` block, the `P3/8` downsample `Conv`, and the `P3/8` `C3k2` block).
  3. Copies all **`46/66`** tensors whose names and exact shapes match 100% (`assert transferred_count > 0`) into `target_model.model` and saves a real Ultralytics checkpoint dictionary `torch.save({"model": target_model.model, ...}, "weights/pico_p2_pretrained_init.pt")`.
  4. In [`scripts/04_train.py`](../scripts/04_train.py), `--pretrained-stem` initializes training directly from the persisted checkpoint file: `model = YOLO("weights/pico_p2_pretrained_init.pt")`, which sets `args.yaml` `model: .../weights/pico_p2_pretrained_init.pt` and forces Ultralytics into checkpoint-finetuning mode.
- **Results:**
  - On `val` ($\tau^* = 0.25$), `m003` achieved the highest **`val_macro_f1 = 0.7500`** (`77.6%` F1 on `0–200 m`, `72.4%` F1 on `200–400 m`) and `val_selection_score = 0.7466`.
  - On `test`, total `0–400 m` True Positives jumped **3.1x** over `m001` (`77 TP` $\to$ **`240 TP`**) and **8.3x** over the scratch control run `m005` (`29 TP` $\to$ **`240 TP`**).
  - In the challenging **`200–400 m`** band (`352` GT vehicles), `m003` reached **`37.50%` Recall (`132 TP`), `52.38%` Precision, `43.71%` F1, and `mAP@0.5 = 0.3596`** across the full test split, and **`mAP@0.5 = 0.4614` (`72.13%` Precision, `49.35%` F1)** on `eval_B` (`13763017`). Overall `eval_B` `0–400 m` reached **`mAP@0.5 = 0.4454`** (`212 TP`, `69.74%` Precision).
  - However, on the low-altitude urban clip `eval` (`32179597`), while recall quadrupled (`7 TP` $\to$ `28 TP`, `32.94%`), high feature sensitivity at $\tau^* = 0.25$ still produced `248 FP` in `0–400 m` (`340 FP` combined across both test clips) on road markings and building facades.

### 2.4 `m004` (`exp4_pretrained_negatives`, SHA `7085ceb1` — High-Precision / Urban Champion ★)
- **Automated Laplacian Hard Negative Mining ([`scripts/03b_generate_hard_negatives.py`](../scripts/03b_generate_hard_negatives.py)):**
  - Because `100%` of the 73 active training frames contain vehicles, the detector never observed pure background tiles during training.
  - We scanned the 73 training frames with a $1024\times 1024$ sliding window (`stride = 256 px`), strictly rejecting any window within a `10 px` safety margin of any ground-truth box (`IoU = 0`).
  - Candidate background crops were scored by **Laplacian texture variance** ($\text{Var}(\nabla^2 I) \ge 25.0$) to prioritize high-frequency urban/highway distractors (lane markings, crosswalks, bridge joints, building roofs, tree shadows) over flat grass or sky.
  - Using round-robin selection across training video roles (`8` from `train_B`, `8` from `train_C`, `8` from `train_D`; `train_A` is `1920×1080` and has dense highway traffic covering all $1024\times 1024$ windows), we extracted **`24` pure background crops** (`data/images_negatives/*.jpg` with empty `0-byte` label files) and combined them with the 73 vehicle frames (`97` total training images in `data/splits/train_with_negatives.txt`).
- **Results & Precision-Recall Trade-off:**
  - On `val`, `m004` calibrated to a stricter threshold $\tau^* = 0.35$, cutting validation false positives to the lowest of all runs (`53 FP` in `0–400 m`, `82.57%` Precision) and achieving the highest **`val_map50 = 0.7576`** and **`val_selection_score = 0.7517`**, promoting `m004` to `active_champion_id` (`weights/best.pt`).
  - On `test` (`0–400 m`), total False Positives dropped by **-55.6%** compared to `m003` (`340 FP` $\to$ **`151 FP`**, cutting false alarms from `318.75` to **`141.56 FA/min`**), while overall Precision improved from `41.38%` to **`48.11%`** and overall `0–400 m` `mAP@0.5` rose from `0.2688` to **`0.2765`** (`0–200 m` `mAP@0.5` jumped from `0.2457` to **`0.3852`**).
  - On the primary low-altitude urban clip `eval` (`32179597`), `m004` achieved a dramatic breakthrough: `0–200 m` Recall jumped to **`67.06%`** (`57/85 TP`, up **8.1x** from `m001`'s `7 TP` and **2.0x** from `m003`'s `28 TP`), False Positives dropped by **-57.3%** vs. `m003` (`248` $\to$ `106 FP`), and clip `mAP@0.5` jumped **12.7x** over `m001` (`0.0348` $\to$ **`0.4417`**, with `0–200 m` `mAP@0.5 = 0.4927`).
  - **Trade-off at `200–400 m` on `eval_B`:** Combining 24 background crops (which teach the model to suppress low-contrast high-frequency textures) with a higher global threshold ($\tau^* = 0.35$ vs. `0.25`) suppressed weak activations on `6–12 px` oblique vehicles in `eval_B` (`200–400 m` Recall dropped from `37.50%` in `m003` to `7.39%` in `m004`, even though `eval_B` `200–400 m` Precision remained `70.27%` and `mAP@0.5 = 0.3303`). Because both checkpoints are `< 1.6 MB` and share the exact same architecture, we ship **both** `weights/best.pt` (`m004`) and `weights/m003_best_recall.pt` (`m003`).

### 2.5 `m005` (`exp3_pretrained_v1_scratch`, SHA `b6ebd73d` — Scratch Control Ablation: `0/66` Stem Tensors)
- **What happened & Role as Scratch Control:** When initially testing stem weight transfer (`model.0` through `model.4` from `yolo26n.pt`), we called `model.model.load_state_dict(...)` in memory after `model = YOLO("configs/yolo26_pico_p2.yaml")`. Unlike `model.load("yolo26n.pt")`, calling `load_state_dict` directly leaves `model.ckpt = None` and `model.model` associated only with `"configs/yolo26_pico_p2.yaml"`. When `model.train()` was invoked, `DetectionTrainer.setup_model()` saw `self.ckpt is None` and rebuilt a fresh `DetectionModel(cfg)` from the YAML file with random Kaiming/Xavier weights (`0/66` pretrained stem tensors survived into epoch 1).
- **Quantitative Impact of Training from Scratch on 73 Frames:**
  - On `val` ($\tau^* = 0.20$), `Macro-F1` dropped to **`0.6149`** (`200–400 m` F1 fell to `50.4%`, `mAP@0.5 = 0.4627`, compared to `0.7500` Macro-F1 in `m003`).
  - On `test`, total True Positives in `0–400 m` collapsed to **`29 TP`** (`22 TP` in `0–200 m`, **`7 TP`** in `200–400 m` — a **7.4x drop** in long-range detections compared to `m001`'s `52 TP` and an **18.9x drop** compared to `m003`'s `132 TP`), while `0–200 m` False Positives nearly doubled (`115 FP` $\to$ **`222 FP`**, `258 FP` total in `0–400 m`, `mAP@0.5 = 0.0601`).
  - This run serves as a clean scratch control ablation proving how much the persisted COCO stem weights in `m003` (`46/66` tensors) contribute when training a `<500k`-parameter detector on only 73 vehicle frames.

---

## 3. Architecture Layer-by-Layer Breakdown (`configs/yolo26_pico_p2.yaml`)

Verified against `results/model_architecture_report.json` and `audit_model_architecture()`:
- **Model Scale:** `n: [depth=0.50, width=0.25, max_channels=384]`, `nc: 1`, `reg_max: 1` (DFL-free bounding box regression), `end2end: True` (NMS-free dual-assignment one-to-one prediction head).
- **Total Unfused Parameters:** **`478,590`** (`21,410` below the strict `500,000` Rule 5 ceiling).
- **Total Fused Parameters (Conv+BN fused for inference):** **`427,567`** (`72,433` below ceiling).
- **Active Detection Strides:** **`P2/4` (`stride 4.0`), `P3/8` (`stride 8.0`), `P4/16` (`stride 16.0`)** — `P5/32` removed entirely.

### 3.1 Layer-by-Layer Parameter Table (`22` Modules)

| Layer Idx | Section | From | Repeats | Module | Config Arguments | Feature Stride | Unfused Params | Pretrained Stem Transfer (`yolo26n.pt`) |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| `0` | **Backbone** | `-1` | 1 | `Conv` | `[64, 3, 2]` (16 ch out) | `P1/2` | `464` | Transferred (`model.0.*`) |
| `1` | **Backbone** | `-1` | 1 | `Conv` | `[128, 3, 2]` (32 ch out) | `P2/4` | `4,672` | Transferred (`model.1.*`) |
| `2` | **Backbone** | `-1` | 2 | `C3k2` | `[256, False, 0.25]` (64 ch out) | `P2/4` | `6,640` | Transferred (`model.2.*`) |
| `3` | **Backbone** | `-1` | 1 | `Conv` | `[256, 3, 2]` (64 ch out) | `P3/8` | `36,992` | Transferred (`model.3.*`) |
| `4` | **Backbone** | `-1` | 2 | `C3k2` | `[256, False, 0.25]` (64 ch out) | `P3/8` | `7,664` | Transferred (`model.4.*`, **46/66** stem tensors matched) |
| `5` | **Backbone** | `-1` | 1 | `Conv` | `[384, 3, 2]` (96 ch out) | `P4/16` | `55,488` | Custom width (`384*0.25=96 ch`) — trained from init |
| `6` | **Backbone** | `-1` | 2 | `C3k2` | `[384, True]` (96 ch out) | `P4/16` | `49,152` | Relocated `C3k2(True)` at `P4/16` |
| `7` | **Backbone** | `-1` | 1 | `SPPF` | `[384, 5, 3, True]` (96 ch out) | `P4/16` | `23,328` | Relocated from `P5/32` to `P4/16` |
| `8` | **Backbone** | `-1` | 2 | `C2PSA` | `[384]` (96 ch out) | `P4/16` | `36,048` | Spatial attention relocated from `P5/32` to `P4/16` |
| *Subtotal* | **Backbone (`0..8`)** | | | | | `P1/2..P4/16` | **`220,448`** *(46.06%)* | `56,432` params in `0..4` (`46/66` tensors transferred) |
| `9` | **Neck (FPN)** | `-1` | 1 | `Upsample` | `[None, 2, "nearest"]` | `P4 -> P3` | `0` | — |
| `10` | **Neck (FPN)** | `[-1, 4]` | 1 | `Concat` | `[1]` (`96 + 64 = 160 ch`) | `P3/8` | `0` | — |
| `11` | **Neck (FPN)** | `-1` | 2 | `C3k2` | `[256, True]` (64 ch out) | `P3/8` | `28,160` | — |
| `12` | **Neck (FPN)** | `-1` | 1 | `Upsample` | `[None, 2, "nearest"]` | `P3 -> P2` | `0` | — |
| `13` | **Neck (FPN)** | `[-1, 2]` | 1 | `Concat` | `[1]` (`64 + 64 = 128 ch`) | `P2/4` | `0` | — |
| `14` | **Neck (P2 Out)** | `-1` | 2 | `C3k2` | `[128, True]` (32 ch out) | **`P2/4`** | `8,704` | High-resolution tiny-target feature map |
| `15` | **Neck (PAN)** | `-1` | 1 | `Conv` | `[128, 3, 2]` (32 ch out) | `P2 -> P3` | `9,280` | — |
| `16` | **Neck (PAN)** | `[-1, 11]` | 1 | `Concat` | `[1]` (`32 + 64 = 96 ch`) | `P3/8` | `0` | — |
| `17` | **Neck (P3 Out)** | `-1` | 2 | `C3k2` | `[256, True]` (64 ch out) | **`P3/8`** | `24,064` | Small-target feature map |
| `18` | **Neck (PAN)** | `-1` | 1 | `Conv` | `[256, 3, 2]` (64 ch out) | `P3 -> P4` | `36,992` | — |
| `19` | **Neck (PAN)** | `[-1, 8]` | 1 | `Concat` | `[1]` (`64 + 96 = 160 ch`) | `P4/16` | `0` | — |
| `20` | **Neck (P4 Out)** | `-1` | 2 | `C3k2` | `[384, True]` (96 ch out) | **`P4/16`** | `55,296` | Medium-target feature map |
| *Subtotal* | **Neck (`9..20`)** | | | | | `P2/4..P4/16` | **`162,496`** *(33.95%)* | |
| `21` | **Detect Head** | `[14, 17, 20]` | 1 | `Detect` | `[nc=1]` (`end2end=True, reg_max=1`) | `P2/4, P3/8, P4/16` | **`95,646`** *(19.98%)* | Dual one-to-many + one-to-one NMS-free head |
| **TOTAL** | **All (`0..21`)** | | | | | **Strides `[4, 8, 16]`** | **`478,590` unfused / `427,567` fused** | |

### 3.2 Real-Time Latency Benchmark (`imgsz=1024, rect=True` on NVIDIA A100 GPU)

| Model ID | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Meets `< 33.3 ms` Real-Time (`>= 30 FPS`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m001`** (`ec0238fb`) | 19.52 ms | 16.28 ms | 19.17 ms | 51.22 FPS | 4.16 ms | 12.62 ms | 1.82 ms | **YES** (`True`) |
| **`m003`** (`3af3f707`) | 19.02 ms | 15.91 ms | 18.83 ms | 52.58 FPS | 3.85 ms | 12.45 ms | 1.86 ms | **YES** (`True`) |
| **`m004`** (`7085ceb1`) | **18.84 ms** | **15.52 ms** | **19.23 ms** | **53.08 FPS** | **3.76 ms** | **12.48 ms** | **1.80 ms** | **YES** (`True`) |
| **`m005`** (`b6ebd73d`) | 18.65 ms | 15.58 ms | 17.42 ms | 53.63 FPS | 3.83 ms | 12.18 ms | 1.78 ms | **YES** (`True`) |

---

## 4. Data Curation Audit &Immutable Registry Logs (`data/audit_report.json` & `data/registry/ledger.json`)

### 4.1 Immutable Dataset Snapshot Ledger (`data/registry/ledger.json`)

| Seq | Snapshot Version ID | Kind | Created At (UTC) | Active Frames | Total Boxes | Merkle Tree SHA-256 |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `1` | `v001_raw_20261104T184015Z_8630536e` | `raw` (YOLOE-26x Auto-Label) | `2026-11-04T18:40:15Z` | 155 | 3,156 | `8630536e5aa6e54e165d0da8f2cc7e073990043baa175a1793ed3622231fccc9` |
| `2` | `v002_curated_20261104T211538Z_174a4443` | `curated` (100% Human CVAT Audit) | `2026-11-04T21:15:38Z` | 155 | 2,936 | `174a444301ccc4dd1c475123913353adc9ad8b03e18b2b0d2f385a4acfdb1a1c` |

### 4.2 Rule 2 Box-Level Curation Audit (`scripts/03_audit_labels.py`)

In [`scripts/03_audit_labels.py`](../scripts/03_audit_labels.py), `compare_frame_labels()` performs optimal bipartite matching (`scipy.optimize.linear_sum_assignment`) between raw auto-labels (`v001_raw`) and human-curated labels (`v002_curated`) per frame with `match_iou_threshold = 0.30` and `unchanged_iou_threshold = 0.98`:
- **Unchanged (`IoU >= 0.98`):** Auto-labeled boxes accepted without geometry modification.
- **Fixed / Resized (`0.30 <= IoU < 0.98`):** Matched boxes whose boundaries were tightened (trimming cast shadows, merging tractor + semi-trailer boxes).
- **Removed / False Positives (`IoU < 0.30` unmatched raw):** Hallucinated non-vehicle objects or excluded 2-wheeled motorcycles/bicycles deleted in CVAT.
- **Added / False Negatives (`IoU < 0.30` unmatched curated):** Missed distant (`200–400 m` / `>400 m`) or dark/occluded vehicles manually drawn in CVAT.

| Split | Active Frames | Raw Boxes (`v001_raw`) | Curated Boxes (`v002_curated`) | Unchanged (`IoU >= 0.98`) | Fixed (`0.30 <= IoU < 0.98`) | Removed (Auto-Label FP) | Added (Auto-Label FN) | Net Box Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | 73 | 1,887 | 1,675 | 1,245 | 65 | 577 | 365 | -212 |
| **`val`** | 18 | 445 | 422 | 299 | 15 | 131 | 108 | -23 |
| **`test`** | 64 | 824 | 839 | 743 | 56 | 25 | 40 | +15 |
| **Total** | **155** | **3,156** | **2,936** | **2,287** *(72.5%)* | **136** *(4.3%)* | **733** *(23.2%)* | **513** *(17.5% of final)* | **-220** |

*Verification invariants:* `raw_total = unchanged + fixed + removed` (`2,287 + 136 + 733 = 3,156`) and `corrected_total = unchanged + fixed + added` (`2,287 + 136 + 513 = 2,936`).

### 4.3 Rule 4 Distance Band Distribution by Video Role & Split (`v002_curated`)

| Video ID | Role | Native Resolution & FPS | Scene Description | Split Assignment (Frames) | `0–200 m` Vehicles | `200–400 m` Vehicles | `>400 m` Vehicles | Total Curated Vehicles |
| :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `8968356` | `train_A` | `1920×1080` @ `29.97 FPS` | Highway interchange (oblique) | `15 train` + `4 val` (`19` total) | 46 | 502 | 218 | **766** |
| `5382494` | `train_B` | `3840×2160` @ `24.00 FPS` | Rural highway, light traffic | `25 train` + `6 val` (`31` total) | 118 | 146 | 110 | **374** |
| `8457857` | `train_C` | `3840×2160` @ `23.98 FPS` | Simple highway, top-down | `14 train` + `3 val` (`17` total) | 282 | 134 | 2 | **418** |
| `3405804` | `train_D` | `3840×2160` @ `29.97 FPS` | Urban intersection (nadir) | `19 train` + `5 val` (`24` total) | 535 | 4 | 0 | **539** |
| *Subtotal* | **`train` split** | — | First 80% of `train_A..D` | **`73` frames** (+24 neg tiles in `m004`) | **751** | **655** | **269** | **1,675** |
| *Subtotal* | **`val` split** | — | Last 20% of `train_A..D` | **`18` frames** | **230** | **131** | **61** | **422** |
| `32179597` | `eval` | `2160×3840` (9:16) @ `30.00 FPS` | City highway, daytime (~30 m nadir) | **`31 test` frames** (`0.00s..30.00s`) | **85** | **0** | **0** | **85** |
| [`13763017`](https://www.pexels.com/video/cars-on-roads-and-viaduct-13763017/) | `eval_B` | `3840×2160` (16:9) @ `59.94 FPS` | High-angle oblique highway (Rule 4) | **`33 test` frames** (`0.00s..31.98s`) | **356** | **352** | **46** | **754** |
| *Subtotal* | **`test` split** | — | Held-out evaluation clips (`Rule 1`) | **`64` frames** | **441** | **352** | **46** | **839** |
| **TOTAL** | **All Videos** | — | 6 Pexels Drone Clips | **`155` active frames** | **1,422** *(48.4%)* | **1,138** *(38.8%)* | **376** *(12.8%)* | **2,936** *(100%)* |
