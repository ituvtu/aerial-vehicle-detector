# Aerial Vehicle Detector

Sub-500k-parameter single-class (`vehicle`) detector for drone footage, evaluated separately on the
`0–200 m` and `200–400 m` distance bands (test assignment for thefourthlaw.ai).

> Status: Stage 1 (frame extraction + auto-labeling) ready; training/evaluation pending.

## 1. Data

TODO: describe the 5 source videos, 1 FPS extraction, temporal 80/20 train/val split and the held-out eval video.

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

