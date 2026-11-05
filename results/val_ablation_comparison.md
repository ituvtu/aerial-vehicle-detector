# Validation Ablation & Experiment Progression Summary (Rule 1 & Rule 6)

**Dataset Snapshot**: `v002_curated_20261104T211538Z_174a4443` | **Validation Frames**: 18 | **Selection Metric**: $J = 0.5 \cdot \text{Macro-F1} + 0.5 \cdot \text{mAP@0.5}$

| Model ID | Experiment | SHA-8 | $\tau^*$ | Val F1 (0–200m) | Val F1 (200–400m) | Val Macro-F1 | Val mAP@0.5 | Selection Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `m001` | **exp1_mosaic** | `ec0238fb` | 0.25 | 77.1% | 71.7% | 74.40% | 0.7416 | **0.7428** | Archived |
| `m002` | **exp2_rect_nomosaic** | `bdf9970a` | 0.20 | 73.3% | 71.1% | 72.16% | 0.7305 | **0.7261** | Archived |
| `m003` | **exp3_pretrained** | `3af3f707` | 0.25 | 77.6% | 72.4% | 75.00% | 0.7432 | **0.7466** | Long-Range / Recall ★ |
| `m004` | **exp4_pretrained_negatives** | `7085ceb1` | 0.35 | 78.2% | 71.0% | 74.58% | 0.7576 | **0.7517** | **CHAMPION** ★ |
| `m005` | **exp3_pretrained_v1_scratch** | `b6ebd73d` | 0.20 | 72.6% | 50.4% | 61.49% | 0.6368 | **0.6259** | Archived |
