# Evaluation Results: YOLO26-Pico-P2 on Held-Out Test Split (Rule 1 & Rule 4)

**Model ID**: `custom_weights` | **Weights SHA**: `7085ceb1` | **Parameters**: 427,567 fused (<500k limit)
**Latency**: 18.8 ms mean (53.1 FPS), P95 = 19.2 ms (<33ms real-time: `True`)
**Confidence Threshold**: $\tau^* = 0.35$ (calibrated on val split)

### Test Split Performance Across Optical Distance Bands

| Distance Band | Target Distance | GT Vehicles | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Detection Rate (Recall) | Precision | F1 Score | mAP@0.5 | TTFD (s) | False Alarms / min |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0-200m** | 0–200 m (near) | 441 | 114 | 105 | 327 | 25.9% | 52.0% | 34.5% | 0.3852 | 0.52s | 98.44 |
| **200-400m** | 200–400 m (far) | 352 | 26 | 46 | 326 | 7.4% | 36.1% | 12.3% | 0.1411 | 9.01s | 43.12 |
| **overall_0_400m** | 0–400 m (target task) | 793 | 140 | 151 | 653 | 17.6% | 48.1% | 25.8% | 0.2765 | 0.52s | 141.56 |
| **>400m** | >400 m (ambient background) | 46 | 0 | 9 | 46 | 0.0% | 0.0% | 0.0% | 0.0000 | N/As | 8.44 |
| **overall_all** | All Distances (0–∞ m) | 839 | 140 | 160 | 699 | 16.7% | 46.7% | 24.6% | 0.2457 | 0.52s | 150.00 |

### Per-Clip Performance Breakdown (Rule 1 & Rule 4)

| Clip Role | Video ID | Frames | 0–200m Recall | 200–400m Recall | 0–400m F1 | mAP@0.5 | TTFD Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **eval** | `32179597` | 31 | 67.1% | 100.0% | 46.0% | 0.4417 | 1.02s |
| **eval_B** | `13763017` | 33 | 16.0% | 7.4% | 19.9% | 0.3803 | 0.02s |
