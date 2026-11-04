"""
Unit & Integration Tests for Phases 4–6:
- YOLO26-Pico-P2 architecture audit (< 500k params, Rule 5)
- Metrics calculation (IoU, greedy matching, AP@0.5, multi-clip TTFD, false alarms/min)
- Validation confidence threshold calibration
- 04_train.py CLI dry-run and Rule 1 split isolation
- 05_evaluate.py components (letterbox canvas, report generation)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.metrics import (
    audit_model_architecture,
    calibrate_confidence_threshold,
    compute_ap50,
    compute_box_iou,
    evaluate_split_metrics,
    match_frame_detections,
)
import importlib

train_mod = importlib.import_module("scripts.04_train")
verify_rule1_split_isolation = train_mod.verify_rule1_split_isolation

eval_mod = importlib.import_module("scripts.05_evaluate")
letterbox_to_canvas = eval_mod.letterbox_to_canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_yolo26_pico_p2_architecture_audit():
    yaml_path = PROJECT_ROOT / "configs" / "yolo26_pico_p2.yaml"
    assert yaml_path.exists(), f"Missing config: {yaml_path}"

    report = audit_model_architecture(yaml_path, max_params=500_000)
    assert report["unfused_params"] < 500_000, f"Unfused params {report['unfused_params']} >= 500k"
    assert report["fused_params"] < 500_000, f"Fused params {report['fused_params']} >= 500k"
    assert report["strides"] == [4.0, 8.0, 16.0]
    assert report["num_layers"] > 0

    with pytest.raises(ValueError, match="exceeds limit"):
        audit_model_architecture(yaml_path, max_params=100_000)


def test_compute_box_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert compute_box_iou(box1, box2) == pytest.approx(1.0)

    # Half overlap: box2 shifted right by 5
    box3 = (5.0, 0.0, 15.0, 10.0)
    # intersection: 5 * 10 = 50, union: 100 + 100 - 50 = 150 -> IoU = 1/3
    assert compute_box_iou(box1, box3) == pytest.approx(50.0 / 150.0)

    # Disjoint boxes
    box4 = (20.0, 20.0, 30.0, 30.0)
    assert compute_box_iou(box1, box4) == pytest.approx(0.0)


def test_match_frame_detections_invariants():
    # 2 GT boxes
    gt_boxes = [
        (100.0, 100.0, 200.0, 200.0),
        (300.0, 300.0, 400.0, 400.0),
    ]
    # Predictions:
    # 1st pred: overlaps GT0 with high conf (TP)
    # 2nd pred: lower conf overlaps same GT0 (should become FP since GT0 already matched)
    # 3rd pred: matches GT1 (TP)
    # 4th pred: no overlap with any GT (FP)
    pred_boxes = [
        (102.0, 102.0, 202.0, 202.0, 0.90),
        (105.0, 105.0, 205.0, 205.0, 0.80),
        (298.0, 298.0, 398.0, 398.0, 0.70),
        (500.0, 500.0, 600.0, 600.0, 0.60),
    ]

    res = match_frame_detections(
        gt_boxes_xyxy=gt_boxes,
        pred_boxes_xyxy_conf=pred_boxes,
        orig_w=1920,
        orig_h=1080,
        iou_threshold=0.50,
    )

    tps = res["matched_tps"]
    fps = res["fps"]
    fns = res["fns"]

    assert len(tps) == 2, f"Expected 2 TPs, got {len(tps)}"
    assert len(fps) == 2, f"Expected 2 FPs, got {len(fps)}"
    assert len(fns) == 0, f"Expected 0 FNs, got {len(fns)}"

    # Invariants
    assert len(tps) + len(fns) == len(gt_boxes)
    assert len(tps) + len(fps) == len(pred_boxes)
    assert res["total_gt"] == len(gt_boxes)
    assert res["total_preds"] == len(pred_boxes)


def test_compute_ap50():
    # Perfect predictions: 3 TPs for 3 GTs
    assert compute_ap50([1, 1, 1], [0.9, 0.8, 0.7], total_gt=3) == pytest.approx(1.0)

    # Zero TPs: only FPs
    assert compute_ap50([0, 0, 0], [0.9, 0.8, 0.7], total_gt=3) == pytest.approx(0.0)

    # Empty
    assert compute_ap50([], [], total_gt=0) == pytest.approx(0.0)
    assert compute_ap50([], [], total_gt=5) == pytest.approx(0.0)

    # Mixed known case:
    # GT = 2
    # Preds: TP (conf 0.9), FP (conf 0.7), TP (conf 0.5)
    # Rank 1: TP=1, FP=0 -> Recall = 1/2 = 0.5, Precision = 1/1 = 1.0
    # Rank 2: TP=1, FP=1 -> Recall = 1/2 = 0.5, Precision = 1/2 = 0.5
    # Rank 3: TP=2, FP=1 -> Recall = 2/2 = 1.0, Precision = 2/3 = 0.6667
    # Monotonic envelope:
    # At recall in [0, 0.5]: max precision = 1.0
    # At recall in [0.5, 1.0]: max precision = 0.6667
    # Area = 0.5 * 1.0 + 0.5 * (2/3) = 0.5 + 0.3333 = 0.8333
    ap = compute_ap50([1, 0, 1], [0.9, 0.7, 0.5], total_gt=2)
    assert ap == pytest.approx(0.8333, abs=1e-3)


def test_evaluate_split_metrics_multi_clip_ttfd():
    frame_records = [
        {"frame_id": "vidA_001", "video_id": "vidA", "timestamp_sec": 1.0, "width": 1920, "height": 1080},
        {"frame_id": "vidA_002", "video_id": "vidA", "timestamp_sec": 2.0, "width": 1920, "height": 1080},
        {"frame_id": "vidB_001", "video_id": "vidB", "timestamp_sec": 1.0, "width": 1920, "height": 1080},
        {"frame_id": "vidB_002", "video_id": "vidB", "timestamp_sec": 3.0, "width": 1920, "height": 1080},
    ]

    # Large box (near, ~0-200m)
    box_near = (100.0, 100.0, 300.0, 300.0)
    # Small box (far, ~200-400m)
    box_far = (100.0, 100.0, 115.0, 115.0)

    gt_boxes = {
        "vidA_001": [box_near],
        "vidA_002": [box_near],
        "vidB_001": [],
        "vidB_002": [box_near],
    }

    # Predictions:
    # vidA_001: detects box_near at t=1.0s (delay = 1.0 - 1.0 = 0.0s)
    # vidA_002: detects box_near
    # vidB_001: false alarm pred
    # vidB_002: detects box_near at t=3.0s (delay = 3.0 - 3.0 = 0.0s)
    raw_preds = {
        "vidA_001": [(100.0, 100.0, 300.0, 300.0, 0.95)],
        "vidA_002": [(100.0, 100.0, 300.0, 300.0, 0.92)],
        "vidB_001": [(500.0, 500.0, 550.0, 550.0, 0.85)],  # FP
        "vidB_002": [(100.0, 100.0, 300.0, 300.0, 0.90)],
    }

    results = evaluate_split_metrics(
        frame_records=frame_records,
        gt_boxes_by_frame=gt_boxes,
        raw_preds_by_frame=raw_preds,
        conf_threshold=0.50,
        fps=1.0,
    )

    bands = results["bands"]
    assert "0-200m" in bands
    assert "200-400m" in bands
    assert "overall" in bands

    # vidA delay = 0.0s, vidB delay = 0.0s -> mean TTFD = 0.0s
    assert bands["overall"]["time_to_first_detection_s"] == pytest.approx(0.0)
    assert bands["overall"]["tp"] == 3
    assert bands["overall"]["fp"] == 1
    assert bands["overall"]["fn"] == 0
    assert bands["overall"]["precision"] == pytest.approx(3 / 4)
    assert bands["overall"]["detection_rate"] == pytest.approx(1.0)
    assert bands["overall"]["map50"] > 0.90


def test_calibrate_confidence_threshold():
    val_records = [
        {"frame_id": f"frame_{i:03d}", "video_id": "val_vid", "timestamp_sec": float(i), "width": 1920, "height": 1080}
        for i in range(10)
    ]
    gt_boxes = {
        f"frame_{i:03d}": [(100.0, 100.0, 250.0, 250.0)] for i in range(10)
    }
    # TPs have conf ~ 0.80, noise has conf ~ 0.20
    raw_preds = {
        f"frame_{i:03d}": [
            (100.0, 100.0, 250.0, 250.0, 0.80),
            (500.0, 500.0, 600.0, 600.0, 0.20),
        ]
        for i in range(10)
    }

    calib = calibrate_confidence_threshold(
        val_records=val_records,
        gt_boxes_by_frame=gt_boxes,
        raw_preds_by_frame=raw_preds,
        candidate_thresholds=[0.10, 0.30, 0.50, 0.70, 0.90],
    )

    best_tau = calib["best_conf_threshold"]
    # At tau in [0.30, 0.70], F1 is perfect (1.0). Due to tie-breaking favoring higher tau, tau* should be 0.70
    assert best_tau in [0.30, 0.50, 0.70]
    assert calib["best_objective_score"] == pytest.approx(1.0)


def test_rule1_split_isolation_guard(tmp_path: Path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "val.txt").write_text("data/images/32179597_0001.jpg\n", encoding="utf-8")
    (splits_dir / "train.txt").write_text("data/images/32179598_0001.jpg\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Rule 1 violation"):
        verify_rule1_split_isolation(splits_dir)


def test_train_dry_run_cli():
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "04_train.py"),
        "--dry-run",
        "--config",
        str(PROJECT_ROOT / "configs" / "project_config.yaml"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    assert "[Architecture Audit] YOLO26-Pico-P2 verified" in res.stdout
    assert (PROJECT_ROOT / "weights" / "model_arch_report.json").exists()


def test_letterbox_to_canvas_aspect_ratios():
    # 9:16 vertical image (2160x3840)
    vert_img = np.zeros((3840, 2160, 3), dtype=np.uint8)
    vert_canvas = letterbox_to_canvas(vert_img, target_w=1920, target_h=1080)
    assert vert_canvas.shape == (1080, 1920, 3)

    # 16:9 horizontal image (3840x2160)
    horiz_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    horiz_canvas = letterbox_to_canvas(horiz_img, target_w=1920, target_h=1080)
    assert horiz_canvas.shape == (1080, 1920, 3)


def test_evaluate_cli_mock(tmp_path: Path):
    # Test evaluation output generation with a mocked YOLO model
    mock_results_dir = tmp_path / "results"
    dummy_weights = tmp_path / "best.pt"
    # Create empty dummy file for weights existence check
    dummy_weights.write_text("dummy weights", encoding="utf-8")

    with patch("src.metrics.audit_model_architecture") as mock_audit, \
         patch("ultralytics.YOLO") as mock_yolo_cls:

        mock_audit.return_value = {
            "unfused_params": 427567,
            "fused_params": 427567,
            "strides": [4.0, 8.0, 16.0],
            "max_params_limit": 500000,
        }

        # Setup mock predictor
        mock_model = MagicMock()
        mock_pred_box = MagicMock()
        mock_pred_box.xyxy = [[100.0, 100.0, 200.0, 200.0]]
        mock_pred_box.conf = [0.85]
        mock_pred_res = MagicMock()
        mock_pred_res.boxes = [mock_pred_box]
        mock_pred_res.__len__.return_value = 1
        mock_pred_res.__iter__.return_value = iter([mock_pred_box])
        mock_model.predict.return_value = [mock_pred_res]
        mock_yolo_cls.return_value = mock_model

        # Run evaluate main logic via CLI invocation or direct test
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
import importlib
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

with patch('src.metrics.audit_model_architecture') as mock_audit, \\
     patch('ultralytics.YOLO') as mock_yolo:
    mock_audit.return_value = {{'unfused_params': 427567, 'fused_params': 427567, 'strides': [4.0, 8.0, 16.0], 'max_params_limit': 500000}}
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu().numpy.return_value = np.array([[100.0, 100.0, 200.0, 200.0]])
    mock_boxes.conf.cpu().numpy.return_value = np.array([0.85])
    mock_res = MagicMock()
    mock_res.boxes = mock_boxes
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_res]
    mock_yolo.return_value = mock_model

    eval_mod = importlib.import_module('scripts.05_evaluate')
    sys.argv = ['05_evaluate.py', '--output-dir', r'{mock_results_dir}', '--weights', r'{dummy_weights}', '--no-video', '--max-vis-frames', '2']
    eval_mod.main()
"""
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Evaluate failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert (mock_results_dir / "eval_metrics.json").exists()
        assert (mock_results_dir / "eval_table.md").exists()

        eval_data = json.loads((mock_results_dir / "eval_metrics.json").read_text(encoding="utf-8"))
        assert "bands" in eval_data
        assert "0-200m" in eval_data["bands"]
        assert "200-400m" in eval_data["bands"]
        assert "overall" in eval_data["bands"]
