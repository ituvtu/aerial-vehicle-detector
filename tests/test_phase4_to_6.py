"""
Unit & Integration Tests for Phases 4–6:
- YOLO26-Pico-P2 architecture inspection (< 500k params, Rule 5)
- Metrics calculation & Invariants (global matching, >400m GT matching never causes FP in near/far)
- 101-point mAP@0.5 calculation
- Multi-clip Time to First Detection (delay + absolute ts)
- Validation confidence threshold calibration
- Model Registry & Champion Guard (append-only, auto-freeze, non-overwriting protection, manual promotion)
- 04_train.py CLI dry-run and Rule 1 split isolation
- 05_evaluate.py components (letterboxing, HUD telemetry, mock evaluation)
"""

from __future__ import annotations

import importlib
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
    inspect_and_verify_architecture,
    match_frame_detections,
)
from src.model_registry import ModelRegistry, compute_file_sha256

train_mod = importlib.import_module("scripts.04_train")
verify_rule1_split_isolation = train_mod.verify_rule1_split_isolation
plot_ablation_comparison_curves = train_mod.plot_ablation_comparison_curves

eval_mod = importlib.import_module("scripts.05_evaluate")
letterbox_to_canvas = eval_mod.letterbox_to_canvas
draw_hud_banner = eval_mod.draw_hud_banner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_yolo26_pico_p2_architecture_audit():
    yaml_path = PROJECT_ROOT / "configs" / "yolo26_pico_p2.yaml"
    assert yaml_path.exists(), f"Missing config: {yaml_path}"

    report = audit_model_architecture(yaml_path, max_params=500_000)
    assert report["unfused_params"] < 500_000, f"Unfused params {report['unfused_params']} >= 500k"
    assert report["fused_params"] < 500_000, f"Fused params {report['fused_params']} >= 500k"
    assert report["strides"] == [4.0, 8.0, 16.0]
    assert report["num_layers"] > 0
    assert report["backbone_params"] > 0
    assert report["neck_params"] > 0
    assert report["head_params"] > 0

    with pytest.raises(ValueError, match="exceeds limit"):
        audit_model_architecture(yaml_path, max_params=100_000)


def test_inspect_and_verify_architecture(tmp_path: Path):
    yaml_path = PROJECT_ROOT / "configs" / "yolo26_pico_p2.yaml"
    out_report = tmp_path / "model_arch_report.json"

    report = inspect_and_verify_architecture(
        yaml_path=yaml_path,
        pretrained_weights="",
        output_report_path=out_report,
        max_params=500_000,
    )

    assert out_report.exists()
    assert report["unfused_params"] < 500_000
    assert report["fused_params"] < 500_000
    assert "blocks" in report
    assert "backbone_layers_0_8" in report["blocks"]


def test_compute_box_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert compute_box_iou(box1, box2) == pytest.approx(1.0)

    # Half overlap
    box3 = (5.0, 0.0, 15.0, 10.0)
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
    # 2nd pred: lower conf overlaps same GT0 (FP since GT0 already matched)
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


def test_greater_than_400m_gt_does_not_become_fp():
    """
    CRITICAL RULE 4 INVARIANT:
    A prediction matching a GT in >400m must become a TP in >400m,
    and must NEVER be counted as an FP in 0-200m, 200-400m, or overall_0_400m.
    """
    # Tiny box corresponding to distant vehicle (> 400m)
    # In 4K (3840x2160), focal length ~2132px. Car length 4.5m at 500m is ~19px
    tiny_box_400m = (100.0, 100.0, 115.0, 115.0)

    gt_boxes = [tiny_box_400m]
    pred_boxes = [(101.0, 101.0, 114.0, 114.0, 0.88)]

    frame_records = [{"frame_id": "f001", "video_id": "vid1", "timestamp_sec": 0.0, "width": 3840, "height": 2160}]

    metrics = evaluate_split_metrics(
        frame_records=frame_records,
        gt_boxes_by_frame={"f001": gt_boxes},
        raw_preds_by_frame={"f001": pred_boxes},
        conf_threshold=0.50,
    )

    bands = metrics["bands"]
    # 0-200m must have 0 TP and 0 FP
    assert bands["0-200m"]["tp"] == 0
    assert bands["0-200m"]["fp"] == 0
    assert bands["0-200m"]["fn"] == 0

    # 200-400m must have 0 TP and 0 FP
    assert bands["200-400m"]["tp"] == 0
    assert bands["200-400m"]["fp"] == 0
    assert bands["200-400m"]["fn"] == 0

    # overall_0_400m must have 0 FP
    assert bands["overall_0_400m"]["fp"] == 0
    assert bands["overall_0_400m"]["tp"] == 0

    # >400m and overall_all must have 1 TP
    assert bands[">400m"]["tp"] == 1
    assert bands["overall_all"]["tp"] == 1


def test_compute_ap50_101_interpolation():
    # Perfect predictions: 3 TPs for 3 GTs
    assert compute_ap50([1, 1, 1], [0.9, 0.8, 0.7], total_gt=3) == pytest.approx(1.0)

    # Zero TPs: only FPs
    assert compute_ap50([0, 0, 0], [0.9, 0.8, 0.7], total_gt=3) == pytest.approx(0.0)

    # Empty
    assert compute_ap50([], [], total_gt=0) == pytest.approx(0.0)
    assert compute_ap50([], [], total_gt=5) == pytest.approx(0.0)


def test_evaluate_split_metrics_multi_clip_ttfd():
    frame_records = [
        {"frame_id": "vidA_001", "video_id": "vidA", "timestamp_sec": 1.0, "width": 1920, "height": 1080},
        {"frame_id": "vidA_002", "video_id": "vidA", "timestamp_sec": 2.0, "width": 1920, "height": 1080},
        {"frame_id": "vidB_001", "video_id": "vidB", "timestamp_sec": 1.0, "width": 1920, "height": 1080},
        {"frame_id": "vidB_002", "video_id": "vidB", "timestamp_sec": 3.0, "width": 1920, "height": 1080},
    ]

    box_near = (100.0, 100.0, 300.0, 300.0)

    gt_boxes = {
        "vidA_001": [box_near],
        "vidA_002": [box_near],
        "vidB_001": [],
        "vidB_002": [box_near],
    }

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
    assert "overall_0_400m" in bands
    assert "overall_all" in bands

    assert bands["overall_0_400m"]["time_to_first_detection_s"] == pytest.approx(0.0)
    assert bands["overall_0_400m"]["tp"] == 3
    assert bands["overall_0_400m"]["fp"] == 1
    assert bands["overall_0_400m"]["fn"] == 0
    assert bands["overall_0_400m"]["precision"] == pytest.approx(3 / 4)
    assert bands["overall_0_400m"]["detection_rate"] == pytest.approx(1.0)


def test_calibrate_confidence_threshold():
    val_records = [
        {"frame_id": f"frame_{i:03d}", "video_id": "val_vid", "timestamp_sec": float(i), "width": 1920, "height": 1080}
        for i in range(10)
    ]
    gt_boxes = {
        f"frame_{i:03d}": [(100.0, 100.0, 250.0, 250.0)] for i in range(10)
    }
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
    assert best_tau in [0.30, 0.50, 0.70]
    assert calib["best_objective_score"] == pytest.approx(1.0)


def test_model_registry_champion_guard(tmp_path: Path):
    """
    Tests Champion Guard invariants:
    1) Register m001 (score 0.78) -> weights/best.pt has m001 weights.
    2) Register m002 (score 0.65) -> archived in registry, but weights/best.pt stays m001!
    3) promote_model('m002') -> forces promotion, weights/best.pt updates to m002.
    """
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    registry = ModelRegistry(weights_dir)

    dummy_m001 = tmp_path / "m001_source.pt"
    dummy_m001.write_bytes(b"M001_WEIGHTS_VERSION_A")
    m001_sha = compute_file_sha256(dummy_m001)

    dummy_m002 = tmp_path / "m002_source.pt"
    dummy_m002.write_bytes(b"M002_WEIGHTS_VERSION_B_WEAKER")
    m002_sha = compute_file_sha256(dummy_m002)

    # 1. Register m001 (score 0.78)
    res1 = registry.register_model_run(
        exp_name="exp1_mosaic",
        best_pt_path=dummy_m001,
        calibration_data={"best_conf_threshold": 0.35},
        val_macro_f1=0.78,
        val_map50=0.78,
    )
    assert res1["is_champion"] is True
    root_best = weights_dir / "best.pt"
    assert root_best.exists()
    assert compute_file_sha256(root_best) == m001_sha

    # 2. Register m002 (score 0.65, weaker)
    res2 = registry.register_model_run(
        exp_name="exp2_rect_nomosaic",
        best_pt_path=dummy_m002,
        calibration_data={"best_conf_threshold": 0.40},
        val_macro_f1=0.65,
        val_map50=0.65,
    )
    assert res2["is_champion"] is False
    # CRITICAL: root weights/best.pt must NOT be overwritten!
    assert compute_file_sha256(root_best) == m001_sha

    # Verify ledger state
    ledger = registry.load_ledger()
    assert ledger["active_champion_id"] == "m001"
    assert len(ledger["models"]) == 2

    # 3. Manual promotion of m002
    registry.promote_model("m002")
    ledger_after = registry.load_ledger()
    assert ledger_after["active_champion_id"] == "m002"
    assert compute_file_sha256(root_best) == m002_sha


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
    assert "[Architecture Verified] YOLO26-Pico-P2" in res.stdout
    assert (PROJECT_ROOT / "results" / "model_architecture_report.json").exists()


def test_letterbox_to_canvas_aspect_ratios():
    # 9:16 vertical image (2160x3840)
    vert_img = np.zeros((3840, 2160, 3), dtype=np.uint8)
    vert_canvas = letterbox_to_canvas(vert_img, target_w=1920, target_h=1080)
    assert vert_canvas.shape == (1080, 1920, 3)

    # 16:9 horizontal image (3840x2160)
    horiz_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    horiz_canvas = letterbox_to_canvas(horiz_img, target_w=1920, target_h=1080)
    assert horiz_canvas.shape == (1080, 1920, 3)


def test_draw_hud_banner():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    banner_img = draw_hud_banner(
        frame_bgr=img,
        model_name="YOLO26-Pico-P2",
        fused_params_k=428.5,
        clip_role="test_clip",
        latency_ms=12.4,
        near_count=3,
        far_count=5,
        conf_thresh=0.35,
    )
    assert banner_img.shape == (1080, 1920, 3)


def test_plot_ablation_comparison_curves(tmp_path: Path):
    csv1 = tmp_path / "exp1.csv"
    csv2 = tmp_path / "exp2.csv"
    csv1.write_text("epoch,train/box_loss,val/box_loss,metrics/mAP50(B)\n0,1.2,1.1,0.5\n1,0.9,0.8,0.7\n", encoding="utf-8")
    csv2.write_text("epoch,train/box_loss,val/box_loss,metrics/mAP50(B)\n0,1.1,1.0,0.55\n1,0.8,0.7,0.75\n", encoding="utf-8")

    out_png = tmp_path / "ablation.png"
    assets_png = tmp_path / "assets_ablation.png"
    plot_ablation_comparison_curves(csv1, csv2, out_png, assets_png)
    assert out_png.exists()
    assert assets_png.exists()


def test_evaluate_cli_mock(tmp_path: Path):
    mock_results_dir = tmp_path / "results"
    dummy_weights = tmp_path / "best.pt"
    dummy_weights.write_text("dummy weights", encoding="utf-8")

    cmd = [
        sys.executable,
        "-c",
        f"""
import sys
import importlib
import numpy as np
from unittest.mock import patch, MagicMock

with patch('src.metrics.audit_model_architecture') as mock_audit, \\
     patch('ultralytics.YOLO') as mock_yolo:
    mock_audit.return_value = {{
        'unfused_params': 427567,
        'fused_params': 427567,
        'strides': [4.0, 8.0, 16.0],
        'max_params_limit': 500000,
    }}
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
    assert (mock_results_dir / "eval_summary_table.md").exists()

    eval_data = json.loads((mock_results_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert "bands" in eval_data
    assert "0-200m" in eval_data["bands"]
    assert "200-400m" in eval_data["bands"]
    assert "overall_0_400m" in eval_data["bands"]
    assert "latency_benchmark" in eval_data
