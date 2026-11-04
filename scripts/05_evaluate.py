#!/usr/bin/env python3
"""
Phase 5: Decoupled Evaluation pipeline for held-out test split (Rule 1 & Rule 4).
Generates:
  - Latency benchmarks (mean, median, p95, fps, <33ms check)
  - eval_metrics.json, eval_summary_table.md (root + versioned in results/evaluations/{model_id}_{ts}/)
  - Sample comparison crops in results/eval_samples/ & assets/eval_samples/
  - 1-FPS GT vs Pred test video: results/eval_gt_vs_pred_1fps.mp4
  - Full 30.0 FPS Demo Video on original 32179597.mp4: results/eval_demo_32179597_full_fps.mp4
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest_manager import (
    ensure_images_available,
    ensure_raw_eval_video_available,
    get_active_frames,
    parse_yolo_label_file,
)
from src.metrics import audit_model_architecture, evaluate_split_metrics
from src.model_registry import ModelRegistry
from src.optics import CameraOpticsConfig, assign_distance_band, estimate_distance_m_from_px


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_hud_banner(
    frame_bgr: np.ndarray,
    model_name: str,
    fused_params_k: float,
    clip_role: str,
    latency_ms: float,
    near_count: int,
    far_count: int,
    conf_thresh: float,
) -> np.ndarray:
    """Renders semi-transparent HUD telemetry banner at the top of the video frame."""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    banner_h = int(h * 0.08)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, frame_bgr)

    fps_val = (1000.0 / latency_ms) if latency_ms > 0 else 0.0
    hud_text_1 = f"{model_name} ({fused_params_k:.1f}k fused) | Clip: {clip_role} | Latency: {latency_ms:.1f}ms ({fps_val:.1f} FPS)"
    hud_text_2 = f"Conf >= {conf_thresh:.2f} | 0-200m: {near_count} cars | 200-400m: {far_count} cars"

    scale = max(0.5, w / 2500.0)
    thickness = max(1, int(round(scale * 2)))

    cv2.putText(frame_bgr, hud_text_1, (20, int(banner_h * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame_bgr, hud_text_2, (20, int(banner_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 215, 255), thickness, cv2.LINE_AA)
    return frame_bgr


def letterbox_to_canvas(
    img: np.ndarray,
    target_w: int = 1920,
    target_h: int = 1080,
) -> np.ndarray:
    """Scales and pads any vertical (9:16) or horizontal (16:9) frame into uniform canvas."""
    h, w = img.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector on held-out test split (Rule 1, Rule 4).")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "project_config.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=None, help="Model ID from registry (e.g. m001)")
    parser.add_argument("--conf", type=float, default=None, help="Explicit confidence threshold override")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--max-vis-frames", type=int, default=8)
    parser.add_argument("--skip-full-video", action="store_true", help="Skip 30 FPS full video rendering")
    parser.add_argument("--max-demo-frames", type=int, default=None, help="Limit full video frames (for quick test)")
    parser.add_argument("--no-video", action="store_true", help="Skip all video rendering (for fast unit testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest_path"]
    images_dir = PROJECT_ROOT / cfg["paths"]["images_dir"]
    labels_dir = PROJECT_ROOT / cfg["paths"]["labels_dir"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve model checkpoint & calibration via ModelRegistry
    registry = ModelRegistry(PROJECT_ROOT / "weights")
    if args.weights is not None:
        weights_path = Path(args.weights)
        calib_file = Path(args.calibration) if args.calibration else PROJECT_ROOT / "weights" / "calibration_val.json"
        model_id = args.model_id or "custom_weights"
        from src.model_registry import compute_file_sha256
        weights_sha = compute_file_sha256(weights_path) if weights_path.exists() else "unknown"
    else:
        weights_path, calib_file, model_id, weights_sha = registry.resolve_model(args.model_id)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    # 2. Audit Model Architecture (<500k params limit)
    arch_info = audit_model_architecture(weights_path, max_params=cfg["training"]["max_parameters"])
    fused_params_k = arch_info["fused_params"] / 1000.0

    # 3. Resolve Confidence Threshold tau*
    if args.conf is not None:
        conf_thresh = args.conf
        print(f"[Evaluation] Using explicit CLI threshold: tau = {conf_thresh:.2f}")
    elif calib_file and calib_file.exists():
        calib_data = json.loads(calib_file.read_text(encoding="utf-8"))
        conf_thresh = float(calib_data.get("best_conf_threshold", 0.25))
        print(f"[Evaluation] Using calibrated validation threshold: tau* = {conf_thresh:.2f} (from {calib_file.name})")
    else:
        conf_thresh = float(cfg["evaluation"].get("default_conf_threshold", 0.25))
        print(f"[Evaluation] Calibration file not found; fallback threshold: {conf_thresh:.2f}")

    # 4. Ensure test images available
    ensure_images_available(
        manifest_path=manifest_path,
        images_dir=images_dir,
        bundle_zip_path=PROJECT_ROOT / cfg["paths"]["images_bundle_zip"],
        bundle_url=cfg.get("remote_data", {}).get("images_bundle_url"),
    )

    # 5. Load held-out test frames (Rule 1: eval 32179597 + eval_B 13763017)
    all_frames = get_active_frames(manifest_path)
    test_frames = all_frames[all_frames["split"] == "test"].copy()
    test_records = test_frames.to_dict(orient="records")
    eval_roles = sorted(test_frames["role"].unique())
    print(f"[Evaluation] Testing on held-out test clips: {eval_roles} ({len(test_records)} total frames)")

    # 6. Load Model & Run GPU Warm-Up
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    optics_cfg = CameraOpticsConfig(
        fov_long_deg=float(cfg["optics"]["fov_long_deg"]),
        car_length_m=float(cfg["optics"]["car_length_m"]),
        estimation_method=cfg["optics"]["estimation_method"],
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        dummy_input = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        for _ in range(5):
            model.predict(source=dummy_input, imgsz=args.imgsz, verbose=False, device=device)
        torch.cuda.synchronize()

    # 7. Run Inference on Test Split
    gt_boxes_by_frame: dict[str, list[tuple[float, float, float, float]]] = {}
    raw_preds_by_frame: dict[str, list[tuple[float, float, float, float, float]]] = {}

    pre_latencies_ms: list[float] = []
    inf_latencies_ms: list[float] = []
    post_latencies_ms: list[float] = []
    total_latencies_ms: list[float] = []

    annotated_test_frames: list[tuple[str, np.ndarray, str, float, int, int]] = []

    print(f"[Evaluation] Running inference on {len(test_records)} held-out test frames (imgsz={args.imgsz}, rect=True) ...")
    for record in test_records:
        fid = record["frame_id"]
        img_p = PROJECT_ROOT / record["image_path"]
        w = int(record["width"])
        h = int(record["height"])

        # Ground truth
        lbl_p = labels_dir / f"{fid}.txt"
        norm_gt = parse_yolo_label_file(lbl_p) if lbl_p.exists() else []
        gt_pixel_boxes = []
        for xc, yc, bw, bh in norm_gt:
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h
            gt_pixel_boxes.append((x1, y1, x2, y2))
        gt_boxes_by_frame[fid] = gt_pixel_boxes

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        pred_res = model.predict(
            source=img_bgr,
            imgsz=args.imgsz,
            rect=True,
            conf=0.01,
            iou=0.50,
            verbose=False,
            device=device,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        frame_total_ms = (t_end - t_start) * 1000.0
        total_latencies_ms.append(frame_total_ms)

        speed = getattr(pred_res[0], "speed", {}) if pred_res else {}
        pre_latencies_ms.append(speed.get("preprocess", frame_total_ms * 0.1))
        inf_latencies_ms.append(speed.get("inference", frame_total_ms * 0.8))
        post_latencies_ms.append(speed.get("postprocess", frame_total_ms * 0.1))

        frame_preds: list[tuple[float, float, float, float, float]] = []
        if pred_res and pred_res[0].boxes is not None:
            boxes_np = pred_res[0].boxes.xyxy.cpu().numpy()
            confs_np = pred_res[0].boxes.conf.cpu().numpy()
            for b, c in zip(boxes_np, confs_np):
                frame_preds.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        raw_preds_by_frame[fid] = frame_preds

        # Annotated image for GT vs Pred visualization
        annotated_bgr = img_bgr.copy()
        near_c = 0
        far_c = 0

        # Draw GT boxes in Green
        for g in gt_pixel_boxes:
            cv2.rectangle(annotated_bgr, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)

        # Draw Predictions: Cyan (0-200m) or Orange (200-400m)
        for p in frame_preds:
            if p[4] < conf_thresh:
                continue
            box_w = max(0.0, p[2] - p[0])
            box_h = max(0.0, p[3] - p[1])
            dist_m = estimate_distance_m_from_px(box_w, box_h, w, h, optics_cfg)
            band = assign_distance_band(dist_m, optics_cfg)

            if band == "0-200m":
                near_c += 1
                color = (255, 255, 0)  # Cyan
            elif band == "200-400m":
                far_c += 1
                color = (0, 165, 255)  # Orange
            else:
                color = (180, 180, 180)  # Gray for >400m

            cv2.rectangle(annotated_bgr, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 2)
            cv2.putText(
                annotated_bgr,
                f"vehicle {p[4]:.2f} | {dist_m:.0f}m",
                (int(p[0]), max(15, int(p[1]) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        hud_frame = draw_hud_banner(
            annotated_bgr,
            model_name="YOLO26-Pico-P2",
            fused_params_k=fused_params_k,
            clip_role=str(record["role"]),
            latency_ms=frame_total_ms,
            near_count=near_c,
            far_count=far_c,
            conf_thresh=conf_thresh,
        )
        annotated_test_frames.append((
            fid,
            hud_frame,
            str(record["role"]),
            float(record.get("timestamp_sec", record.get("timestamp_s", 0.0))),
            near_c,
            far_c,
        ))

    # 8. Compute Telemetry & Latency Benchmarks
    mean_lat = float(np.mean(total_latencies_ms)) if total_latencies_ms else 0.0
    median_lat = float(np.median(total_latencies_ms)) if total_latencies_ms else 0.0
    p95_lat = float(np.percentile(total_latencies_ms, 95)) if total_latencies_ms else 0.0
    fps_val = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    meets_30fps = bool(p95_lat < 33.33)

    latency_benchmark = {
        "device": device,
        "mean_ms": round(mean_lat, 2),
        "median_ms": round(median_lat, 2),
        "p95_ms": round(p95_lat, 2),
        "fps": round(fps_val, 2),
        "meets_realtime_33ms": meets_30fps,
        "breakdown_mean_ms": {
            "preprocess_ms": round(float(np.mean(pre_latencies_ms)), 2) if pre_latencies_ms else 0.0,
            "inference_ms": round(float(np.mean(inf_latencies_ms)), 2) if inf_latencies_ms else 0.0,
            "postprocess_ms": round(float(np.mean(post_latencies_ms)), 2) if post_latencies_ms else 0.0,
        },
    }

    # 9. Compute Overall & Per-Clip Metrics
    eval_results = evaluate_split_metrics(
        frame_records=test_records,
        gt_boxes_by_frame=gt_boxes_by_frame,
        raw_preds_by_frame=raw_preds_by_frame,
        conf_threshold=conf_thresh,
        fps=float(cfg["extraction"].get("fps", 1.0)),
        iou_threshold=float(cfg["evaluation"].get("iou_threshold", 0.50)),
        fov_deg=float(cfg["optics"]["fov_long_deg"]),
        car_length_m=float(cfg["optics"]["car_length_m"]),
        method=cfg["optics"]["estimation_method"],
        mean_inference_latency_s=mean_lat / 1000.0,
    )

    eval_results["model_id"] = model_id
    eval_results["weights_sha256"] = weights_sha
    eval_results["architecture"] = arch_info
    eval_results["latency_benchmark"] = latency_benchmark

    # Per-clip evaluation breakdown
    per_clip_eval: dict[str, Any] = {}
    for role_name in eval_roles:
        sub_records = [r for r in test_records if r["role"] == role_name]
        clip_m = evaluate_split_metrics(
            frame_records=sub_records,
            gt_boxes_by_frame=gt_boxes_by_frame,
            raw_preds_by_frame=raw_preds_by_frame,
            conf_threshold=conf_thresh,
            fps=float(cfg["extraction"].get("fps", 1.0)),
            iou_threshold=float(cfg["evaluation"].get("iou_threshold", 0.50)),
            fov_deg=float(cfg["optics"]["fov_long_deg"]),
            car_length_m=float(cfg["optics"]["car_length_m"]),
            method=cfg["optics"]["estimation_method"],
            mean_inference_latency_s=mean_lat / 1000.0,
        )
        per_clip_eval[role_name] = clip_m["bands"]
    eval_results["per_clip_bands"] = per_clip_eval

    # 10. Generate Summary Markdown Table
    bands = eval_results["bands"]
    md_lines = [
        "# Evaluation Results: YOLO26-Pico-P2 on Held-Out Test Split (Rule 1 & Rule 4)",
        "",
        f"**Model ID**: `{model_id}` | **Weights SHA**: `{weights_sha[:8]}` | **Parameters**: {arch_info['fused_params']:,} fused (<500k limit)",
        f"**Latency**: {mean_lat:.1f} ms mean ({fps_val:.1f} FPS), P95 = {p95_lat:.1f} ms (<33ms real-time: `{meets_30fps}`)",
        f"**Confidence Threshold**: $\\tau^* = {conf_thresh:.2f}$ (calibrated on val split)",
        "",
        "### Test Split Performance Across Optical Distance Bands",
        "",
        "| Distance Band | Target Distance | GT Vehicles | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Detection Rate (Recall) | Precision | F1 Score | mAP@0.5 | TTFD (s) | False Alarms / min |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    band_rows = [
        ("0-200m", "0–200 m (near)"),
        ("200-400m", "200–400 m (far)"),
        ("overall_0_400m", "0–400 m (target task)"),
        (">400m", ">400 m (ambient background)"),
        ("overall_all", "All Distances (0–∞ m)"),
    ]
    for b_key, target_desc in band_rows:
        b = bands[b_key]
        ttfd_str = f"{b['time_to_first_detection_s']:.2f}" if b['time_to_first_detection_s'] is not None else "N/A"
        md_lines.append(
            f"| **{b_key}** | {target_desc} | {b['num_gt']} | {b['tp']} | {b['fp']} | {b['fn']} | "
            f"{b['detection_rate'] * 100:.1f}% | {b['precision'] * 100:.1f}% | {b['f1'] * 100:.1f}% | "
            f"{b['map50']:.4f} | {ttfd_str}s | {b['false_alarms_per_min']:.2f} |"
        )

    md_lines.extend([
        "",
        "### Per-Clip Performance Breakdown (Rule 1 & Rule 4)",
        "",
        "| Clip Role | Video ID | Frames | 0–200m Recall | 200–400m Recall | 0–400m F1 | mAP@0.5 | TTFD Delay |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for role_name in eval_roles:
        sub_frames = [r for r in test_records if r["role"] == role_name]
        cb = per_clip_eval[role_name]
        ttfd_v = cb["overall_0_400m"]["time_to_first_detection_s"]
        ttfd_s = f"{ttfd_v:.2f}s" if ttfd_v is not None else "N/A"
        md_lines.append(
            f"| **{role_name}** | `{sub_frames[0]['video_id']}` | {len(sub_frames)} | "
            f"{cb['0-200m']['detection_rate']*100:.1f}% | {cb['200-400m']['detection_rate']*100:.1f}% | "
            f"{cb['overall_0_400m']['f1']*100:.1f}% | {cb['overall_0_400m']['map50']:.4f} | {ttfd_s} |"
        )

    # 11. Save Root & Versioned Evaluation Outputs
    ts_now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned_eval_dir = out_dir / "evaluations" / f"{model_id}_{ts_now}"
    versioned_eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_text = json.dumps(eval_results, indent=2) + "\n"
    md_text = "\n".join(md_lines) + "\n"

    # Root outputs
    (out_dir / "eval_metrics.json").write_text(metrics_text, encoding="utf-8")
    (out_dir / "eval_summary_table.md").write_text(md_text, encoding="utf-8")
    (out_dir / "eval_table.md").write_text(md_text, encoding="utf-8")  # backward compat

    # Versioned outputs
    (versioned_eval_dir / "eval_metrics.json").write_text(metrics_text, encoding="utf-8")
    (versioned_eval_dir / "eval_summary_table.md").write_text(md_text, encoding="utf-8")
    print(f"[Evaluation] Saved metrics and summary tables to {out_dir} and {versioned_eval_dir}")

    # 12. Save Visual Sample Crops in results/eval_samples/ & assets/eval_samples/
    eval_samples_dir = out_dir / "eval_samples"
    eval_samples_dir.mkdir(parents=True, exist_ok=True)
    assets_samples_dir = assets_dir / "eval_samples"
    assets_samples_dir.mkdir(parents=True, exist_ok=True)

    saved_crops = 0
    for fid, hud_img, role, ts, near_c, far_c in annotated_test_frames:
        if saved_crops >= args.max_vis_frames:
            break
        crop_path = eval_samples_dir / f"{fid}_vis.jpg"
        cv2.imwrite(str(crop_path), hud_img)
        if saved_crops < 4:
            cv2.imwrite(str(assets_samples_dir / f"{fid}_vis.jpg"), hud_img)
        saved_crops += 1
    print(f"[Evaluation] Saved {saved_crops} sample crops to {eval_samples_dir}")

    # 13. Render 1-FPS Test Comparison Video (results/eval_gt_vs_pred_1fps.mp4)
    if not args.no_video and annotated_test_frames:
        gt_pred_video_path = out_dir / "eval_gt_vs_pred_1fps.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw_1fps = cv2.VideoWriter(str(gt_pred_video_path), fourcc, 1.0, (1920, 1080))
        for fid, hud_img, role, ts, near_c, far_c in annotated_test_frames:
            canvas = letterbox_to_canvas(hud_img, target_w=1920, target_h=1080)
            vw_1fps.write(canvas)
        vw_1fps.release()
        print(f"[Evaluation] Rendered 1-FPS test comparison video: {gt_pred_video_path}")

    # 14. Render Full 30.0 FPS Demo Video on Raw Video 32179597.mp4 (Full HD 1080x1920 9:16)
    if not args.no_video and not args.skip_full_video:
        try:
            raw_video_path = ensure_raw_eval_video_available("32179597")
            full_fps_out = out_dir / "eval_demo_32179597_full_fps.mp4"
            print(f"\n[Evaluation] Rendering Full 30.0 FPS Demo Video on {raw_video_path.name} ...")

            cap = cv2.VideoCapture(str(raw_video_path))
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 2160
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 3840
            total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 930

            target_out_w = 1080
            target_out_h = 1920

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw_full = cv2.VideoWriter(str(full_fps_out), fourcc, orig_fps, (target_out_w, target_out_h))

            frame_idx = 0
            max_frames = args.max_demo_frames or total_raw_frames

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                t0 = time.perf_counter()
                res = model.predict(
                    source=frame_bgr,
                    imgsz=args.imgsz,
                    rect=True,
                    conf=conf_thresh,
                    iou=0.50,
                    verbose=False,
                    device=device,
                )
                t1 = time.perf_counter()
                f_lat_ms = (t1 - t0) * 1000.0

                near_c = 0
                far_c = 0

                if res and res[0].boxes is not None:
                    b_xyxy = res[0].boxes.xyxy.cpu().numpy()
                    b_conf = res[0].boxes.conf.cpu().numpy()
                    for b, c in zip(b_xyxy, b_conf):
                        box_w = max(0.0, b[2] - b[0])
                        box_h = max(0.0, b[3] - b[1])
                        dist_m = estimate_distance_m_from_px(box_w, box_h, orig_w, orig_h, optics_cfg)
                        band = assign_distance_band(dist_m, optics_cfg)

                        if band == "0-200m":
                            near_c += 1
                            col = (255, 255, 0)  # Cyan
                        elif band == "200-400m":
                            far_c += 1
                            col = (0, 165, 255)  # Orange
                        else:
                            col = (180, 180, 180)

                        cv2.rectangle(frame_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                        cv2.putText(
                            frame_bgr,
                            f"vehicle {c:.2f} | {dist_m:.0f}m",
                            (int(b[0]), max(15, int(b[1]) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            col,
                            1,
                            cv2.LINE_AA,
                        )

                hud_frame = draw_hud_banner(
                    frame_bgr,
                    model_name="YOLO26-Pico-P2",
                    fused_params_k=fused_params_k,
                    clip_role="eval (32179597)",
                    latency_ms=f_lat_ms,
                    near_count=near_c,
                    far_count=far_c,
                    conf_thresh=conf_thresh,
                )

                # Resize to target vertical resolution 1080x1920
                if hud_frame.shape[1] != target_out_w or hud_frame.shape[0] != target_out_h:
                    hud_frame = cv2.resize(hud_frame, (target_out_w, target_out_h), interpolation=cv2.INTER_LINEAR)

                vw_full.write(hud_frame)
                frame_idx += 1

            cap.release()
            vw_full.release()
            print(f"[Evaluation] Successfully rendered {frame_idx} frames at {orig_fps:.1f} FPS to: {full_fps_out}")
            # Also copy to versioned folder
            shutil.copy2(full_fps_out, versioned_eval_dir / "eval_demo_32179597_full_fps.mp4")
        except Exception as exc:
            print(f"[Evaluation] Note: Full video generation skipped or encountered error: {exc}")


if __name__ == "__main__":
    main()
