#!/usr/bin/env python3
"""
Phase 5: Evaluation pipeline for held-out eval splits (eval + eval_B) per Rule 1 & Rule 4.
Generates eval_metrics.json, eval_table.md, side-by-side verification crops, and eval_demo.mp4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest_manager import ensure_images_available, get_active_frames, parse_yolo_label_file
from src.metrics import audit_model_architecture, evaluate_split_metrics
from src.optics import CameraOpticsConfig, assign_distance_band, estimate_distance_m_from_px


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_hud_banner(
    frame_bgr: np.ndarray,
    model_name: str,
    clip_role: str,
    latency_ms: float,
    near_count: int,
    far_count: int,
    conf_thresh: float,
) -> np.ndarray:
    """Renders semi-transparent HUD banner at the top of the video frame."""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    banner_h = int(h * 0.08)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    hud_text_1 = f"{model_name} | Clip: {clip_role} | Latency: {latency_ms:.1f}ms"
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
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "weights" / "calibration_val.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--max-vis-frames", type=int, default=8)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest_path"]
    images_dir = PROJECT_ROOT / cfg["paths"]["images_dir"]
    labels_dir = PROJECT_ROOT / cfg["paths"]["labels_dir"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "eval_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure test images available
    ensure_images_available(
        manifest_path=manifest_path,
        images_dir=images_dir,
        bundle_zip_path=PROJECT_ROOT / cfg["paths"]["images_bundle_zip"],
        bundle_url=cfg.get("remote_data", {}).get("images_bundle_url"),
    )

    # 2. Audit Model Architecture against <500k params limit
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    arch_info = audit_model_architecture(weights_path, max_params=cfg["training"]["max_parameters"])

    # 3. Load calibrated threshold tau*
    conf_thresh = float(cfg["evaluation"].get("default_conf_threshold", 0.25))
    calib_file = Path(args.calibration)
    if calib_file.exists():
        calib_data = json.loads(calib_file.read_text(encoding="utf-8"))
        conf_thresh = float(calib_data.get("best_conf_threshold", conf_thresh))
        print(f"[Evaluation] Using calibrated validation threshold: tau* = {conf_thresh:.2f}")
    else:
        print(f"[Evaluation] Calibration file not found; using fallback threshold: {conf_thresh:.2f}")

    # 4. Filter test frames (eval + eval_B)
    test_df = get_active_frames(manifest_path, split="test")
    test_records = test_df.to_dict(orient="records")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    gt_boxes_by_frame: dict[str, list[tuple[float, float, float, float]]] = {}
    raw_preds_by_frame: dict[str, list[tuple[float, float, float, float, float]]] = {}
    latencies_s: list[float] = []

    optics_cfg = CameraOpticsConfig(
        fov_long_deg=float(cfg["optics"]["fov_long_deg"]),
        car_length_m=float(cfg["optics"]["car_length_m"]),
        estimation_method=cfg["optics"]["estimation_method"],
    )

    vis_selected_count = 0
    demo_frames_bgr: list[np.ndarray] = []

    print(f"[Evaluation] Running inference on {len(test_records)} held-out test frames ...")

    for rec in test_records:
        fid = str(rec["frame_id"])
        w = int(rec["width"])
        h = int(rec["height"])
        role = str(rec["role"])

        lbl_path = labels_dir / f"{fid}.txt"
        yolo_boxes = parse_yolo_label_file(lbl_path)
        gt_pixel_boxes = []
        for box in yolo_boxes:
            cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            gt_pixel_boxes.append((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0))
        gt_boxes_by_frame[fid] = gt_pixel_boxes

        img_path = str(PROJECT_ROOT / rec["image_path"])
        img_bgr = cv2.imread(img_path)

        t0 = time.perf_counter()
        pred_res = model.predict(
            source=img_bgr,
            imgsz=int(cfg["training"].get("imgsz", 1024)),
            rect=bool(cfg["training"].get("eval_rect", True)),
            conf=0.01,
            iou=0.50,
            verbose=False,
        )
        t1 = time.perf_counter()
        latencies_s.append(t1 - t0)

        frame_preds: list[tuple[float, float, float, float, float]] = []
        if pred_res and pred_res[0].boxes is not None:
            boxes_np = pred_res[0].boxes.xyxy.cpu().numpy()
            confs_np = pred_res[0].boxes.conf.cpu().numpy()
            for b, c in zip(boxes_np, confs_np):
                frame_preds.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        raw_preds_by_frame[fid] = frame_preds

        # Annotated image for demo video & visual inspection
        annotated_bgr = img_bgr.copy()
        near_c = 0
        far_c = 0

        # Draw GT boxes in Green
        for g in gt_pixel_boxes:
            cv2.rectangle(annotated_bgr, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)

        # Draw Preds in Cyan (0-200m) or Orange (200-400m)
        for p in frame_preds:
            if p[4] >= conf_thresh:
                pw = max(0.0, p[2] - p[0])
                ph = max(0.0, p[3] - p[1])
                dist = estimate_distance_m_from_px(pw, ph, w, h, optics_cfg)
                band = assign_distance_band(dist, optics_cfg)

                if band == "0-200m":
                    color = (255, 255, 0) # Cyan
                    near_c += 1
                elif band == "200-400m":
                    color = (0, 165, 255) # Orange
                    far_c += 1
                else:
                    color = (0, 0, 255) # Red

                cv2.rectangle(annotated_bgr, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 2)
                label_text = f"{dist:.0f}m|{p[4]:.2f}"
                cv2.putText(
                    annotated_bgr,
                    label_text,
                    (int(p[0]), max(15, int(p[1]) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        draw_hud_banner(
            annotated_bgr,
            model_name=f"YOLO26-Pico-P2 ({arch_info['unfused_params']//1000}k)",
            clip_role=role,
            latency_ms=(t1 - t0) * 1000.0,
            near_count=near_c,
            far_count=far_c,
            conf_thresh=conf_thresh,
        )

        if vis_selected_count < args.max_vis_frames:
            crop_path = frames_dir / f"{fid}_gt_vs_pred.jpg"
            cv2.imwrite(str(crop_path), annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            vis_selected_count += 1

        if not args.no_video:
            canvas = letterbox_to_canvas(annotated_bgr, target_w=1920, target_h=1080)
            demo_frames_bgr.append(canvas)

    mean_latency_s = float(np.mean(latencies_s)) if latencies_s else 0.0

    # 5. Compute full evaluation metrics
    eval_metrics = evaluate_split_metrics(
        frame_records=test_records,
        gt_boxes_by_frame=gt_boxes_by_frame,
        raw_preds_by_frame=raw_preds_by_frame,
        conf_threshold=conf_thresh,
        fps=1.0,
        iou_threshold=float(cfg["evaluation"].get("iou_matching_threshold", 0.50)),
        fov_deg=optics_cfg.fov_long_deg,
        car_length_m=optics_cfg.car_length_m,
        method=optics_cfg.estimation_method,
        mean_inference_latency_s=mean_latency_s,
    )

    eval_metrics["model_architecture"] = arch_info
    eval_metrics["mean_inference_latency_ms"] = round(mean_latency_s * 1000.0, 2)

    # 6. Save JSON and Markdown summary table
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(eval_metrics, indent=2) + "\n", encoding="utf-8")

    bands = eval_metrics["bands"]
    b_near = bands["0-200m"]
    b_far = bands["200-400m"]
    b_all = bands["overall"]

    def _fmt_ttfd(val: Any) -> str:
        return f"{val:.2f}s" if val is not None else "N/A"

    table_md = f"""# Evaluation Metrics Summary (Held-out Test Split, Rule 1 & Rule 4)

**Model:** YOLO26-Pico-P2 ({arch_info['unfused_params']:,} unfused params, {arch_info['fused_params']:,} fused params)
**Confidence Threshold $\\tau^*$:** {conf_thresh:.2f} (calibrated on temporal val split)
**Inference Latency:** {eval_metrics['mean_inference_latency_ms']:.1f} ms / frame

| Metric | 0–200 m (Near) | 200–400 m (Far) | Overall (0–400 m) |
| :--- | :---: | :---: | :---: |
| **Ground Truth Vehicles** | {b_near['num_gt']} | {b_far['num_gt']} | {b_all['num_gt']} |
| **Detection Rate (Recall)** | {b_near['detection_rate'] * 100:.1f}% | {b_far['detection_rate'] * 100:.1f}% | {b_all['detection_rate'] * 100:.1f}% |
| **Precision** | {b_near['precision'] * 100:.1f}% | {b_far['precision'] * 100:.1f}% | {b_all['precision'] * 100:.1f}% |
| **False Alarms / min** | {b_near['false_alarms_per_min']:.2f} | {b_far['false_alarms_per_min']:.2f} | {b_all['false_alarms_per_min']:.2f} |
| **Time to First Detection** | {_fmt_ttfd(b_near['time_to_first_detection_s'])} | {_fmt_ttfd(b_far['time_to_first_detection_s'])} | {_fmt_ttfd(b_all['time_to_first_detection_s'])} |
| **mAP@0.5** | {b_near['map50']:.4f} | {b_far['map50']:.4f} | {b_all['map50']:.4f} |
"""
    table_path = out_dir / "eval_table.md"
    table_path.write_text(table_md, encoding="utf-8")
    print(f"\n{table_md}\n[Evaluation] Saved results to: {metrics_path} and {table_path}")

    # 7. Render eval demo video
    if not args.no_video and demo_frames_bgr:
        video_out_path = out_dir / "eval_demo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out_path), fourcc, 5.0, (1920, 1080))
        for f in demo_frames_bgr:
            writer.write(f)
        writer.release()
        print(f"[Evaluation] Saved demo video: {video_out_path} ({len(demo_frames_bgr)} frames @ 5 FPS)")


if __name__ == "__main__":
    main()

