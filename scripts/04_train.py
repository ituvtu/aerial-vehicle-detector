#!/usr/bin/env python3
"""
Phase 4: Training pipeline for YOLO26-Pico-P2 (<500k params) & Validation Threshold Calibration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest_manager import (
    EVAL_VIDEO_IDS,
    ensure_images_available,
    get_active_frames,
    parse_yolo_label_file,
    sync_manifest_splits,
)
from src.metrics import audit_model_architecture, calibrate_confidence_threshold


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_rule1_split_isolation(splits_dir: Path) -> None:
    """Verifies that no eval video IDs appear in train.txt or val.txt."""
    for split_name in ("train", "val"):
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file missing: {split_file}")
        content = split_file.read_text(encoding="utf-8")
        for eval_id in EVAL_VIDEO_IDS:
            if eval_id in content:
                raise ValueError(
                    f"Rule 1 violation: held-out eval clip '{eval_id}' found in {split_name}.txt!"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26-Pico-P2 detector & calibrate on validation split.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project_config.yaml",
        help="Path to project_config.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override training batch size")
    parser.add_argument("--device", type=str, default="", help="CUDA device or cpu")
    parser.add_argument("--dry-run", action="store_true", help="Audit model architecture, verify splits, and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest_path"]
    images_dir = PROJECT_ROOT / cfg["paths"]["images_dir"]
    splits_dir = PROJECT_ROOT / cfg["paths"]["splits_dir"]
    labels_dir = PROJECT_ROOT / cfg["paths"]["labels_dir"]
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure images and splits are synchronized
    ensure_images_available(
        manifest_path=manifest_path,
        images_dir=images_dir,
        bundle_zip_path=PROJECT_ROOT / cfg["paths"]["images_bundle_zip"],
        bundle_url=cfg.get("remote_data", {}).get("images_bundle_url"),
    )
    sync_manifest_splits(manifest_path, splits_dir)

    # 2. Verify Rule 1 isolation
    verify_rule1_split_isolation(splits_dir)

    # 3. Model Architecture Audit (<500k params, P2-P3-P4)
    model_arch_yaml = PROJECT_ROOT / cfg["paths"]["model_arch_yaml"]
    arch_report = audit_model_architecture(model_arch_yaml, max_params=cfg["training"]["max_parameters"])
    arch_report_path = weights_dir / "model_arch_report.json"
    arch_report_path.write_text(json.dumps(arch_report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Architecture Audit] YOLO26-Pico-P2 verified: {arch_report['unfused_params']:,} unfused params (<500k limit)")

    if args.dry_run:
        print("[Dry-Run] Architecture and splits verified successfully. Exiting clean.")
        return

    # 4. Initialize model & Train
    from ultralytics import YOLO

    model = YOLO(str(model_arch_yaml))

    # Attempt compatible stem transfer from yolo26n if available
    try:
        model.load("yolo26n.pt")
        print("[Pretrained Transfer] Successfully loaded stem weights from yolo26n.pt")
    except Exception as exc:
        print(f"[Pretrained Transfer] Training from scratch (yolo26n.pt transfer note: {exc})")

    t_cfg = cfg["training"]
    epochs = args.epochs or int(t_cfg.get("epochs", 150))
    batch = args.batch or int(t_cfg.get("batch_size", 16))

    print(f"[Training] Starting YOLO26-Pico-P2 training ({epochs} epochs, batch {batch}) ...")
    train_results = model.train(
        data=str((PROJECT_ROOT / cfg["paths"]["dataset_yaml"]).resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=int(t_cfg.get("imgsz", 1024)),
        rect=bool(t_cfg.get("train_rect", False)),
        patience=int(t_cfg.get("patience", 35)),
        mosaic=float(t_cfg.get("mosaic", 1.0)),
        mixup=float(t_cfg.get("mixup", 0.15)),
        degrees=float(t_cfg.get("degrees", 15.0)),
        flipud=float(t_cfg.get("flipud", 0.5)),
        fliplr=float(t_cfg.get("fliplr", 0.5)),
        close_mosaic=15,
        device=args.device,
        save=True,
        verbose=True,
    )

    best_pt_src = Path(model.trainer.save_dir) / "weights" / "best.pt"
    best_pt_dest = PROJECT_ROOT / cfg["paths"]["best_weights_path"]
    if best_pt_src.exists():
        shutil.copy2(best_pt_src, best_pt_dest)
        print(f"[Model Saved] Best weights copied to: {best_pt_dest}")
    else:
        # Fallback if no training took place (e.g. 0 epochs)
        model.save(str(best_pt_dest))

    # Re-verify best weights against < 500k limit
    audit_model_architecture(best_pt_dest, max_params=cfg["training"]["max_parameters"])

    # 5. Calibrate confidence threshold tau* on val split (Rule 1)
    val_df = get_active_frames(manifest_path, split="val")
    val_records = val_df.to_dict(orient="records")

    gt_boxes_by_frame: dict[str, list[tuple[float, float, float, float]]] = {}
    raw_preds_by_frame: dict[str, list[tuple[float, float, float, float, float]]] = {}

    eval_model = YOLO(str(best_pt_dest))

    for rec in val_records:
        fid = str(rec["frame_id"])
        w = int(rec["width"])
        h = int(rec["height"])

        lbl_path = labels_dir / f"{fid}.txt"
        yolo_boxes = parse_yolo_label_file(lbl_path)
        gt_pixel_boxes = []
        for box in yolo_boxes:
            cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            gt_pixel_boxes.append((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0))
        gt_boxes_by_frame[fid] = gt_pixel_boxes

        img_path = str(PROJECT_ROOT / rec["image_path"])
        pred_res = eval_model.predict(
            source=img_path,
            imgsz=int(t_cfg.get("imgsz", 1024)),
            rect=bool(t_cfg.get("eval_rect", True)),
            conf=0.01,
            iou=0.50,
            verbose=False,
        )
        frame_preds: list[tuple[float, float, float, float, float]] = []
        if pred_res and pred_res[0].boxes is not None:
            boxes_np = pred_res[0].boxes.xyxy.cpu().numpy()
            confs_np = pred_res[0].boxes.conf.cpu().numpy()
            for b, c in zip(boxes_np, confs_np):
                frame_preds.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        raw_preds_by_frame[fid] = frame_preds

    calib_res = calibrate_confidence_threshold(
        val_records=val_records,
        gt_boxes_by_frame=gt_boxes_by_frame,
        raw_preds_by_frame=raw_preds_by_frame,
        fov_deg=float(cfg["optics"]["fov_long_deg"]),
        car_length_m=float(cfg["optics"]["car_length_m"]),
        method=cfg["optics"]["estimation_method"],
    )

    calib_path = weights_dir / "calibration_val.json"
    calib_path.write_text(json.dumps(calib_res, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n[Validation Calibration] Calibrated tau* = {calib_res['best_conf_threshold']:.2f} "
        f"(J = {calib_res['best_objective_score']:.4f}). Saved to {calib_path}"
    )


if __name__ == "__main__":
    main()
