#!/usr/bin/env python3
"""
Phase 4: Training pipeline for YOLO26-Pico-P2 (<500k params)
Supports Ablation Study:
  - exp1_mosaic: rect=False, mosaic=1.0, close_mosaic=25, scale=0.5
  - exp2_rect_nomosaic: rect=True, mosaic=0.0, close_mosaic=0, scale=0.25
Features:
  - Architecture inspection & verification (<500k params, P2-P4 strides)
  - Validation threshold calibration strictly on val split (Rule 1)
  - Append-only Model Registry & Champion Guard
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from src.metrics import (
    audit_model_architecture,
    calibrate_confidence_threshold,
    evaluate_split_metrics,
    inspect_and_verify_architecture,
    transfer_coco_stem_weights,
)
from src.model_registry import ModelRegistry, create_pretrained_pico_checkpoint


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


def run_val_calibration(
    weights_path: Path,
    val_records: list[dict[str, Any]],
    labels_dir: Path,
    imgsz: int = 1024,
    device: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluates model on 18 val frames at imgsz=1024, rect=True across confidence thresholds
    to find tau* maximizing macro_f1 = 0.5 * (F1_0_200m + F1_200_400m).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    gt_boxes_by_frame: dict[str, list[tuple[float, float, float, float]]] = {}
    raw_preds_by_frame: dict[str, list[tuple[float, float, float, float, float]]] = {}

    for r in val_records:
        fid = r["frame_id"]
        img_p = PROJECT_ROOT / r["image_path"]
        w = int(r["width"])
        h = int(r["height"])

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

        # Predict
        pred_res = model.predict(
            source=str(img_p),
            imgsz=imgsz,
            rect=True,
            conf=0.01,
            iou=0.50,
            verbose=False,
            device=device or None,
        )

        frame_preds: list[tuple[float, float, float, float, float]] = []
        if pred_res and pred_res[0].boxes is not None:
            boxes_np = pred_res[0].boxes.xyxy.cpu().numpy()
            confs_np = pred_res[0].boxes.conf.cpu().numpy()
            for b, c in zip(boxes_np, confs_np):
                frame_preds.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        raw_preds_by_frame[fid] = frame_preds

    candidate_thresholds = [round(0.10 + 0.05 * i, 2) for i in range(16)]  # 0.10 .. 0.85
    calib = calibrate_confidence_threshold(
        val_records=val_records,
        gt_boxes_by_frame=gt_boxes_by_frame,
        raw_preds_by_frame=raw_preds_by_frame,
        candidate_thresholds=candidate_thresholds,
    )

    best_tau = calib["best_conf_threshold"]
    metrics_at_best = evaluate_split_metrics(
        frame_records=val_records,
        gt_boxes_by_frame=gt_boxes_by_frame,
        raw_preds_by_frame=raw_preds_by_frame,
        conf_threshold=best_tau,
    )
    return calib, metrics_at_best


def plot_single_experiment_curves(csv_path: Path, output_png: Path, exp_name: str) -> None:
    """Plots training and validation loss curves from Ultralytics results.csv."""
    if not csv_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if "epoch" in df.columns:
        epochs = df["epoch"]
    else:
        epochs = range(len(df))

    # Loss plot
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    for col in loss_cols[:4]:
        ax1.plot(epochs, df[col], label=col)
    ax1.set_title(f"Loss Curves ({exp_name})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metric plot
    metric_cols = [c for c in df.columns if "map" in c.lower() or "precision" in c.lower() or "recall" in c.lower()]
    for col in metric_cols[:4]:
        ax2.plot(epochs, df[col], label=col)
    ax2.set_title(f"Validation Metrics ({exp_name})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_ablation_comparison_curves(
    csv1: Path,
    csv2: Path,
    out_png: Path,
    assets_png: Path | None = None,
) -> None:
    """
    Plots training progression curves across all registered runs (m001..m005)
    using exact README run identifiers, falling back to csv1/csv2 for unit-test mocks.
    """
    import pandas as pd

    base_dir = csv1.parent
    c_m003 = base_dir / "train_history_exp3_pretrained.csv"
    c_m004 = base_dir / "train_history_exp4_pretrained_negatives.csv"
    c_m005 = base_dir / "train_history_exp3_pretrained_v1_scratch.csv"

    has_full_progression = c_m003.exists() and c_m004.exists()

    if has_full_progression:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1.15, 0.95]})
        run_specs = [
            (c_m004, "m004 (best.pt: Pretrained Stem + 24 Negatives)", "#1b9e77", "-", 2.6, 1.0, 5),
            (c_m003, "m003 (m003_best_recall.pt: Pretrained Stem 46/66)", "#2b5c8f", "-", 2.3, 0.95, 4),
            (csv1,   "m001 (Baseline .yaml + Mosaic)", "#7570b3", "-.", 1.7, 0.85, 3),
            (csv2,   "m002 (.yaml + rect=True, mosaic=0.0)", "#d95f02", "--", 1.6, 0.80, 2),
            (c_m005, "m005 (Scratch Control: 0/66 Random Init)", "#888888", ":", 1.8, 0.80, 1),
        ]
        for csv_path, label, color, ls, lw, alpha, zord in run_specs:
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

            val_box_col = [c for c in df.columns if "val/box_loss" in c]
            if val_box_col:
                ax1.plot(epochs, df[val_box_col[0]], label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=zord)

            map_col = [c for c in df.columns if "metrics/mAP50(B)" in c or "mAP50" in c]
            if map_col:
                ax2.plot(epochs, df[map_col[0]], label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=zord)

        ax1.set_title("Validation Box Loss Convergence (150 Epochs)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Validation Box Loss (lower is better)", fontsize=10)
        ax1.set_ylim(1.95, 4.90)
        ax1.legend(loc="upper right", fontsize=8.3, framealpha=0.92)
        ax1.grid(True, alpha=0.3)

        ax2.set_title("Epoch-by-Epoch Val mAP@0.5 (All Val Boxes)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Validation mAP@0.5 (higher is better)", fontsize=10)
        ax2.set_ylim(0.0, 0.72)
        ax2.legend(loc="lower right", fontsize=8.3, framealpha=0.92)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Final Calibrated Val Metrics (0-400m, rect=True @ tau*) matching README table
        runs_order = ["m001\n(Baseline)", "m002\n(No Mosaic)", "m003\n(Stem 46/66)", "m004 (best)\n(+24 Negs)", "m005\n(Scratch 0/66)"]
        macro_f1s = [0.744, 0.722, 0.750, 0.746, 0.615]
        map50s = [0.742, 0.731, 0.743, 0.758, 0.637]

        x = np.arange(len(runs_order))
        width = 0.36
        b1 = ax3.bar(x - width / 2, macro_f1s, width, label="Val Macro-F1 (0–400m @ τ*)", color="#2b5c8f", alpha=0.90)
        b2 = ax3.bar(x + width / 2, map50s, width, label="Val mAP@0.5 (0–400m, rect=True)", color="#1b9e77", alpha=0.90)

        for bar in b1:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.006, f"{h:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1d3f63")
        for bar in b2:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.006, f"{h:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#136f52")

        ax3.set_title("Calibrated Val Metrics (0–400 m @ τ*)", fontsize=11, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(runs_order, fontsize=8.5)
        ax3.set_ylim(0.50, 0.81)
        ax3.set_ylabel("Score on 18 Val Frames (Rule 1)", fontsize=10)
        ax3.legend(loc="lower left", fontsize=8.3, framealpha=0.92)
        ax3.grid(axis="y", alpha=0.3)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        def _plot_csv(csv_path: Path, label_prefix: str, color: str):
            if not csv_path.exists():
                return
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            epochs = df["epoch"] if "epoch" in df.columns else range(len(df))

            box_col = [c for c in df.columns if "train/box_loss" in c or "box_loss" in c]
            if box_col:
                ax1.plot(epochs, df[box_col[0]], label=f"{label_prefix} Train Box Loss", color=color, linestyle="-")
            val_box_col = [c for c in df.columns if "val/box_loss" in c]
            if val_box_col:
                ax1.plot(epochs, df[val_box_col[0]], label=f"{label_prefix} Val Box Loss", color=color, linestyle="--")

            map_col = [c for c in df.columns if "metrics/mAP50(B)" in c or "mAP50" in c]
            if map_col:
                ax2.plot(epochs, df[map_col[0]], label=f"{label_prefix} mAP@0.5", color=color, linewidth=2)

        _plot_csv(csv1, "m001 (exp1_mosaic)", "#1f77b4")
        _plot_csv(csv2, "m002 (exp2_rect_nomosaic)", "#ff7f0e")

        ax1.set_title("Ablation Study: Loss Convergence")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_title("Ablation Study: Validation mAP@0.5")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mAP@0.5")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    if assets_png:
        assets_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(assets_png, dpi=200)
    plt.close()


def train_single_experiment(
    exp_name: str,
    arch_yaml: Path,
    data_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    weights_dir: Path,
    results_dir: Path,
    exp_opts: dict[str, Any],
    pretrained_stem: bool = False,
) -> Path:
    """Trains a single model experiment and returns path to its best.pt."""
    from ultralytics import YOLO

    print(f"\n{'='*70}\n[Training] Starting Experiment: {exp_name}\nHyperparameters: {exp_opts}\n{'='*70}")
    if pretrained_stem:
        # Persist transferred stem into a real .pt so Ultralytics' trainer cannot re-init it.
        init_pt = weights_dir / "pico_p2_pretrained_init.pt"
        info = create_pretrained_pico_checkpoint(
            yaml_path=arch_yaml,
            backbone_source="yolo26n.pt",
            output_path=init_pt,
            report_path=results_dir / "model_architecture_report.json",
        )
        print(f"[Pretrained Transfer] {info['message']}")
        model = YOLO(str(init_pt))
    else:
        model = YOLO(str(arch_yaml))
        try:
            model.load("yolo26n.pt")
            print("[Pretrained Transfer] Transferred stem weights from yolo26n.pt")
        except Exception as exc:
            print(f"[Pretrained Transfer] Starting from scratch ({exc})")

    train_args = {
        "data": str(data_yaml.resolve()),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device or None,
        "patience": 40,
        "cos_lr": True,
        "warmup_epochs": 5.0,
        "cache": "ram",
        "amp": True,
        "workers": 4,
        "single_cls": True,
        "seed": 42,
        # Safe aerial augmentations
        "flipud": 0.5,
        "fliplr": 0.5,
        "degrees": 10.0,
        "translate": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "project": str(weights_dir / "runs"),
        "name": exp_name,
        "exist_ok": True,
        "verbose": True,
    }
    train_args.update(exp_opts)

    train_results = model.train(**train_args)
    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else weights_dir / "runs" / exp_name
    best_checkpoint = save_dir / "weights" / "best.pt"

    dest_best_pt = weights_dir / f"{exp_name}_best.pt"
    if best_checkpoint.exists():
        shutil.copy2(best_checkpoint, dest_best_pt)
    else:
        # Fallback to model weights on disk
        dest_best_pt = weights_dir / f"{exp_name}_best.pt"
        torch_model = getattr(model, "model", None)
        if torch_model is not None:
            import torch
            torch.save({"model": torch_model}, dest_best_pt)

    # Save training history CSV and curves
    results_csv = save_dir / "results.csv"
    dest_csv = results_dir / f"train_history_{exp_name}.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, dest_csv)
        plot_single_experiment_curves(
            dest_csv, results_dir / f"training_curves_{exp_name}.png", exp_name
        )

    return dest_best_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26-Pico-P2 & Validation Threshold Calibration.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "project_config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ablation", "exp1_mosaic", "exp2_rect_nomosaic", "exp3_pretrained", "exp4_pretrained_negatives"],
        default="ablation",
    )
    parser.add_argument("--pretrained-stem", action="store_true", help="Transfer pretrained COCO stem/backbone weights")
    parser.add_argument("--data", type=Path, default=None, help="Override path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--force-winner",
        type=str,
        default=None,
        choices=["exp1_mosaic", "exp2_rect_nomosaic", "exp3_pretrained", "exp4_pretrained_negatives"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-models", action="store_true", help="List registered models in weights/registry/")
    parser.add_argument("--promote-model", type=str, default=None, help="Promote specific model_id (e.g. m001)")
    args = parser.parse_args()

    registry = ModelRegistry(PROJECT_ROOT / "weights")

    if args.list_models:
        print("\n" + registry.list_models_table() + "\n")
        return

    if args.promote_model:
        registry.promote_model(args.promote_model)
        return

    cfg = load_config(args.config)
    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest_path"]
    images_dir = PROJECT_ROOT / cfg["paths"]["images_dir"]
    splits_dir = PROJECT_ROOT / cfg["paths"]["splits_dir"]
    labels_dir = PROJECT_ROOT / cfg["paths"]["labels_dir"]
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

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

    # 3. Model Architecture Audit & Inspection (<500k params, P2-P4)
    model_arch_yaml = PROJECT_ROOT / cfg["paths"]["model_arch_yaml"]
    arch_report = inspect_and_verify_architecture(
        yaml_path=model_arch_yaml,
        pretrained_weights="yolo26n.pt",
        output_report_path=results_dir / "model_architecture_report.json",
        max_params=cfg["training"]["max_parameters"],
    )
    # Also save copy in weights/ for backward compatibility
    (weights_dir / "model_arch_report.json").write_text(
        json.dumps(arch_report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\n[Architecture Verified] YOLO26-Pico-P2: {arch_report['unfused_params']:,} unfused / "
        f"{arch_report['fused_params']:,} fused params (< 500k limit)"
    )

    if args.dry_run:
        print("[Dry-Run] Architecture and splits verified successfully. Exiting clean.")
        return

    # 4. Define Experiments & Dataset YAML
    t_cfg = cfg.get("training", {})
    epochs = args.epochs or int(t_cfg.get("epochs", 150))
    batch = args.batch or int(t_cfg.get("batch_size", 8))
    imgsz = args.imgsz or int(t_cfg.get("imgsz", 1024))

    if args.data is not None:
        data_yaml = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    elif args.mode == "exp4_pretrained_negatives":
        data_yaml = PROJECT_ROOT / "data" / "dataset_with_negatives.yaml"
    else:
        data_yaml = PROJECT_ROOT / cfg["paths"]["dataset_yaml"]

    use_pretrained_stem = bool(args.pretrained_stem or (args.mode in ("exp3_pretrained", "exp4_pretrained_negatives")))

    experiments: dict[str, dict[str, Any]] = {}
    if args.mode in ("ablation", "exp1_mosaic"):
        experiments["exp1_mosaic"] = {
            "rect": False,
            "mosaic": 1.0,
            "close_mosaic": 25,
            "scale": 0.5,
        }
    if args.mode in ("ablation", "exp2_rect_nomosaic"):
        experiments["exp2_rect_nomosaic"] = {
            "rect": True,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "scale": 0.25,
        }
    if args.mode == "exp3_pretrained":
        experiments["exp3_pretrained"] = {
            "rect": False,
            "mosaic": 1.0,
            "close_mosaic": 25,
            "scale": 0.5,
        }
    if args.mode == "exp4_pretrained_negatives":
        experiments["exp4_pretrained_negatives"] = {
            "rect": False,
            "mosaic": 1.0,
            "close_mosaic": 25,
            "scale": 0.5,
        }

    # Load active curated ledger info for provenance
    ledger_file = PROJECT_ROOT / "data" / "registry" / "ledger.json"
    snapshot_id = None
    if ledger_file.exists():
        try:
            l_data = json.loads(ledger_file.read_text(encoding="utf-8"))
            snapshot_id = l_data.get("active_curated_version")
        except Exception:
            pass

    # Load val records (18 frames)
    all_frames = get_active_frames(manifest_path)
    val_frames = all_frames[all_frames["split"] == "val"].copy()
    val_records = val_frames.to_dict(orient="records")

    exp_results: dict[str, Any] = {}

    for exp_name, exp_opts in experiments.items():
        best_pt = train_single_experiment(
            exp_name=exp_name,
            arch_yaml=model_arch_yaml,
            data_yaml=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=args.device,
            weights_dir=weights_dir,
            results_dir=results_dir,
            exp_opts=exp_opts,
            pretrained_stem=use_pretrained_stem,
        )

        # Calibrate strictly on val
        print(f"\n[Validation Calibration] Running calibration for {exp_name} on {len(val_records)} val frames ...")
        calib_data, val_metrics = run_val_calibration(
            weights_path=best_pt,
            val_records=val_records,
            labels_dir=labels_dir,
            imgsz=imgsz,
            device=args.device,
        )

        # Save individual calibration
        calib_out = weights_dir / f"calibration_val_{exp_name}.json"
        calib_out.write_text(json.dumps(calib_data, indent=2) + "\n", encoding="utf-8")

        bands = val_metrics["bands"]
        f1_near = bands["0-200m"]["f1"]
        f1_far = bands["200-400m"]["f1"]
        macro_f1 = float(np.mean([f1_near, f1_far]))
        map50_0_400 = bands["overall_0_400m"]["map50"]

        is_force = (args.force_winner == exp_name)

        # Register in Model Registry
        manifest = registry.register_model_run(
            exp_name=exp_name,
            best_pt_path=best_pt,
            calibration_data=calib_data,
            val_macro_f1=macro_f1,
            val_map50=map50_0_400,
            train_history_csv=results_dir / f"train_history_{exp_name}.csv",
            training_curves_png=results_dir / f"training_curves_{exp_name}.png",
            hyperparameters={"epochs": epochs, "batch": batch, "imgsz": imgsz, **exp_opts},
            dataset_snapshot_id=snapshot_id,
            unfused_params=arch_report["unfused_params"],
            fused_params=arch_report["fused_params"],
            force_champion=is_force,
        )

        exp_results[exp_name] = {
            "model_id": manifest["model_id"],
            "val_selection_score": manifest["val_selection_score"],
            "val_macro_f1": macro_f1,
            "val_map50": map50_0_400,
            "f1_0_200m": f1_near,
            "f1_200_400m": f1_far,
            "best_conf_threshold": calib_data["best_conf_threshold"],
            "checkpoint": str(best_pt),
            "is_champion": manifest["is_champion"],
        }

    # 5. Generate Comparison Reports
    if len(experiments) > 1:
        plot_ablation_comparison_curves(
            csv1=results_dir / "train_history_exp1_mosaic.csv",
            csv2=results_dir / "train_history_exp2_rect_nomosaic.csv",
            out_png=results_dir / "val_ablation_comparison.png",
            assets_png=assets_dir / "training_curves.png",
        )

    # Save comparison JSON
    comparison_json = {
        "dataset_snapshot_id": snapshot_id,
        "val_frames_count": len(val_records),
        "experiments": exp_results,
    }
    (results_dir / "val_ablation_comparison.json").write_text(
        json.dumps(comparison_json, indent=2) + "\n", encoding="utf-8"
    )

    # Save comparison Markdown table
    md_lines = [
        "# Ablation Study Comparison on Validation Split (Rule 1 & Rule 6)",
        "",
        f"**Dataset Snapshot**: `{snapshot_id}` | **Validation Frames**: {len(val_records)} | **Metric**: $J = 0.5 \\cdot \\text{{Macro-F1}} + 0.5 \\cdot \\text{{mAP@0.5}}$",
        "",
        "| Experiment | Model ID | Rect Mode | Mosaic | $\\tau^*$ | Val F1 (0–200m) | Val F1 (200–400m) | Val Macro-F1 | Val mAP@0.5 | Selection Score | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ename, eres in exp_results.items():
        is_rect = "True" if "rect" in ename else "False"
        is_mos = "0.0" if "nomosaic" in ename else "1.0"
        status = "**CHAMPION** ★" if eres["is_champion"] else "Archived"
        md_lines.append(
            f"| **{ename}** | `{eres['model_id']}` | {is_rect} | {is_mos} | $\\tau={eres['best_conf_threshold']:.2f}$ | "
            f"{eres['f1_0_200m']*100:.1f}% | {eres['f1_200_400m']*100:.1f}% | "
            f"{eres['val_macro_f1']*100:.1f}% | {eres['val_map50']:.4f} | "
            f"**{eres['val_selection_score']:.4f}** | {status} |"
        )

    md_path = results_dir / "val_ablation_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n[Ablation Report] Saved comparison to {md_path}")
    print("\n" + registry.list_models_table() + "\n")


if __name__ == "__main__":
    main()
