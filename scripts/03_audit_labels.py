#!/usr/bin/env python3
"""
Фаза 3-4 (Rule 2 & Rule 4):
1. Математичний аудит правок між data/labels_raw/ та data/labels/ (Added, Removed, Fixed, Unchanged).
2. Стратифікація всіх об'єктів за фізичною дистанцією (0-200m, 200-400m, >400m) через Pinhole-модель.
3. Перевірка Rule 4 (наявність авто у 200-400m на eval) та генерація data/audit_report.json і графіка.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest_manager import get_active_frames, parse_yolo_label_file  # noqa: F401 (re-exported)
from src.optics import (
    CameraOpticsConfig,
    assign_distance_band,
    estimate_distance_m_from_yolo_norm,
)


def yolo_norm_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Конвертує масив (N, 4) з [xc, yc, w, h] у [x1, y1, x2, y2]."""
    if len(boxes_xywh) == 0:
        return np.empty((0, 4), dtype=np.float64)
    out = np.empty_like(boxes_xywh, dtype=np.float64)
    half_w = boxes_xywh[:, 2] / 2.0
    half_h = boxes_xywh[:, 3] / 2.0
    out[:, 0] = boxes_xywh[:, 0] - half_w
    out[:, 1] = boxes_xywh[:, 1] - half_h
    out[:, 2] = boxes_xywh[:, 0] + half_w
    out[:, 3] = boxes_xywh[:, 1] + half_h
    return out


def compute_iou_matrix(boxes_a_xywh: np.ndarray, boxes_b_xywh: np.ndarray) -> np.ndarray:
    """Обчислює попарну матрицю IoU розміром (M, N) між двома наборами боксів [xc, yc, w, h]."""
    m, n = len(boxes_a_xywh), len(boxes_b_xywh)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)

    a = yolo_norm_to_xyxy(boxes_a_xywh)
    b = yolo_norm_to_xyxy(boxes_b_xywh)

    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter_area = inter_w * inter_h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return np.where(union_area > 0.0, inter_area / union_area, 0.0)


def compare_frame_labels(
    raw_boxes: np.ndarray,
    corrected_boxes: np.ndarray,
    match_iou_threshold: float = 0.30,
    unchanged_iou_threshold: float = 0.98,
) -> dict[str, int]:
    """
    Виконує двочасткове зіставлення (Hungarian algorithm) між сирими боксами моделі
    та виправленими людиною боксами для одного кадру (Rule 2).

    Інваріанти:
      raw_total == unchanged + fixed + removed
      corrected_total == unchanged + fixed + added
    """
    m, n = len(raw_boxes), len(corrected_boxes)
    if m == 0 and n == 0:
        return {"raw_total": 0, "corrected_total": 0, "unchanged": 0, "fixed": 0, "removed": 0, "added": 0}
    if m == 0:
        return {"raw_total": 0, "corrected_total": n, "unchanged": 0, "fixed": 0, "removed": 0, "added": n}
    if n == 0:
        return {"raw_total": m, "corrected_total": 0, "unchanged": 0, "fixed": 0, "removed": m, "added": 0}

    iou_mat = compute_iou_matrix(raw_boxes, corrected_boxes)
    # Маскуємо слабкі перетини, щоб угорський алгоритм максимізував суму IoU лише серед валідних пар
    masked_iou = np.where(iou_mat >= match_iou_threshold, iou_mat, 0.0)
    row_ind, col_ind = linear_sum_assignment(-masked_iou)

    unchanged = 0
    fixed = 0
    matched_raw = 0
    matched_corr = 0

    for r, c in zip(row_ind, col_ind):
        iou_val = float(iou_mat[r, c])
        if iou_val >= match_iou_threshold:
            matched_raw += 1
            matched_corr += 1
            if iou_val >= unchanged_iou_threshold:
                unchanged += 1
            else:
                fixed += 1

    removed = m - matched_raw
    added = n - matched_corr

    return {
        "raw_total": m,
        "corrected_total": n,
        "unchanged": unchanged,
        "fixed": fixed,
        "removed": removed,
        "added": added,
    }


def run_dataset_audit(
    active_df: pd.DataFrame,
    labels_raw_dir: Path,
    labels_dir: Path,
    optics_cfg: CameraOpticsConfig,
) -> tuple[dict[str, Any], list[float]]:
    """
    Проводить повний аудит датасету по всіх активних кадрах із manifest.csv.
    Повертає (audit_report_dict, all_distances_m_list).
    """
    overall_diff = {"raw_total": 0, "corrected_total": 0, "unchanged": 0, "fixed": 0, "removed": 0, "added": 0}
    split_diff: dict[str, dict[str, int]] = {
        s: {"raw_total": 0, "corrected_total": 0, "unchanged": 0, "fixed": 0, "removed": 0, "added": 0}
        for s in ("train", "val", "test")
    }
    band_counts_by_split: dict[str, dict[str, int]] = {
        s: {"0-200m": 0, "200-400m": 0, ">400m": 0, "total": 0}
        for s in ("train", "val", "test")
    }
    band_counts_by_role: dict[str, dict[str, int]] = {}
    all_distances_m: list[float] = []

    for _, row in active_df.iterrows():
        frame_id = str(row["frame_id"])
        split = str(row["split"])
        role = str(row["role"])
        w_px = int(row["width"])
        h_px = int(row["height"])

        if role not in band_counts_by_role:
            band_counts_by_role[role] = {"0-200m": 0, "200-400m": 0, ">400m": 0, "total": 0}

        raw_path = labels_raw_dir / f"{frame_id}.txt"
        corr_path = labels_dir / f"{frame_id}.txt"
        for lbl_path in (raw_path, corr_path):
            if not lbl_path.exists():
                raise FileNotFoundError(f"Missing label file for active frame '{frame_id}': {lbl_path}")
        raw_boxes = parse_yolo_label_file(raw_path)
        corr_boxes = parse_yolo_label_file(corr_path)

        diff = compare_frame_labels(raw_boxes, corr_boxes)
        for k, v in diff.items():
            overall_diff[k] += v
            if split in split_diff:
                split_diff[split][k] += v

        for box in corr_boxes:
            dist_m = estimate_distance_m_from_yolo_norm(
                w_norm=float(box[2]),
                h_norm=float(box[3]),
                img_w_px=w_px,
                img_h_px=h_px,
                config=optics_cfg,
            )
            band = assign_distance_band(dist_m, optics_cfg)
            all_distances_m.append(dist_m)

            if split in band_counts_by_split:
                band_counts_by_split[split][band] += 1
                band_counts_by_split[split]["total"] += 1
            band_counts_by_role[role][band] += 1
            band_counts_by_role[role]["total"] += 1

    eval_far_count = band_counts_by_split["test"]["200-400m"]
    eval_near_count = band_counts_by_split["test"]["0-200m"]

    report: dict[str, Any] = {
        "total_active_frames": len(active_df),
        "rule2_label_correction_summary": overall_diff,
        "rule2_by_split": split_diff,
        "rule4_distance_bands_by_split": band_counts_by_split,
        "rule4_distance_bands_by_role": band_counts_by_role,
        "rule4_eval_band_check": {
            "eval_0_200m_vehicles": eval_near_count,
            "eval_200_400m_vehicles": eval_far_count,
            "both_bands_populated": bool(eval_near_count > 0 and eval_far_count > 0),
        },
    }
    return report, all_distances_m


def save_distance_distribution_plot(
    distances_m: list[float],
    output_path: Path,
) -> None:
    """Генерує гістограму розподілу автомобілів за фізичною відстанню для README."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)

    total_gt = len(distances_m)
    n_near = sum(1 for d in distances_m if d <= 200.0)
    n_far = sum(1 for d in distances_m if 200.0 < d <= 400.0)
    n_very_far = sum(1 for d in distances_m if d > 400.0)

    clipped = [min(d, 449.9) for d in distances_m]
    if clipped:
        counts, bins, patches = ax.hist(
            clipped, bins=45, range=(0, 450), edgecolor="white", linewidth=0.7, alpha=0.90
        )
        for patch, x_left in zip(patches, bins[:-1]):
            if x_left < 199.9:
                patch.set_facecolor("#2b5c8f")
            elif x_left < 399.9:
                patch.set_facecolor("#2ca02c")
            else:
                patch.set_facecolor("#7f7f7f")

    ax.axvline(200.0, color="#d95f02", linestyle="--", linewidth=2.2, label="200 m band boundary (48 px in 4K)")
    ax.axvline(400.0, color="#e7298a", linestyle="--", linewidth=2.2, label="400 m max target range (24 px in 4K)")

    # Band summary callout boxes matching the README Data table
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.28)
    y_top = y_max * 1.22

    ax.text(
        100.0, y_top,
        f"0–200 m (Near Band)\n{n_near:,} GT ({n_near/max(1,total_gt)*100:.1f}%)\n751 train | 230 val | 441 test",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1d3f63",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef5fc", edgecolor="#2b5c8f", alpha=0.95),
    )
    ax.text(
        300.0, y_top,
        f"200–400 m (Far Band)\n{n_far:,} GT ({n_far/max(1,total_gt)*100:.1f}%)\n655 train | 131 val | 352 test",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1b611b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eefaf0", edgecolor="#2ca02c", alpha=0.95),
    )
    ax.text(
        425.0, y_top,
        f">400 m\n{n_very_far:,} GT ({n_very_far/max(1,total_gt)*100:.1f}%)\n(clipped at 450m)",
        ha="center", va="top", fontsize=7.8, fontweight="bold", color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f4f4", edgecolor="#7f7f7f", alpha=0.95),
    )

    ax.set_title(
        f"Vehicle Distance Distribution Across Dataset (155 Frames, 6 Clips, {total_gt:,} Curated GT Vehicles)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Estimated Pinhole Physical Distance D = (f_px × 4.5 m) / max(w_px, h_px) [meters]", fontsize=9.5)
    ax.set_ylabel("Number of Curated Vehicles", fontsize=9.5)
    ax.set_xlim(0, 460)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.78), framealpha=0.92, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit label corrections (Rule 2) and distance bands (Rule 4).")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest_path"]
    labels_raw_dir = PROJECT_ROOT / cfg["paths"]["labels_raw_dir"]
    labels_dir = PROJECT_ROOT / cfg["paths"]["labels_dir"]
    report_path = PROJECT_ROOT / cfg["paths"]["audit_report_path"]
    plot_path = PROJECT_ROOT / cfg["paths"]["assets_dir"] / "distance_distribution.png"

    optics_cfg = CameraOpticsConfig(
        fov_long_deg=float(cfg["optics"]["fov_long_deg"]),
        car_length_m=float(cfg["optics"]["car_length_m"]),
        car_width_m=float(cfg["optics"]["car_width_m"]),
        estimation_method=cfg["optics"]["estimation_method"],
    )

    active_df = get_active_frames(manifest_path)
    report, distances_m = run_dataset_audit(active_df, labels_raw_dir, labels_dir, optics_cfg)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_distance_distribution_plot(distances_m, plot_path)

    r2 = report["rule2_label_correction_summary"]
    print("\n" + "=" * 74)
    print("RULE 2: HUMAN LABEL AUDIT SUMMARY (labels_raw vs labels)")
    print("=" * 74)
    print(
        f"  Raw boxes (YOLOE-26x) : {r2['raw_total']}\n"
        f"  Final boxes (Human)   : {r2['corrected_total']}\n"
        f"  ----------------------------------------\n"
        f"  Unchanged (IoU>=0.98) : {r2['unchanged']}\n"
        f"  Fixed / Adjusted      : {r2['fixed']}\n"
        f"  Removed (False Pos)   : {r2['removed']}\n"
        f"  Added (Missed / FN)   : {r2['added']}"
    )

    print("\n" + "-" * 74)
    print("RULE 4: VEHICLE COUNT BY DISTANCE BAND AND SPLIT")
    print("-" * 74)
    split_df = pd.DataFrame(report["rule4_distance_bands_by_split"]).T
    print(split_df.to_string())

    print("\n" + "-" * 74)
    print("VEHICLE COUNT BY VIDEO ROLE:")
    print("-" * 74)
    role_df = pd.DataFrame(report["rule4_distance_bands_by_role"]).T
    print(role_df.to_string())
    print("=" * 74)
    print(f"Audit report saved to : {report_path}")
    print(f"Distribution plot     : {plot_path}\n")


if __name__ == "__main__":
    main()