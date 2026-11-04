#!/usr/bin/env python3
"""
Phase 4 Extension: Hard Negative Background Mining for Aerial Vehicle Detection.
Extracts informative, non-vehicle 1024x1024 tiles from authorized training videos (train_A..train_D).
Filters and ranks candidate tiles using Laplacian texture variance (detecting road markings, building roofs, bridge shadows).
Generates:
  - data/images_negatives/*.jpg
  - data/labels_negatives/*.txt (0 bytes)
  - data/splits/train_with_negatives.txt
  - data/dataset_with_negatives.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest_manager import ensure_images_available, get_active_frames, parse_yolo_label_file


def check_box_overlap(
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    gt_pixel_boxes: list[tuple[float, float, float, float]],
    margin: int = 10,
) -> bool:
    """
    Returns True if the crop overlaps with ANY ground-truth box (including safety margin).
    Zero overlap is required for pure background tiles.
    """
    for gx1, gy1, gx2, gy2 in gt_pixel_boxes:
        # Expand GT box by safety margin
        ex_gx1 = max(0.0, gx1 - margin)
        ex_gy1 = max(0.0, gy1 - margin)
        ex_gx2 = gx2 + margin
        ex_gy2 = gy2 + margin

        # Check intersection
        inter_x1 = max(float(crop_x1), ex_gx1)
        inter_y1 = max(float(crop_y1), ex_gy1)
        inter_x2 = min(float(crop_x2), ex_gx2)
        inter_y2 = min(float(crop_y2), ex_gy2)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            return True
    return False


def compute_texture_variance(img_bgr: np.ndarray) -> float:
    """Computes edge / texture variance using Laplacian operator."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def generate_hard_negatives(
    manifest_path: Path,
    labels_dir: Path,
    images_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    output_split_file: Path,
    output_yaml: Path,
    max_negatives: int = 24,
    crop_size: int = 1024,
    stride: int = 256,
    margin: int = 10,
) -> list[dict[str, Any]]:
    """
    Slides crop_size window across train frames, validates zero GT overlap,
    scores candidates by texture variance, and selects top diverse hard negatives.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    df = get_active_frames(manifest_path)
    train_df = df[df["split"] == "train"].copy()

    candidates_by_role: dict[str, list[dict[str, Any]]] = {}

    print(f"[Hard Negative Mining] Scanning {len(train_df)} train frames with window {crop_size}x{crop_size} (stride {stride}) ...")

    for _, row in train_df.iterrows():
        fid = str(row["frame_id"])
        role = str(row["role"])
        img_p = PROJECT_ROOT / str(row["image_path"])
        w_px = int(row["width"])
        h_px = int(row["height"])

        if not img_p.exists():
            continue

        lbl_p = labels_dir / f"{fid}.txt"
        norm_gt = parse_yolo_label_file(lbl_p) if lbl_p.exists() else []

        gt_boxes = []
        for xc, yc, bw, bh in norm_gt:
            x1 = (xc - bw / 2.0) * w_px
            y1 = (yc - bh / 2.0) * h_px
            x2 = (xc + bw / 2.0) * w_px
            y2 = (yc + bh / 2.0) * h_px
            gt_boxes.append((x1, y1, x2, y2))

        # Check if frame can fit crop_size
        if w_px < crop_size or h_px < crop_size:
            continue

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        for y in range(0, h_px - crop_size + 1, stride):
            for x in range(0, w_px - crop_size + 1, stride):
                # Verify zero overlap with GT
                if check_box_overlap(x, y, x + crop_size, y + crop_size, gt_boxes, margin=margin):
                    continue

                crop = img_bgr[y : y + crop_size, x : x + crop_size]
                var_score = compute_texture_variance(crop)

                # Skip completely flat / empty regions
                if var_score < 25.0:
                    continue

                candidate = {
                    "frame_id": fid,
                    "role": role,
                    "x": x,
                    "y": y,
                    "var_score": var_score,
                    "crop_bgr": crop,
                }
                candidates_by_role.setdefault(role, []).append(candidate)

    total_candidates = sum(len(v) for v in candidates_by_role.values())
    print(f"[Hard Negative Mining] Found {total_candidates} pure background candidate tiles across roles: {list(candidates_by_role.keys())}")

    # Sort candidates in each role descending by texture variance
    for role in candidates_by_role:
        candidates_by_role[role].sort(key=lambda c: c["var_score"], reverse=True)

    # Round-robin selection across roles for maximum scene diversity
    selected: list[dict[str, Any]] = []
    roles = sorted(candidates_by_role.keys())
    role_pointers = {r: 0 for r in roles}

    while len(selected) < max_negatives and any(role_pointers[r] < len(candidates_by_role[r]) for r in roles):
        for r in roles:
            if len(selected) >= max_negatives:
                break
            ptr = role_pointers[r]
            if ptr < len(candidates_by_role[r]):
                selected.append(candidates_by_role[r][ptr])
                role_pointers[r] += 1

    print(f"[Hard Negative Mining] Selected top {len(selected)} diverse hard negatives (target: {max_negatives}).")

    # Save images and empty label files
    saved_records = []
    new_image_rel_paths: list[str] = []

    for idx, item in enumerate(selected, start=1):
        crop_id = f"bg_{item['role']}_{item['frame_id']}_c{idx:02d}"
        img_name = f"{crop_id}.jpg"
        lbl_name = f"{crop_id}.txt"

        dest_img_path = output_images_dir / img_name
        dest_lbl_path = output_labels_dir / lbl_name

        cv2.imwrite(str(dest_img_path), item["crop_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        # Write empty label file (0 bytes)
        dest_lbl_path.write_text("", encoding="utf-8")

        rel_img_path = dest_img_path.relative_to(PROJECT_ROOT).as_posix()
        new_image_rel_paths.append(rel_img_path)

        saved_records.append({
            "crop_id": crop_id,
            "source_frame": item["frame_id"],
            "role": item["role"],
            "variance_score": round(item["var_score"], 2),
            "image_path": rel_img_path,
        })

    # Generate new train split file (original 73 frames + negatives)
    orig_train_split = PROJECT_ROOT / "data" / "splits" / "train.txt"
    orig_train_lines = (
        [line.strip() for line in orig_train_split.read_text(encoding="utf-8").splitlines() if line.strip()]
        if orig_train_split.exists()
        else []
    )
    combined_train_lines = orig_train_lines + new_image_rel_paths

    output_split_file.parent.mkdir(parents=True, exist_ok=True)
    output_split_file.write_text("\n".join(combined_train_lines) + "\n", encoding="utf-8")
    print(
        f"[Hard Negative Mining] Generated train split with negatives: {output_split_file} "
        f"({len(orig_train_lines)} original + {len(new_image_rel_paths)} negatives = {len(combined_train_lines)} total)"
    )

    # Generate data/dataset_with_negatives.yaml
    yaml_content = {
        "path": ".",
        "train": output_split_file.relative_to(PROJECT_ROOT).as_posix(),
        "val": "data/splits/val.txt",
        "test": "data/splits/test.txt",
        "nc": 1,
        "names": {0: "vehicle"},
    }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"[Hard Negative Mining] Generated dataset config: {output_yaml}")

    return saved_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hard negative background tiles from train videos.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    parser.add_argument("--labels-dir", type=Path, default=PROJECT_ROOT / "data" / "labels")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data" / "images")
    parser.add_argument("--output-images-dir", type=Path, default=PROJECT_ROOT / "data" / "images_negatives")
    parser.add_argument("--output-labels-dir", type=Path, default=PROJECT_ROOT / "data" / "labels_negatives")
    parser.add_argument("--output-split-file", type=Path, default=PROJECT_ROOT / "data" / "splits" / "train_with_negatives.txt")
    parser.add_argument("--output-yaml", type=Path, default=PROJECT_ROOT / "data" / "dataset_with_negatives.yaml")
    parser.add_argument("--max-negatives", type=int, default=24)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--margin", type=int, default=10)
    args = parser.parse_args()

    # Ensure source images are ready
    ensure_images_available(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        bundle_zip_path=PROJECT_ROOT / "data" / "images_bundle.zip",
    )

    records = generate_hard_negatives(
        manifest_path=args.manifest,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        output_images_dir=args.output_images_dir,
        output_labels_dir=args.output_labels_dir,
        output_split_file=args.output_split_file,
        output_yaml=args.output_yaml,
        max_negatives=args.max_negatives,
        crop_size=args.crop_size,
        stride=args.stride,
        margin=args.margin,
    )

    print(f"\n[Hard Negative Mining Complete] Successfully created {len(records)} tiles in {args.output_images_dir}")


if __name__ == "__main__":
    main()
