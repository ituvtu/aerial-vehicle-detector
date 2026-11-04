#!/usr/bin/env python3
"""
Фаза 1-2: Нарізка кадрів із сирих відео, формування data/manifest.csv,
генерація індексів data/splits/{train,val,test}.txt та пакування data/images_bundle.zip.
"""

from __future__ import annotations

import argparse
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

# Забезпечуємо імпорт пакету src при прямому запуску скрипта
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest_manager import (
    MANIFEST_COLUMNS,
    load_manifest,
    save_manifest,
    sync_manifest_splits,
)
from src.optics import CameraOpticsConfig, compute_focal_length_px, expected_box_px_at_distance


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sample_frame_indices(
    total_frames: int,
    native_fps: float,
    target_fps: float,
) -> list[tuple[int, float]]:
    """
    Обчислює список (frame_idx, timestamp_sec) для вибірки з частотою target_fps.
    Використовує round(duration_sec * target_fps), щоб уникати відсікання останньої
    секунди, коли total_frames / native_fps дає 18.985с на 29.97 FPS.
    """
    if total_frames <= 0 or native_fps <= 0 or target_fps <= 0:
        return []

    duration_sec = total_frames / native_fps
    num_samples = int(round(duration_sec * target_fps))

    samples: list[tuple[int, float]] = []
    seen_indices: set[int] = set()

    for k in range(num_samples):
        target_time_sec = k / target_fps
        frame_idx = int(round(target_time_sec * native_fps))
        if frame_idx >= total_frames:
            break
        if frame_idx not in seen_indices:
            actual_time_sec = frame_idx / native_fps
            samples.append((frame_idx, actual_time_sec))
            seen_indices.add(frame_idx)

    return samples


def assign_temporal_splits(
    num_frames: int,
    is_eval_video: bool,
    train_ratio: float = 0.80,
) -> list[str]:
    """
    Призначає спліт кожному кадру відео за часовим порядком (Rule 1):
    - Якщо is_eval_video=True: усі кадри отримують 'test'.
    - Якщо is_eval_video=False: перші train_ratio (80%) кадрів отримують 'train',
      останні (20%) кадрів отримують 'val'.
    """
    if is_eval_video:
        return ["test"] * num_frames

    n_train = int(round(num_frames * train_ratio))
    # Гарантуємо щонайменше 1 кадр у val, якщо у відео >= 2 кадрів
    if num_frames >= 2:
        n_train = max(1, min(num_frames - 1, n_train))

    n_val = num_frames - n_train
    return (["train"] * n_train) + (["val"] * n_val)


def extract_video_frames(
    video_id: str,
    video_meta: dict[str, Any],
    raw_videos_dir: Path,
    images_dir: Path,
    target_fps: float,
    train_ratio: float,
    jpeg_quality: int,
    overwrite_images: bool = False,
) -> list[dict[str, Any]]:
    """Витягує кадри з одного відеофайлу та повертає рядки для manifest.csv."""
    video_path = raw_videos_dir / video_meta["filename"]
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: '{video_path}' (role: {video_meta['role']})"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: '{video_path}'")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    samples = compute_sample_frame_indices(
        total_frames=total_frames,
        native_fps=native_fps,
        target_fps=target_fps,
    )
    splits = assign_temporal_splits(
        num_frames=len(samples),
        is_eval_video=bool(video_meta["is_eval"]),
        train_ratio=train_ratio,
    )

    records: list[dict[str, Any]] = []
    target_indices_map = {idx: (pos, ts) for pos, (idx, ts) in enumerate(samples)}
    max_target_idx = samples[-1][0] if samples else -1

    pbar = tqdm(
        total=len(samples),
        desc=f"Extracting {video_id} ({video_meta['role']}, {width}x{height} @ {native_fps:.2f}fps)",
        unit="frame",
    )

    current_idx = 0
    try:
        while current_idx <= max_target_idx:
            if current_idx in target_indices_map:
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"Failed to decode frame {current_idx} from {video_path}")

                pos, timestamp_sec = target_indices_map[current_idx]
                split_name = splits[pos]
                frame_id = f"{video_id}_f{current_idx:05d}"
                rel_img_path = Path("data/images") / f"{frame_id}.jpg"
                abs_img_path = images_dir / f"{frame_id}.jpg"

                if overwrite_images or not abs_img_path.exists():
                    cv2.imwrite(
                        str(abs_img_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )

                actual_h, actual_w = frame.shape[:2]
                records.append(
                    {
                        "frame_id": frame_id,
                        "video_id": str(video_id),
                        "role": str(video_meta["role"]),
                        "split": split_name,
                        "is_active": True,
                        "frame_idx": int(current_idx),
                        "timestamp_sec": round(float(timestamp_sec), 4),
                        "width": int(actual_w),
                        "height": int(actual_h),
                        "fps": round(native_fps, 4),
                        "image_path": rel_img_path.as_posix(),
                    }
                )
                pbar.update(1)
            else:
                # Швидкий пропуск проміжних кадрів без декодування (grab швидший за read)
                if not cap.grab():
                    raise RuntimeError(f"Failed to grab frame {current_idx} from {video_path}")

            current_idx += 1
    finally:
        pbar.close()
        cap.release()
    return records


def build_images_bundle_zip(
    df: pd.DataFrame,
    project_root: Path,
    bundle_zip_path: Path,
) -> float:
    """
    Пакує усі активні зображення з manifest.csv у ZIP-архів (без повторного стиснення JPEG)
    для завантаження на Google Drive. Повертає розмір архіву в МБ.
    """
    bundle_zip_path.parent.mkdir(parents=True, exist_ok=True)
    active_df = df[df["is_active"]]

    with zipfile.ZipFile(bundle_zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for rel_path_str in tqdm(
            active_df["image_path"].tolist(),
            desc="Packing data/images_bundle.zip",
            unit="img",
        ):
            abs_path = project_root / rel_path_str
            if not abs_path.exists():
                raise FileNotFoundError(f"Missing extracted frame on disk: {abs_path}")
            # Зберігаємо всередині архіву лише ім'я файлу {frame_id}.jpg для простого розпакування в data/images/
            zf.write(abs_path, arcname=abs_path.name)

    size_mb = bundle_zip_path.stat().st_size / (1024 * 1024)
    return size_mb


def print_dataset_and_optics_summary(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Виводить підсумкову таблицю по відео, сплітах та оптичних розмірах боксів."""
    print("\n" + "=" * 78)
    print("DATASET MANIFEST & OPTICS VERIFICATION SUMMARY")
    print("=" * 78)

    summary = (
        df[df["is_active"]]
        .groupby(["video_id", "role", "width", "height", "fps", "split"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ("train", "val", "test"):
        if col not in summary.columns:
            summary[col] = 0
    summary["total"] = summary["train"] + summary["val"] + summary["test"]
    print(summary[["train", "val", "test", "total"]].to_string())

    optics_cfg = CameraOpticsConfig(
        fov_long_deg=float(cfg["optics"]["fov_long_deg"]),
        car_length_m=float(cfg["optics"]["car_length_m"]),
        car_width_m=float(cfg["optics"]["car_width_m"]),
        estimation_method=cfg["optics"]["estimation_method"],
    )

    print("\n" + "-" * 78)
    print(
        f"PINHOLE OPTICS REFERENCE (FOV_long = {optics_cfg.fov_long_deg:.1f}°, "
        f"L_car = {optics_cfg.car_length_m}m, W_car = {optics_cfg.car_width_m}m):"
    )
    print("-" * 78)
    unique_res = df[["width", "height"]].drop_duplicates().to_records(index=False)
    for w, h in unique_res:
        f_px = compute_focal_length_px(int(w), int(h), optics_cfg.fov_long_deg)
        l_200, w_200 = expected_box_px_at_distance(200.0, int(w), int(h), optics_cfg)
        l_400, w_400 = expected_box_px_at_distance(400.0, int(w), int(h), optics_cfg)
        orientation = "VERTICAL 9:16" if h > w else "HORIZONTAL 16:9"
        print(
            f"  Resolution {w}x{h} ({orientation}) | f_px = {f_px:.1f} px:\n"
            f"    @ 200m boundary -> car box ≈ {l_200:.1f} x {w_200:.1f} px\n"
            f"    @ 400m boundary -> car box ≈ {l_400:.1f} x {w_400:.1f} px"
        )
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames, build manifest.csv, generate splits, and pack images_bundle.zip."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project_config.yaml",
        help="Path to project_config.yaml",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Re-encode and overwrite existing .jpg frames on disk",
    )
    parser.add_argument(
        "--preserve-existing-flags",
        action="store_true",
        help="Preserve manual changes to 'split' and 'is_active' if manifest.csv already exists",
    )
    parser.add_argument(
        "--skip-zip",
        action="store_true",
        help="Skip creating data/images_bundle.zip",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_videos_dir = PROJECT_ROOT / cfg["paths"]["raw_videos_dir"]
    images_dir = PROJECT_ROOT / cfg["paths"]["images_dir"]
    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest_path"]
    splits_dir = PROJECT_ROOT / cfg["paths"]["splits_dir"]
    bundle_zip_path = PROJECT_ROOT / cfg["paths"]["images_bundle_zip"]

    images_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / cfg["paths"]["labels_raw_dir"]).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / cfg["paths"]["labels_dir"]).mkdir(parents=True, exist_ok=True)

    target_fps = float(cfg["extraction"]["target_fps"])
    train_ratio = float(cfg["extraction"]["train_temporal_ratio"])
    jpeg_quality = int(cfg["extraction"]["jpeg_quality"])

    all_records: list[dict[str, Any]] = []
    for video_id, video_meta in cfg["videos"].items():
        records = extract_video_frames(
            video_id=str(video_id),
            video_meta=video_meta,
            raw_videos_dir=raw_videos_dir,
            images_dir=images_dir,
            target_fps=target_fps,
            train_ratio=train_ratio,
            jpeg_quality=jpeg_quality,
            overwrite_images=args.overwrite_images,
        )
        all_records.extend(records)

    new_df = pd.DataFrame(all_records, columns=MANIFEST_COLUMNS)

    if args.preserve_existing_flags and manifest_path.exists():
        old_df = load_manifest(manifest_path)
        old_flags = old_df.set_index("frame_id")[["split", "is_active"]]
        for idx, row in new_df.iterrows():
            fid = row["frame_id"]
            if fid in old_flags.index:
                new_df.at[idx, "split"] = old_flags.loc[fid, "split"]
                new_df.at[idx, "is_active"] = bool(old_flags.loc[fid, "is_active"])

    save_manifest(new_df, manifest_path)
    split_counts = sync_manifest_splits(manifest_path, splits_dir)

    print_dataset_and_optics_summary(new_df, cfg)
    print(
        f"Manifest saved to: {manifest_path} ({len(new_df)} total frames)\n"
        f"Synced Ultralytics split files in {splits_dir}/:\n"
        f"  train.txt : {split_counts['train']} frames\n"
        f"  val.txt   : {split_counts['val']} frames\n"
        f"  test.txt  : {split_counts['test']} frames"
    )

    if not args.skip_zip:
        size_mb = build_images_bundle_zip(new_df, PROJECT_ROOT, bundle_zip_path)
        print(f"Created Colab bundle: {bundle_zip_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()