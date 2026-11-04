#!/usr/bin/env python3
"""
Фаза 3 (Rule 2): Автоматична первинна розмітка всіх активних кадрів із manifest.csv
за допомогою open-vocabulary моделі YOLOE-26x із підтримкою 4K-тайлінгу (2x2) та NMS.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest_manager import ensure_images_available, get_active_frames


def compute_tile_windows(
    img_w: int,
    img_h: int,
    grid_rows: int = 2,
    grid_cols: int = 2,
    overlap_ratio: float = 0.15,
) -> list[tuple[int, int, int, int]]:
    """
    Обчислює координати вікон (x1, y1, x2, y2) у пікселях для розбиття кадру
    на сітку grid_rows x grid_cols із заданим коефіцієнтом перекриття overlap_ratio.
    """
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be >= 1")
    if not (0.0 <= overlap_ratio < 0.5):
        raise ValueError("overlap_ratio must be in [0.0, 0.5)")

    base_w = img_w / grid_cols
    base_h = img_h / grid_rows
    pad_x = int(round(base_w * (overlap_ratio / 2.0)))
    pad_y = int(round(base_h * (overlap_ratio / 2.0)))

    windows: list[tuple[int, int, int, int]] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = max(0, int(round(c * base_w)) - pad_x)
            y1 = max(0, int(round(r * base_h)) - pad_y)
            x2 = min(img_w, int(round((c + 1) * base_w)) + pad_x)
            y2 = min(img_h, int(round((r + 1) * base_h)) + pad_y)
            windows.append((x1, y1, x2, y2))

    return windows


def is_box_clipped_by_internal_tile_edge(
    local_box_xyxy: tuple[float, float, float, float],
    tile_xyxy: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    margin_px: float = 3.0,
) -> bool:
    """
    Перевіряє, чи впирається локальний бокс (bx1, by1, bx2, by2) у ВНУТРІШНЮ межу розрізу тайла.
    Якщо межа тайла збігається із зовнішнім краєм усього кадру (0 або img_w/img_h),
    бокс НЕ вважається розрізаним.
    """
    bx1, by1, bx2, by2 = local_box_xyxy
    tx1, ty1, tx2, ty2 = tile_xyxy
    tile_w = tx2 - tx1
    tile_h = ty2 - ty1

    if tx1 > 0 and bx1 <= margin_px:
        return True
    if ty1 > 0 and by1 <= margin_px:
        return True
    if tx2 < img_w and bx2 >= (tile_w - margin_px):
        return True
    if ty2 < img_h and by2 >= (tile_h - margin_px):
        return True
    return False


def numpy_nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    ioa_threshold: float = 0.75,
) -> list[int]:
    """
    Детермінований Non-Maximum Suppression (NMS) на чистому NumPy для злиття боксів
    між full-frame проходом та 4K-тайлами.
    boxes: (N, 4) у форматі [x1, y1, x2, y2]
    scores: (N,)
    Бокс пригнічується, якщо IoU > iou_threshold АБО IoA (inter / min(area_i, area_j)) > ioa_threshold,
    що усуває вкладені дублікати з тайлів і full-frame проходу.
    Повертає список індексів збережених боксів.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(np.float64)
    y1 = boxes[:, 1].astype(np.float64)
    x2 = boxes[:, 2].astype(np.float64)
    y2 = boxes[:, 3].astype(np.float64)

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
        min_area = np.minimum(areas[i], areas[order[1:]])
        ioa = np.divide(inter, min_area, out=np.zeros_like(inter), where=min_area > 0.0)

        remaining_mask = np.where((iou <= iou_threshold) & (ioa <= ioa_threshold))[0]
        order = order[remaining_mask + 1]

    return keep


def xyxy_px_to_yolo_norm(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Переводить піксельні координати [x1, y1, x2, y2] у нормалізований YOLO формат [xc, yc, w, h] in [0, 1]."""
    x1_c = max(0.0, min(float(img_w), x1))
    y1_c = max(0.0, min(float(img_h), y1))
    x2_c = max(0.0, min(float(img_w), x2))
    y2_c = max(0.0, min(float(img_h), y2))

    bw = max(0.0, x2_c - x1_c)
    bh = max(0.0, y2_c - y1_c)
    xc = x1_c + bw / 2.0
    yc = y1_c + bh / 2.0

    return (
        xc / float(img_w),
        yc / float(img_h),
        bw / float(img_w),
        bh / float(img_h),
    )


def format_yolo_label_lines(
    boxes_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> list[str]:
    """Форматує масив боксів [x1, y1, x2, y2] у рядки YOLO .txt файлу."""
    lines: list[str] = []
    for box in boxes_xyxy:
        xc, yc, w, h = xyxy_px_to_yolo_norm(
            float(box[0]), float(box[1]), float(box[2]), float(box[3]), img_w, img_h
        )
        if w > 1e-5 and h > 1e-5:
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def run_inference_on_crop(
    model: Any,
    img_bgr: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Запускає інференс YOLOE на одному зображенні/тайлі та повертає (boxes_xyxy, scores)."""
    import torch

    use_half = torch.cuda.is_available()
    results = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        half=use_half,
        agnostic_nms=True,
        retina_masks=False,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = results[0].boxes.conf.detach().cpu().numpy().astype(np.float32)
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return boxes, scores


def detect_vehicles_multiscale(
    model: Any,
    img_bgr: np.ndarray,
    full_imgsz: int = 1280,
    conf_thres: float = 0.20,
    iou_thres: float = 0.45,
    use_tiling_4k: bool = True,
    tile_grid: tuple[int, int] = (2, 2),
    tile_overlap: float = 0.15,
    tile_imgsz: int = 1024,
) -> np.ndarray:
    """
    Виконує комбінований інференс (Full-frame + 2x2 тайли для всіх кадрів >= 1000px)
    та об'єднує всі знайдені бокси через глобальний NMS.
    """
    img_h, img_w = img_bgr.shape[:2]
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    # 1. Прохід по повному кадру
    ff_boxes, ff_scores = run_inference_on_crop(
        model, img_bgr, imgsz=full_imgsz, conf=conf_thres, iou=iou_thres
    )
    if len(ff_boxes) > 0:
        all_boxes.append(ff_boxes)
        all_scores.append(ff_scores)

    # 2. Додатковий прохід по тайлах 2x2 для кадрів >= 1000px (і 4K, і 1080p train_A)
    if use_tiling_4k and max(img_w, img_h) >= 1000:
        windows = compute_tile_windows(
            img_w, img_h, grid_rows=tile_grid[0], grid_cols=tile_grid[1], overlap_ratio=tile_overlap
        )
        for tx1, ty1, tx2, ty2 in windows:
            crop = img_bgr[ty1:ty2, tx1:tx2]
            t_boxes, t_scores = run_inference_on_crop(
                model, crop, imgsz=tile_imgsz, conf=conf_thres, iou=iou_thres
            )
            for b, s in zip(t_boxes, t_scores):
                if is_box_clipped_by_internal_tile_edge(
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    (tx1, ty1, tx2, ty2),
                    img_w,
                    img_h,
                ):
                    continue
                global_box = np.array(
                    [b[0] + tx1, b[1] + ty1, b[2] + tx1, b[3] + ty1], dtype=np.float32
                )
                all_boxes.append(global_box[None, :])
                all_scores.append(np.array([s], dtype=np.float32))

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32)

    merged_boxes = np.vstack(all_boxes)
    merged_scores = np.concatenate(all_scores)
    keep_indices = numpy_nms_xyxy(merged_boxes, merged_scores, iou_threshold=iou_thres)
    return merged_boxes[keep_indices]


def load_yoloe_model(weights_name: str, prompt_classes: list[str]) -> Any:
    """
    Завантажує open-vocabulary модель YOLOE, встановлює текстові промпти
    та звільняє тимчасовий кеш текстового енкодера з VRAM.
    Без тихого fallback: якщо YOLOE API недоступний — RuntimeError.
    """
    import gc

    import torch

    err_msg = "YOLOE open-vocabulary API requires ultralytics>=8.4.0"
    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise RuntimeError(err_msg) from exc

    model = YOLOE(weights_name)
    if not (hasattr(model, "get_text_pe") and hasattr(model, "set_classes")):
        raise RuntimeError(err_msg)

    text_pe = model.get_text_pe(prompt_classes)
    model.set_classes(prompt_classes, text_pe)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label active frames with YOLOE-26x.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project_config.yaml",
    )
    parser.add_argument(
        "--overwrite-corrected",
        action="store_true",
        help="Also overwrite files in data/labels/ (by default only missing files in data/labels/ are initialized)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest_path"]
    images_dir = PROJECT_ROOT / cfg["paths"]["images_dir"]
    labels_raw_dir = PROJECT_ROOT / cfg["paths"]["labels_raw_dir"]
    labels_dir = PROJECT_ROOT / cfg["paths"]["labels_dir"]
    bundle_zip = PROJECT_ROOT / cfg["paths"]["images_bundle_zip"]
    bundle_url = cfg.get("remote_data", {}).get("images_bundle_url")

    labels_raw_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Автоматично розпаковує або завантажує кадри, якщо запускаємо в Google Colab
    ensure_images_available(
        manifest_path=manifest_path,
        images_dir=images_dir,
        bundle_zip_path=bundle_zip,
        bundle_url=bundle_url,
    )

    al_cfg = cfg["auto_label"]
    model = load_yoloe_model(al_cfg["model_weights"], al_cfg["prompt_classes"])

    active_df = get_active_frames(manifest_path)
    total_boxes = 0

    for _, row in tqdm(
        active_df.iterrows(),
        total=len(active_df),
        desc=f"Auto-labeling ({al_cfg['model_weights']})",
        unit="frame",
    ):
        frame_id = row["frame_id"]
        img_path = PROJECT_ROOT / row["image_path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img_h, img_w = img_bgr.shape[:2]
        boxes_xyxy = detect_vehicles_multiscale(
            model=model,
            img_bgr=img_bgr,
            full_imgsz=int(al_cfg["full_frame_imgsz"]),
            conf_thres=float(al_cfg["conf_threshold"]),
            iou_thres=float(al_cfg["iou_threshold"]),
            use_tiling_4k=bool(al_cfg["use_tiling_4k"]),
            tile_grid=tuple(al_cfg["tile_grid"]),
            tile_overlap=float(al_cfg["tile_overlap_ratio"]),
            tile_imgsz=int(al_cfg["tile_imgsz"]),
        )

        lines = format_yolo_label_lines(boxes_xyxy, img_w=img_w, img_h=img_h, class_id=0)
        total_boxes += len(lines)
        content = "\n".join(lines) + ("\n" if lines else "")

        raw_txt = labels_raw_dir / f"{frame_id}.txt"
        raw_txt.write_text(content, encoding="utf-8")

        corrected_txt = labels_dir / f"{frame_id}.txt"
        if args.overwrite_corrected or not corrected_txt.exists():
            shutil.copy2(raw_txt, corrected_txt)

    print(
        f"\nAuto-labeling complete!\n"
        f"  Frames processed : {len(active_df)}\n"
        f"  Total boxes found: {total_boxes} (avg {total_boxes / max(1, len(active_df)):.1f} vehicles/frame)\n"
        f"  Raw labels saved : {labels_raw_dir}\n"
        f"  Working labels   : {labels_dir}"
    )


if __name__ == "__main__":
    main()