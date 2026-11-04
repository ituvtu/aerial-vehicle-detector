from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "test"]

VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})
EVAL_VIDEO_ID = "32179597"
EVAL_VIDEO_IDS: frozenset[str] = frozenset({"32179597", "13763017"})
EVAL_ROLE = "eval"

MANIFEST_COLUMNS: list[str] = [
    "frame_id",
    "video_id",
    "role",
    "split",
    "is_active",
    "frame_idx",
    "timestamp_sec",
    "width",
    "height",
    "fps",
    "image_path",
]

_BOOL_MAP: dict[str, bool] = {
    "true": True,
    "1": True,
    "1.0": True,
    "false": False,
    "0": False,
    "0.0": False,
}

RULE1_ERROR = "Rule 1 violation: eval frames must stay isolated in 'test' split"


def validate_rule1_isolation(df: pd.DataFrame) -> None:
    """
    Rule 1 isolation guard:
    - Every frame with video_id in EVAL_VIDEO_IDS (or role.startswith("eval")) MUST have split == "test" and role.startswith("eval").
    - Every frame with split == "test" MUST have video_id in EVAL_VIDEO_IDS and role.startswith("eval").
    - Every frame with video_id not in EVAL_VIDEO_IDS MUST have split in ("train", "val") and not role.startswith("eval").
    """
    splits = df["split"].astype(str).str.strip()
    bad_splits = sorted(set(splits) - VALID_SPLITS)
    if bad_splits:
        raise ValueError(f"Invalid split value(s) {bad_splits}; allowed: {sorted(VALID_SPLITS)}")

    roles = df["role"].astype(str).str.strip()
    video_ids = df["video_id"].astype(str).str.strip()

    is_eval_vid = video_ids.isin(EVAL_VIDEO_IDS)
    is_eval_role = roles.str.startswith("eval")
    is_eval = is_eval_vid | is_eval_role

    # Check 1: All frames with role.startswith('eval') or known EVAL_VIDEO_IDS must have split == 'test'
    if (splits[is_eval] != "test").any():
        raise ValueError(RULE1_ERROR)

    # Check 2: All test frames must have role.startswith('eval')
    if (~is_eval_role[splits == "test"]).any():
        raise ValueError(RULE1_ERROR)

    # Check 3: Non-eval frames must have split in ('train', 'val') and cannot be in EVAL_VIDEO_IDS
    if (splits[~is_eval_role].isin(["test"])).any() or (is_eval_vid[~is_eval_role]).any():
        raise ValueError(RULE1_ERROR)


def _parse_strict_bool(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapped = normalized.map(_BOOL_MAP)
    if mapped.isna().any():
        bad = sorted(set(normalized[mapped.isna()]))
        raise ValueError(f"Invalid boolean value in 'is_active': {bad}")
    return mapped.astype(bool)


def load_manifest(manifest_path: str | Path = "data/manifest.csv") -> pd.DataFrame:
    """
    Читає та валідує data/manifest.csv.
    Гарантує правильні типи колонок (суворий bool для is_active) та інваріант Rule 1.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest file not found at '{path}'. Run scripts/01_extract_frames.py first."
        )

    # Спочатку перевіряємо заголовок до примусового приведення типів
    raw_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_cols = set(MANIFEST_COLUMNS) - set(raw_df.columns)
    if missing_cols:
        raise ValueError(f"Manifest '{path}' is missing required columns: {missing_cols}")

    df = pd.DataFrame()
    df["frame_id"] = raw_df["frame_id"].astype(str).str.strip()
    df["video_id"] = raw_df["video_id"].astype(str).str.strip()
    df["role"] = raw_df["role"].astype(str).str.strip()
    df["split"] = raw_df["split"].astype(str).str.strip()
    df["is_active"] = _parse_strict_bool(raw_df["is_active"])
    df["frame_idx"] = raw_df["frame_idx"].astype(int)
    df["timestamp_sec"] = raw_df["timestamp_sec"].astype(float)
    df["width"] = raw_df["width"].astype(int)
    df["height"] = raw_df["height"].astype(int)
    df["fps"] = raw_df["fps"].astype(float)
    df["image_path"] = raw_df["image_path"].astype(str).str.strip()

    validate_rule1_isolation(df)
    return df[MANIFEST_COLUMNS]


def save_manifest(
    df: pd.DataFrame,
    manifest_path: str | Path = "data/manifest.csv",
) -> None:
    """Зберігає таблицю manifest.csv у канонічному порядку колонок (з перевіркою Rule 1)."""
    missing_cols = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required manifest columns: {missing_cols}")

    validate_rule1_isolation(df)

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df[MANIFEST_COLUMNS].copy()
    df_out["is_active"] = df_out["is_active"].astype(bool)
    df_out.to_csv(path, index=False, float_format="%.4f")


def get_active_frames(
    manifest_path: str | Path = "data/manifest.csv",
    split: SplitName | None = None,
) -> pd.DataFrame:
    """
    Повертає лише активні кадри (is_active == True) з manifest.csv.
    Якщо вказано split ('train', 'val', або 'test') — додатково фільтрує за сплітом.
    """
    df = load_manifest(manifest_path)
    active_df = df[df["is_active"]].copy()
    if split is not None:
        active_df = active_df[active_df["split"] == split].copy()
    return active_df.reset_index(drop=True)


def _normalize_split_path(raw_path: str) -> str:
    """
    Нормалізує шлях до POSIX і прибирає префікс './',
    не чіпаючи абсолютні шляхи ('/...') та відносні ('../...').
    """
    posix_path = raw_path.replace("\\", "/").strip()
    while posix_path.startswith("./"):
        posix_path = posix_path.removeprefix("./")
    return posix_path


def sync_manifest_splits(
    manifest_path: str | Path = "data/manifest.csv",
    splits_dir: str | Path = "data/splits",
) -> dict[str, int]:
    """
    Читає data/manifest.csv, фільтрує is_active == True і пише
    файли data/splits/{train,val,test}.txt для Ultralytics YOLO (завжди з LF-переводами рядків).
    """
    df = load_manifest(manifest_path)
    active_df = df[df["is_active"]]

    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_df = active_df[active_df["split"] == split_name]
        image_paths = [_normalize_split_path(p) for p in split_df["image_path"].tolist()]
        split_file = out_dir / f"{split_name}.txt"
        content = "\n".join(image_paths) + ("\n" if image_paths else "")
        with open(split_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        counts[split_name] = len(image_paths)

    return counts


def label_path_for_frame(
    frame_id: str,
    labels_dir: str | Path = "data/labels",
) -> Path:
    """Повертає шлях до .txt файлу міток для заданого frame_id."""
    return Path(labels_dir) / f"{frame_id}.txt"


def parse_yolo_label_file(txt_path: Path) -> np.ndarray:
    """
    Читає YOLO .txt файл і повертає масив форми (N, 4) зі стовпцями [xc, yc, w, h] у [0, 1].
    Якщо файл не існує або порожній — повертає порожній масив (0, 4).
    Некоректний рядок (<5 токенів, class_id != 0, w/h <= 0, координати поза [0, 1])
    викликає ValueError із вказанням файлу та номера рядка.
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return np.empty((0, 4), dtype=np.float64)

    boxes: list[list[float]] = []
    for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        where = f"{txt_path}:{line_no}"
        if len(parts) < 5:
            raise ValueError(f"{where}: expected >= 5 tokens, got {len(parts)}: '{line.strip()}'")
        try:
            class_id = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:5])
        except ValueError as exc:
            raise ValueError(f"{where}: non-numeric token in '{line.strip()}'") from exc
        if class_id != 0 or float(parts[0]) != 0.0:
            raise ValueError(f"{where}: class_id must be 0, got {parts[0]}")
        if not all(np.isfinite(v) for v in (xc, yc, w, h)):
            raise ValueError(f"{where}: non-finite coordinate")
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"{where}: non-positive box size w={w}, h={h}")
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            raise ValueError(f"{where}: coordinates must be within [0, 1]")
        boxes.append([xc, yc, w, h])

    if not boxes:
        return np.empty((0, 4), dtype=np.float64)
    return np.array(boxes, dtype=np.float64)


def _download_atomic(url: str, dest: Path) -> None:
    """Завантажує url у '<dest>.part' і атомарно перейменовує у dest; при помилці .part видаляється."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    print(f"[Auto-Fetch] Downloading {url} -> {dest} ...")
    try:
        urllib.request.urlretrieve(url, part_path)
        part_path.replace(dest)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise


def _extract_bundle_safely(zip_path: Path, img_dir: Path) -> None:
    """
    Перевіряє валідність ZIP (is_zipfile + testzip) та імена членів (zip-slip), потім розпаковує.
    Піднімає zipfile.BadZipFile для пошкодженого архіву та ValueError для небезпечних імен.
    """
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"'{zip_path}' is not a valid ZIP archive")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupted member '{bad_member}' in '{zip_path}'")
        for member in zf.namelist():
            if ".." in member or Path(member).name != member:
                raise ValueError(f"Unsafe path in archive '{zip_path}': '{member}'")
        print(f"[Auto-Fetch] Extracting '{zip_path}' into '{img_dir}' ...")
        zf.extractall(img_dir)


def ensure_images_available(
    manifest_path: str | Path = "data/manifest.csv",
    images_dir: str | Path = "data/images",
    bundle_zip_path: str | Path = "data/images_bundle.zip",
    bundle_url: str | None = None,
) -> None:
    """
    Гарантує, що всі кадри з manifest.csv лежать у data/images/.
    Алгоритм:
      1) Якщо всі кадри вже є — нічого не робимо.
      2) Якщо є локальний images_bundle.zip — використовуємо його.
      3) Інакше (або якщо локальний архів пошкоджений) завантажуємо bundle_url (Hugging Face)
         атомарно через .part-файл; пошкоджений архів видаляється і завантажується повторно один раз.
      4) Після розпаковки перевіряємо, що всі кадри фізично існують, інакше RuntimeError.
    """
    df = get_active_frames(manifest_path)
    img_dir = Path(images_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    def _missing() -> list[str]:
        return [fid for fid in df["frame_id"].tolist() if not (img_dir / f"{fid}.jpg").exists()]

    missing_files = _missing()
    if not missing_files:
        return

    zip_path = Path(bundle_zip_path)
    url_ok = bool(bundle_url) and "<HF_USERNAME>" not in str(bundle_url)

    if not zip_path.exists():
        if not url_ok:
            raise FileNotFoundError(
                f"{len(missing_files)} frames are missing in '{img_dir}', and '{zip_path}' was not found. "
                f"Run 'scripts/01_extract_frames.py' locally or set 'remote_data.images_bundle_url' in project_config.yaml."
            )
        _download_atomic(str(bundle_url), zip_path)

    try:
        _extract_bundle_safely(zip_path, img_dir)
    except zipfile.BadZipFile:
        if not url_ok:
            raise
        zip_path.unlink(missing_ok=True)
        print(f"[Auto-Fetch] WARNING: '{zip_path}' is corrupted; deleting it and re-downloading once.")
        _download_atomic(str(bundle_url), zip_path)
        try:
            _extract_bundle_safely(zip_path, img_dir)
        except zipfile.BadZipFile:
            zip_path.unlink(missing_ok=True)
            raise

    still_missing = _missing()
    if still_missing:
        raise RuntimeError(
            f"{len(still_missing)} frames are still missing in '{img_dir}' after extraction: {still_missing[:20]}"
        )


DEFAULT_HF_RAW_VIDEO_URL = (
    "https://huggingface.co/datasets/ituvtu/aerial-vehicle-detector-data/resolve/main/{video_id}.mp4"
)


def ensure_raw_eval_video_available(
    video_id: str = "32179597",
    raw_videos_dir: str | Path = "data/raw_videos",
    video_url: str | None = None,
) -> Path:
    """
    Ensures that raw evaluation video {video_id}.mp4 is present in data/raw_videos/.
    1) If local file exists and size > 1 MB, returns its path.
    2) Otherwise (e.g. in Google Colab), downloads from Hugging Face atomically via .part file.
    """
    vdir = Path(raw_videos_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    target_path = vdir / f"{video_id}.mp4"

    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return target_path

    url = video_url or DEFAULT_HF_RAW_VIDEO_URL.format(video_id=video_id)
    print(f"[Raw Video] Downloading '{video_id}.mp4' from {url} ...")
    _download_atomic(url, target_path)

    if not target_path.exists() or target_path.stat().st_size <= 1_000_000:
        raise RuntimeError(
            f"Failed to fetch valid raw video '{target_path}' (size: {target_path.stat().st_size if target_path.exists() else 0} bytes)"
        )
    print(f"[Raw Video] Successfully fetched '{video_id}.mp4' ({target_path.stat().st_size / 1e6:.1f} MB)")
    return target_path

