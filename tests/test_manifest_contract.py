from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.manifest_manager import (
    MANIFEST_COLUMNS,
    get_active_frames,
    label_path_for_frame,
    load_manifest,
    save_manifest,
    sync_manifest_splits,
)


def _sample_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_id": "8968356_f00000",
                "video_id": "8968356",
                "role": "train_A",
                "split": "train",
                "is_active": True,
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "width": 1920,
                "height": 1080,
                "fps": 29.97,
                "image_path": "data/images/8968356_f00000.jpg",
            },
            {
                "frame_id": "8968356_f00030",
                "video_id": "8968356",
                "role": "train_A",
                "split": "val",
                "is_active": True,
                "frame_idx": 30,
                "timestamp_sec": 1.001,
                "width": 1920,
                "height": 1080,
                "fps": 29.97,
                "image_path": "./data/images/8968356_f00030.jpg",
            },
            {
                "frame_id": "32179597_f00000",
                "video_id": "32179597",
                "role": "eval",
                "split": "test",
                "is_active": True,
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "width": 2160,
                "height": 3840,
                "fps": 30.0,
                "image_path": "data\\images\\32179597_f00000.jpg",
            },
        ]
    )


class TestManifestSchemaAndDirtyCsvParsing:
    """Контрактні тести на цілісність схеми та стійкість до ручного редагування CSV."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.csv")

    def test_missing_required_column_raises_value_error(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad_manifest.csv"
        df = _sample_manifest_df().drop(columns=["is_active"])
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_manifest(csv_path)

    def test_valid_boolean_spellings_parsed_strictly_to_bool_dtype(self, tmp_path: Path) -> None:
        """Дозволені лише true/false/1/0/1.0/0.0 (регістр і пробіли ігноруються)."""
        csv_path = tmp_path / "dirty_manifest.csv"
        df = _sample_manifest_df()
        rows = pd.concat([df, df.iloc[:2], df.iloc[:1]], ignore_index=True)
        rows["frame_id"] = [f"f_{i}" for i in range(6)]
        rows["is_active"] = ["True", " FALSE ", "1", "0", "1.0", "0.0"]
        rows.to_csv(csv_path, index=False)

        loaded = load_manifest(csv_path)
        assert loaded["is_active"].dtype == bool
        assert loaded["is_active"].tolist() == [True, False, True, False, True, False]

    @pytest.mark.parametrize("garbage", ["invalid_garbage", "yes", ""])
    def test_garbage_boolean_values_raise_value_error(self, tmp_path: Path, garbage: str) -> None:
        csv_path = tmp_path / "garbage_manifest.csv"
        df = _sample_manifest_df()
        df["is_active"] = ["True", garbage, "True"]
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Invalid boolean value in 'is_active'"):
            load_manifest(csv_path)

    def test_canonical_column_ordering_preserved(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "manifest.csv"
        df = _sample_manifest_df()
        # Перемішуємо колонки
        shuffled_df = df[list(reversed(MANIFEST_COLUMNS))]
        save_manifest(shuffled_df, csv_path)

        loaded = load_manifest(csv_path)
        assert list(loaded.columns) == MANIFEST_COLUMNS


class TestUltralyticsSplitsCompatibilityContract:
    """
    Перевіряє, що згенеровані data/splits/{train,val,test}.txt на 100% сумісні
    з алгоритмом читання шляхів всередині Ultralytics BaseDataset.
    """

    def test_ultralytics_get_img_files_and_img2label_paths_simulation(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "data" / "manifest.csv"
        splits_dir = tmp_path / "data" / "splits"
        df = _sample_manifest_df()
        save_manifest(df, manifest_path)

        counts = sync_manifest_splits(manifest_path, splits_dir)
        assert counts == {"train": 1, "val": 1, "test": 1}

        for split_name in ("train", "val", "test"):
            p = splits_dir / f"{split_name}.txt"
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            line = lines[0]

            # 1. Жоден рядок не повинен починатися з './' (захист від багу parent + os.sep в Ultralytics)
            assert not line.startswith("./"), f"Line '{line}' starts with './'!"
            # 2. Жоден рядок не повинен містити зворотних слешів Windows
            assert "\\" not in line

            # 3. Симуляція Ultralytics BaseDataset.get_img_files:
            parent = str(p.parent) + os.sep
            resolved = line.replace("./", parent) if line.startswith("./") else line
            assert resolved.startswith("data/images/")

            # 4. Симуляція Ultralytics img2label_paths:
            sa, sb = "/images/", "/labels/"
            label_path = sb.join(resolved.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"
            assert label_path.startswith("data/labels/")
            assert label_path.endswith(".txt")

    def test_deactivating_or_moving_frame_in_manifest_updates_splits_atomically(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "data" / "manifest.csv"
        splits_dir = tmp_path / "data" / "splits"
        df = _sample_manifest_df()
        save_manifest(df, manifest_path)

        # Вимикаємо кадр із val та переносимо кадр із train у val
        df.loc[df["frame_id"] == "8968356_f00030", "is_active"] = False
        df.loc[df["frame_id"] == "8968356_f00000", "split"] = "val"
        save_manifest(df, manifest_path)

        counts = sync_manifest_splits(manifest_path, splits_dir)
        assert counts == {"train": 0, "val": 1, "test": 1}

        active_val = get_active_frames(manifest_path, split="val")
        assert len(active_val) == 1
        assert active_val.iloc[0]["frame_id"] == "8968356_f00000"
        assert label_path_for_frame("8968356_f00000") == Path("data/labels/8968356_f00000.txt")