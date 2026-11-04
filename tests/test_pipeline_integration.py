from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from src.manifest_manager import load_manifest, save_manifest
from tests.conftest import create_synthetic_mp4


class TestEndToEndExtractionPipeline:
    """
    Інтеграційний тест повного циклу scripts/01_extract_frames.py:
    від сирих .mp4 (горизонтальних і вертикальних) до manifest.csv, splits/*.txt та images_bundle.zip.
    """

    def test_full_pipeline_execution_and_flag_preservation(
        self,
        extract_frames_mod: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 1. Готуємо ізольоване робоче середовище
        raw_dir = tmp_path / "data" / "raw_videos"
        create_synthetic_mp4(raw_dir / "train_a.mp4", width=640, height=360, fps=10.0, num_frames=50)
        create_synthetic_mp4(raw_dir / "eval.mp4", width=360, height=640, fps=10.0, num_frames=30)

        config_data = {
            "project": {"name": "test", "seed": 42, "num_classes": 1, "class_name": "vehicle"},
            "paths": {
                "raw_videos_dir": "data/raw_videos",
                "images_dir": "data/images",
                "labels_raw_dir": "data/labels_raw",
                "labels_dir": "data/labels",
                "splits_dir": "data/splits",
                "manifest_path": "data/manifest.csv",
                "images_bundle_zip": "data/images_bundle.zip",
            },
            "videos": {
                "111": {"filename": "train_a.mp4", "role": "train_A", "scene": "hwy", "is_eval": False},
                "222": {"filename": "eval.mp4", "role": "eval", "scene": "city", "is_eval": True},
            },
            "extraction": {"target_fps": 1.0, "train_temporal_ratio": 0.80, "jpeg_quality": 90},
            "optics": {
                "fov_long_deg": 84.0,
                "car_length_m": 4.5,
                "car_width_m": 1.8,
                "estimation_method": "max_side",
            },
        }
        cfg_path = tmp_path / "configs" / "project_config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

        # Підміняємо PROJECT_ROOT у модулі на tmp_path
        monkeypatch.setattr(extract_frames_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["01_extract_frames.py", "--config", str(cfg_path)])

        # 2. Перший запуск пайплайну
        extract_frames_mod.main()

        manifest_path = tmp_path / "data" / "manifest.csv"
        splits_dir = tmp_path / "data" / "splits"
        bundle_zip = tmp_path / "data" / "images_bundle.zip"

        assert manifest_path.exists()
        assert bundle_zip.exists()

        df = load_manifest(manifest_path)
        # 5 кадрів із train_a (4 train, 1 val) + 3 кадри з eval (3 test) = 8 кадрів
        assert len(df) == 8
        assert len(df[df["split"] == "train"]) == 4
        assert len(df[df["split"] == "val"]) == 1
        assert len(df[df["split"] == "test"]) == 3

        # Перевіряємо орієнтацію кадрів у manifest.csv (16:9 для train_A та 9:16 для eval)
        eval_rows = df[df["role"] == "eval"]
        assert (eval_rows["width"] == 360).all()
        assert (eval_rows["height"] == 640).all()

        # Перевіряємо вміст ZIP-архіву для Colab
        with zipfile.ZipFile(bundle_zip, "r") as zf:
            zipped_names = set(zf.namelist())
            assert len(zipped_names) == 8
            assert "111_f00000.jpg" in zipped_names
            assert "222_f00000.jpg" in zipped_names

        # 3. Вносимо ручну правку в manifest.csv (вимикаємо 1 кадр) і перезапускаємо з --preserve-existing-flags
        df.loc[df["frame_id"] == "111_f00000", "is_active"] = False
        save_manifest(df, manifest_path)

        monkeypatch.setattr(
            sys,
            "argv",
            ["01_extract_frames.py", "--config", str(cfg_path), "--preserve-existing-flags"],
        )
        extract_frames_mod.main()

        # Перевіряємо, що ручна правка збереглася і синхронізувалася у train.txt та images_bundle.zip
        df_reloaded = load_manifest(manifest_path)
        assert not bool(
            df_reloaded.loc[df_reloaded["frame_id"] == "111_f00000", "is_active"].iloc[0]
        )

        train_lines = (splits_dir / "train.txt").read_text(encoding="utf-8").strip().splitlines()
        assert len(train_lines) == 3

        with zipfile.ZipFile(bundle_zip, "r") as zf:
            assert len(zf.namelist()) == 7
            assert "111_f00000.jpg" not in zf.namelist()