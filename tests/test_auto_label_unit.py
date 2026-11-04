from __future__ import annotations

import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.manifest_manager import ensure_images_available, save_manifest
from tests.conftest import load_script_module


@pytest.fixture(scope="session")
def auto_label_mod() -> ModuleType:
    return load_script_module("02_auto_label.py")


class TestTilingGeometryAndEdgeClipping:
    """Перевірка геометрії 4K-тайлінгу (2x2) для горизонтальних (16:9) та вертикальних (9:16) кадрів."""

    @pytest.mark.parametrize("w,h", [(3840, 2160), (2160, 3840)])
    def test_tile_windows_cover_100_percent_of_frame_with_overlap(
        self, auto_label_mod: ModuleType, w: int, h: int
    ) -> None:
        windows = auto_label_mod.compute_tile_windows(w, h, grid_rows=2, grid_cols=2, overlap_ratio=0.15)
        assert len(windows) == 4

        # Перевіряємо, що зовнішні межі тайлів точно збігаються з краями кадру (0 та w/h)
        assert windows[0][0] == 0 and windows[0][1] == 0
        assert windows[3][2] == w and windows[3][3] == h

        # Перевіряємо наявність перекриття між лівим і правим тайлом
        left_x2 = windows[0][2]
        right_x1 = windows[1][0]
        assert left_x2 > right_x1, "Tiles must overlap horizontally!"

    def test_internal_tile_edge_clipping_filter(self, auto_label_mod: ModuleType) -> None:
        # Верхній лівий тайл (0, 0, 2064, 1161) на кадрі 3840x2160:
        # Ліва (x=0) і верхня (y=0) межі — ЗОВНІШНІ, права (x=2064) і нижня (y=1161) — ВНУТРІШНІ.
        tile = (0, 0, 2064, 1161)

        # Бокс біля зовнішнього лівого краю кадру (x1=1) НЕ має відкидатися
        assert not auto_label_mod.is_box_clipped_by_internal_tile_edge(
            (1.0, 100.0, 50.0, 140.0), tile, 3840, 2160
        )
        # Бокс, що впирається у внутрішній правий розріз тайла (x2=2063), ПОВИНЕН відкидатися
        assert auto_label_mod.is_box_clipped_by_internal_tile_edge(
            (2020.0, 100.0, 2063.0, 140.0), tile, 3840, 2160
        )


class TestNumpyNmsAndCoordinateConversion:
    """Перевірка NMS-злиття дублікатів на стиках тайлів та нормалізації координат YOLO."""

    def test_numpy_nms_merges_overlapping_boxes_and_keeps_highest_score(
        self, auto_label_mod: ModuleType
    ) -> None:
        boxes = np.array(
            [
                [100.0, 100.0, 150.0, 120.0],  # score 0.75
                [102.0, 101.0, 151.0, 120.0],  # дублікат того ж авто з тайла, score 0.92
                [500.0, 500.0, 540.0, 520.0],  # інше авто далеко, score 0.60
            ],
            dtype=np.float32,
        )
        scores = np.array([0.75, 0.92, 0.60], dtype=np.float32)

        keep = auto_label_mod.numpy_nms_xyxy(boxes, scores, iou_threshold=0.45)
        assert keep == [1, 2]

    def test_multiscale_inference_merges_full_frame_and_tiles_for_4k(
        self, auto_label_mod: ModuleType
    ) -> None:
        img_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
        mock_model = MagicMock()

        # Нехай кожен виклик predict повертає один бокс у центрі поданого кропу (300, 300, 348, 320)
        mock_result = MagicMock()
        mock_result.boxes.xyxy.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [[300.0, 300.0, 348.0, 320.0]], dtype=np.float32
        )
        mock_result.boxes.conf.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [0.85], dtype=np.float32
        )
        mock_model.predict.return_value = [mock_result]

        merged = auto_label_mod.detect_vehicles_multiscale(
            model=mock_model,
            img_bgr=img_4k,
            use_tiling_4k=True,
        )
        # 1 прохід full-frame + 4 проходи по тайлах = 5 викликів predict.
        # Оскільки у верхньому лівому тайлі зміщення (tx1=0, ty1=0), його бокс (300,300,348,320)
        # повністю збігається з full-frame боксом і зливається через NMS: залишається 4 унікальні бокси!
        assert mock_model.predict.call_count == 5
        assert len(merged) == 4


class TestAutoUnpackBundleContract:
    """Перевіряє, що ensure_images_available автоматично розпаковує images_bundle.zip."""

    def test_ensure_images_available_extracts_from_local_bundle(self, tmp_path: Path) -> None:
        import pandas as pd

        manifest_path = tmp_path / "data" / "manifest.csv"
        images_dir = tmp_path / "data" / "images"
        bundle_zip = tmp_path / "data" / "images_bundle.zip"

        df = pd.DataFrame(
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
                }
            ]
        )
        save_manifest(df, manifest_path)

        # Створюємо bundle.zip із тестовим файлом
        with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("8968356_f00000.jpg", b"fake_jpeg_bytes")

        assert not (images_dir / "8968356_f00000.jpg").exists()
        ensure_images_available(manifest_path, images_dir, bundle_zip)
        assert (images_dir / "8968356_f00000.jpg").exists()