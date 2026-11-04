from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.optics import CameraOpticsConfig
from tests.conftest import load_script_module


@pytest.fixture(scope="session")
def audit_mod() -> ModuleType:
    return load_script_module("03_audit_labels.py")


class TestRule2LabelDiffInvariants:
    """Перевірка математичної точності класифікації правок (Unchanged, Fixed, Removed, Added)."""

    def test_exact_classification_of_unchanged_fixed_removed_and_added_boxes(
        self, audit_mod: ModuleType
    ) -> None:
        # 3 сирі бокси від моделі:
        # 1) [0.5, 0.5, 0.10, 0.10] -> залишиться незмінним
        # 2) [0.2, 0.2, 0.10, 0.10] -> буде трохи підправлений людиною (fixed)
        # 3) [0.8, 0.8, 0.05, 0.05] -> хибне спрацювання, буде видалений людиною (removed)
        raw_boxes = np.array(
            [
                [0.50, 0.50, 0.10, 0.10],
                [0.20, 0.20, 0.10, 0.10],
                [0.80, 0.80, 0.05, 0.05],
            ],
            dtype=np.float64,
        )

        # 3 фінальні бокси після перевірки людиною:
        # 1) [0.50, 0.50, 0.10, 0.10] -> той самий (IoU = 1.0 >= 0.98 -> unchanged)
        # 2) [0.21, 0.20, 0.09, 0.10] -> зсунутий/підтягнутий (IoU ~ 0.75 -> fixed)
        # 3) [0.10, 0.90, 0.04, 0.04] -> новий бокс, доданий людиною (added)
        corrected_boxes = np.array(
            [
                [0.50, 0.50, 0.10, 0.10],
                [0.21, 0.20, 0.09, 0.10],
                [0.10, 0.90, 0.04, 0.04],
            ],
            dtype=np.float64,
        )

        diff = audit_mod.compare_frame_labels(raw_boxes, corrected_boxes)
        assert diff["raw_total"] == 3
        assert diff["corrected_total"] == 3
        assert diff["unchanged"] == 1
        assert diff["fixed"] == 1
        assert diff["removed"] == 1
        assert diff["added"] == 1

        # Перевірка фундаментального балансового інваріанта
        assert diff["raw_total"] == diff["unchanged"] + diff["fixed"] + diff["removed"]
        assert diff["corrected_total"] == diff["unchanged"] + diff["fixed"] + diff["added"]


class TestRule4DatasetAuditIntegration:
    """Перевірка агрегації діапазонів дистанцій 0-200m та 200-400m по сплітах."""

    def test_run_dataset_audit_stratifies_near_and_far_bands(
        self, audit_mod: ModuleType, tmp_path: Path
    ) -> None:
        raw_dir = tmp_path / "labels_raw"
        corr_dir = tmp_path / "labels"
        raw_dir.mkdir()
        corr_dir.mkdir()

        # У кадрі 4K (2160x3840):
        # Бокс 1: висота 96 px (h_norm = 96/3840 = 0.025) -> дистанція ~100 м ('0-200m')
        # Бокс 2: висота 32 px (h_norm = 32/3840 = 0.008333) -> дистанція ~300 м ('200-400m')
        (raw_dir / "eval_f00000.txt").write_text("0 0.5 0.5 0.01 0.025000\n", encoding="utf-8")
        (corr_dir / "eval_f00000.txt").write_text(
            "0 0.5 0.5 0.01 0.025000\n0 0.3 0.3 0.005 0.008333\n",
            encoding="utf-8",
        )

        active_df = pd.DataFrame(
            [
                {
                    "frame_id": "eval_f00000",
                    "video_id": "32179597",
                    "role": "eval",
                    "split": "test",
                    "is_active": True,
                    "frame_idx": 0,
                    "timestamp_sec": 0.0,
                    "width": 2160,
                    "height": 3840,
                    "fps": 30.0,
                    "image_path": "data/images/eval_f00000.jpg",
                }
            ]
        )

        report, distances = audit_mod.run_dataset_audit(
            active_df, raw_dir, corr_dir, CameraOpticsConfig()
        )
        assert len(distances) == 2
        assert report["rule2_label_correction_summary"]["added"] == 1
        assert report["rule4_distance_bands_by_split"]["test"]["0-200m"] == 1
        assert report["rule4_distance_bands_by_split"]["test"]["200-400m"] == 1
        assert report["rule4_eval_band_check"]["both_bands_populated"] is True