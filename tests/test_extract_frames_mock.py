from __future__ import annotations

from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPhaseDriftAndTemporalSamplingMath:
    """Перевірка точності дискретизації кадрів на дробових FPS (29.97 та 23.98)."""

    @pytest.mark.parametrize(
        "native_fps,duration_sec",
        [
            (29.97, 19.0),
            (24.00, 31.0),
            (23.98, 16.0),
            (29.97, 24.0),
            (30.00, 30.0),
        ],
    )
    def test_zero_cumulative_phase_drift_across_all_project_videos(
        self,
        extract_frames_mod: ModuleType,
        native_fps: float,
        duration_sec: float,
    ) -> None:
        """
        Інваріант: для кожної цільової секунди k = 0, 1, ..., N-1
        різниця між фактичним часом кадру (frame_idx / native_fps) та ідеальним часом k
        не повинна перевищувати пів кадру: |t_actual - k| <= 0.5 / native_fps.
        """
        total_frames = int(round(native_fps * duration_sec))
        samples = extract_frames_mod.compute_sample_frame_indices(
            total_frames=total_frames,
            native_fps=native_fps,
            target_fps=1.0,
        )

        assert len(samples) == int(duration_sec)
        max_allowed_error_sec = 0.5001 / native_fps

        for k, (frame_idx, actual_ts) in enumerate(samples):
            assert 0 <= frame_idx < total_frames
            assert abs(actual_ts - float(k)) <= max_allowed_error_sec


class TestRule1TemporalSplitInvariants:
    """Перевірка дотримання Rule 1: відсутність data leakage та 100% ізоляція eval."""

    def test_train_video_chronological_80_20_split(
        self, extract_frames_mod: ModuleType
    ) -> None:
        splits = extract_frames_mod.assign_temporal_splits(
            num_frames=20, is_eval_video=False, train_ratio=0.80
        )
        assert splits.count("train") == 16
        assert splits.count("val") == 4
        assert splits.count("test") == 0

        # Усі індекси train суворо передують усім індексам val
        train_indices = [i for i, s in enumerate(splits) if s == "train"]
        val_indices = [i for i, s in enumerate(splits) if s == "val"]
        assert max(train_indices) < min(val_indices)

    def test_eval_video_goes_100_percent_to_test(
        self, extract_frames_mod: ModuleType
    ) -> None:
        splits = extract_frames_mod.assign_temporal_splits(
            num_frames=30, is_eval_video=True, train_ratio=0.80
        )
        assert splits == ["test"] * 30


class TestVideoCaptureMocks:
    """Mock-тести декодування відео без читання реальних 4K файлів із диска."""

    def test_extract_video_frames_uses_grab_for_non_sampled_frames(
        self, extract_frames_mod: ModuleType, tmp_path: Path
    ) -> None:
        """
        Перевіряє, що на проміжних кадрах викликається швидкий cap.grab(),
        а важкий декодер cap.read() викликається ЛИШЕ на цільових секундах.
        """
        raw_dir = tmp_path / "raw_videos"
        img_dir = tmp_path / "images"
        raw_dir.mkdir()
        img_dir.mkdir()

        dummy_video = raw_dir / "test.mp4"
        dummy_video.write_bytes(b"dummy")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # 10 FPS, 30 кадрів = 3 секунди -> має бути 3 виклики read() (кадри 0, 10, 20) і 18 grab()
        prop_map = {
            extract_frames_mod.cv2.CAP_PROP_FPS: 10.0,
            extract_frames_mod.cv2.CAP_PROP_FRAME_COUNT: 30,
            extract_frames_mod.cv2.CAP_PROP_FRAME_WIDTH: 1920,
            extract_frames_mod.cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }
        mock_cap.get.side_effect = lambda prop: prop_map[prop]
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.grab.return_value = True

        video_meta = {"filename": "test.mp4", "role": "train_A", "is_eval": False}

        with (
            patch.object(extract_frames_mod.cv2, "VideoCapture", return_value=mock_cap),
            patch.object(extract_frames_mod.cv2, "imwrite", return_value=True) as mock_imwrite,
        ):
            records = extract_frames_mod.extract_video_frames(
                video_id="999",
                video_meta=video_meta,
                raw_videos_dir=raw_dir,
                images_dir=img_dir,
                target_fps=1.0,
                train_ratio=0.8,
                jpeg_quality=95,
                overwrite_images=True,
            )

        assert len(records) == 3
        assert [r["frame_idx"] for r in records] == [0, 10, 20]
        assert mock_cap.read.call_count == 3
        assert mock_cap.grab.call_count == 18
        assert mock_imwrite.call_count == 3
        mock_cap.release.assert_called_once()

    def test_unopenable_video_raises_runtime_error(
        self, extract_frames_mod: ModuleType, tmp_path: Path
    ) -> None:
        raw_dir = tmp_path / "raw_videos"
        raw_dir.mkdir()
        (raw_dir / "broken.mp4").write_bytes(b"corrupt")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch.object(extract_frames_mod.cv2, "VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Failed to open video file"):
                extract_frames_mod.extract_video_frames(
                    video_id="000",
                    video_meta={"filename": "broken.mp4", "role": "eval", "is_eval": True},
                    raw_videos_dir=raw_dir,
                    images_dir=tmp_path,
                    target_fps=1.0,
                    train_ratio=0.8,
                    jpeg_quality=95,
                )