from __future__ import annotations

import sys
import types
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.manifest_manager import (
    RULE1_ERROR,
    ensure_images_available,
    load_manifest,
    parse_yolo_label_file,
    save_manifest,
    sync_manifest_splits,
)
from tests.conftest import load_script_module


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _row(frame_id: str, video_id: str, role: str, split: str, w: int = 1920, h: int = 1080) -> dict:
    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "role": role,
        "split": split,
        "is_active": True,
        "frame_idx": 0,
        "timestamp_sec": 0.0,
        "width": w,
        "height": h,
        "fps": 30.0,
        "image_path": f"data/images/{frame_id}.jpg",
    }


def _two_frame_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("111_f00000", "111", "train_A", "train"),
            _row("32179597_f00000", "32179597", "eval", "test", 2160, 3840),
        ]
    )


def _write_zip(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for n in names:
            zf.writestr(n, b"jpeg")


@pytest.fixture()
def env(tmp_path: Path):
    manifest = tmp_path / "data" / "manifest.csv"
    save_manifest(_two_frame_df(), manifest)
    return {
        "manifest": manifest,
        "images": tmp_path / "data" / "images",
        "zip": tmp_path / "data" / "images_bundle.zip",
        "root": tmp_path,
    }


FRAMES = ["111_f00000.jpg", "32179597_f00000.jpg"]
URL = "https://example.invalid/resolve/main/images_bundle.zip"


# ----------------------------------------------------------------------------
# ensure_images_available
# ----------------------------------------------------------------------------
class TestEnsureImagesAvailableRobustness:
    def test_noop_when_all_frames_present(self, env) -> None:
        env["images"].mkdir(parents=True)
        for n in FRAMES:
            (env["images"] / n).write_bytes(b"x")
        ensure_images_available(env["manifest"], env["images"], env["zip"], None)

    def test_missing_zip_and_no_url_raises(self, env) -> None:
        with pytest.raises(FileNotFoundError):
            ensure_images_available(env["manifest"], env["images"], env["zip"], None)

    def test_placeholder_url_is_rejected(self, env) -> None:
        with pytest.raises(FileNotFoundError):
            ensure_images_available(
                env["manifest"], env["images"], env["zip"], "https://huggingface.co/<HF_USERNAME>/x.zip"
            )

    def test_download_is_atomic_and_extracts(self, env, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_retrieve(url: str, filename: Path):
            assert str(filename).endswith(".zip.part")
            _write_zip(Path(filename), FRAMES)

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
        ensure_images_available(env["manifest"], env["images"], env["zip"], URL)

        assert env["zip"].exists()
        assert not env["zip"].with_suffix(".zip.part").exists()
        assert all((env["images"] / n).exists() for n in FRAMES)

    def test_failed_download_leaves_no_part_and_no_zip(self, env, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_retrieve(url: str, filename: Path):
            Path(filename).write_bytes(b"partial")
            raise OSError("network down")

        monkeypatch.setattr(urllib.request, "urlretrieve", broken_retrieve)
        with pytest.raises(OSError, match="network down"):
            ensure_images_available(env["manifest"], env["images"], env["zip"], URL)

        assert not env["zip"].exists()
        assert not env["zip"].with_suffix(".zip.part").exists()

    def test_corrupted_local_zip_is_deleted_and_redownloaded_once(
        self, env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env["zip"].parent.mkdir(parents=True, exist_ok=True)
        env["zip"].write_bytes(b"<html>not a zip</html>")
        calls: list[str] = []

        def fake_retrieve(url: str, filename: Path):
            calls.append(url)
            _write_zip(Path(filename), FRAMES)

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
        ensure_images_available(env["manifest"], env["images"], env["zip"], URL)

        assert calls == [URL]
        assert zipfile.is_zipfile(env["zip"])
        assert all((env["images"] / n).exists() for n in FRAMES)

    def test_corrupted_zip_without_url_is_kept_and_raises(self, env) -> None:
        env["zip"].parent.mkdir(parents=True, exist_ok=True)
        env["zip"].write_bytes(b"garbage")
        with pytest.raises(zipfile.BadZipFile):
            ensure_images_available(env["manifest"], env["images"], env["zip"], None)
        assert env["zip"].exists()

    def test_redownload_that_is_still_corrupt_raises_and_cleans_up(
        self, env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env["zip"].parent.mkdir(parents=True, exist_ok=True)
        env["zip"].write_bytes(b"garbage")
        monkeypatch.setattr(
            urllib.request, "urlretrieve", lambda url, filename: Path(filename).write_bytes(b"html")
        )
        with pytest.raises(zipfile.BadZipFile):
            ensure_images_available(env["manifest"], env["images"], env["zip"], URL)
        assert not env["zip"].exists()

    def test_incomplete_archive_raises_runtime_error_listing_missing(self, env) -> None:
        _write_zip(env["zip"], ["111_f00000.jpg"])
        with pytest.raises(RuntimeError, match="32179597_f00000"):
            ensure_images_available(env["manifest"], env["images"], env["zip"], None)

    @pytest.mark.parametrize("evil", ["../evil.jpg", "sub/dir.jpg", "..\\evil.jpg"])
    def test_zip_slip_is_blocked(self, env, evil: str) -> None:
        _write_zip(env["zip"], FRAMES + [evil])
        with pytest.raises(ValueError, match="Unsafe path"):
            ensure_images_available(env["manifest"], env["images"], env["zip"], None)
        assert not (env["root"] / "data" / "evil.jpg").exists()
        # ніщо не розпаковано частково
        assert not (env["images"] / FRAMES[0]).exists()


# ----------------------------------------------------------------------------
# manifest Rule 1 guard
# ----------------------------------------------------------------------------
class TestManifestRule1Guard:
    def test_save_rejects_eval_frame_moved_to_train(self, tmp_path: Path) -> None:
        df = _two_frame_df()
        df.loc[df["role"] == "eval", "split"] = "train"
        with pytest.raises(ValueError, match="Rule 1 violation"):
            save_manifest(df, tmp_path / "m.csv")

    def test_save_rejects_train_frame_moved_to_test(self, tmp_path: Path) -> None:
        df = _two_frame_df()
        df.loc[df["role"] == "train_A", "split"] = "test"
        with pytest.raises(ValueError, match="isolated in 'test' split"):
            save_manifest(df, tmp_path / "m.csv")

    def test_eval_video_id_with_wrong_role_is_still_protected(self, tmp_path: Path) -> None:
        df = _two_frame_df()
        df.loc[df["video_id"] == "32179597", ["role", "split"]] = ["train_X", "val"]
        with pytest.raises(ValueError, match="Rule 1 violation"):
            save_manifest(df, tmp_path / "m.csv")

    def test_load_rejects_hand_edited_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "m.csv"
        save_manifest(_two_frame_df(), path)
        text = path.read_text(encoding="utf-8").replace(",test,", ",train,")
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=RULE1_ERROR.split(":")[0]):
            load_manifest(path)

    def test_unknown_split_value_rejected(self, tmp_path: Path) -> None:
        df = _two_frame_df()
        df.loc[0, "split"] = "valid"
        with pytest.raises(ValueError, match="Invalid split"):
            save_manifest(df, tmp_path / "m.csv")

    def test_split_files_are_written_with_lf_only(self, tmp_path: Path) -> None:
        save_manifest(_two_frame_df(), tmp_path / "m.csv")
        sync_manifest_splits(tmp_path / "m.csv", tmp_path / "splits")
        raw = (tmp_path / "splits" / "train.txt").read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")


# ----------------------------------------------------------------------------
# parse_yolo_label_file
# ----------------------------------------------------------------------------
class TestParseYoloLabelFile:
    def test_missing_and_empty_files_give_empty_array(self, tmp_path: Path) -> None:
        assert parse_yolo_label_file(tmp_path / "none.txt").shape == (0, 4)
        (tmp_path / "e.txt").write_text("\n \n", encoding="utf-8")
        assert parse_yolo_label_file(tmp_path / "e.txt").shape == (0, 4)

    def test_valid_file_parsed(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.2\n0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
        out = parse_yolo_label_file(tmp_path / "a.txt")
        assert out.shape == (2, 4) and out[1].tolist() == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.parametrize(
        "line",
        [
            "0 0.5 0.5 0.1",  # <5 токенів
            "1 0.5 0.5 0.1 0.1",  # class_id != 0
            "0 0.5 0.5 0.0 0.1",  # w == 0
            "0 0.5 0.5 0.1 -0.1",  # h < 0
            "0 1.5 0.5 0.1 0.1",  # xc поза [0,1]
            "0 0.5 0.5 0.1 1.2",  # h поза [0,1]
            "0 abc 0.5 0.1 0.1",  # не число
            "0 nan 0.5 0.1 0.1",  # NaN
        ],
    )
    def test_bad_line_raises_with_file_and_line_number(self, tmp_path: Path, line: str) -> None:
        p = tmp_path / "bad.txt"
        p.write_text("0 0.5 0.5 0.1 0.1\n" + line + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.txt:2"):
            parse_yolo_label_file(p)


# ----------------------------------------------------------------------------
# 02_auto_label: NMS + YOLOE loader
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def al() -> ModuleType:
    return load_script_module("02_auto_label.py")


class TestNmsIoa:
    BOXES = np.array(
        [
            [0.0, 0.0, 100.0, 100.0],  # великий, score 0.9
            [55.0, 10.0, 105.0, 60.0],  # IoA = 0.90, IoU ~ 0.22
            [300.0, 300.0, 340.0, 320.0],
        ],
        dtype=np.float32,
    )
    SCORES = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    def test_nested_box_suppressed_by_ioa_although_iou_is_low(self, al: ModuleType) -> None:
        assert al.numpy_nms_xyxy(self.BOXES, self.SCORES, iou_threshold=0.45) == [0, 2]

    def test_disabling_ioa_keeps_nested_box(self, al: ModuleType) -> None:
        keep = al.numpy_nms_xyxy(self.BOXES, self.SCORES, iou_threshold=0.45, ioa_threshold=1.0)
        assert keep == [0, 1, 2]

    def test_empty_input(self, al: ModuleType) -> None:
        assert al.numpy_nms_xyxy(np.empty((0, 4)), np.empty((0,))) == []

    def test_degenerate_zero_area_boxes_do_not_crash(self, al: ModuleType) -> None:
        boxes = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
        assert len(al.numpy_nms_xyxy(boxes, np.array([0.5, 0.4], dtype=np.float32))) == 2


class TestLoadYoloeModel:
    def test_calls_get_text_pe_then_set_classes(self, al: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.get_text_pe.return_value = "PE"
        fake_cls = MagicMock(return_value=model)
        fake_mod = types.ModuleType("ultralytics")
        fake_mod.YOLOE = fake_cls
        monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)

        out = al.load_yoloe_model("w.pt", ["car", "bus"])

        fake_cls.assert_called_once_with("w.pt")
        model.get_text_pe.assert_called_once_with(["car", "bus"])
        model.set_classes.assert_called_once_with(["car", "bus"], "PE")
        assert out is model

    def test_model_without_open_vocab_methods_raises(self, al: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        class Plain:
            def __init__(self, *_a, **_k) -> None:
                pass

        fake_mod = types.ModuleType("ultralytics")
        fake_mod.YOLOE = Plain
        monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
        with pytest.raises(RuntimeError, match=r"ultralytics>=8\.4\.0"):
            al.load_yoloe_model("w.pt", ["car"])

    def test_missing_yoloe_class_raises(self, al: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "ultralytics", types.ModuleType("ultralytics"))
        with pytest.raises(RuntimeError, match=r"ultralytics>=8\.4\.0"):
            al.load_yoloe_model("w.pt", ["car"])


class TestTileSeamAndFormatting:
    def test_box_on_internal_seam_dropped_but_box_on_outer_edge_kept(self, al: ModuleType) -> None:
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        windows = al.compute_tile_windows(3840, 2160)
        calls = {"n": 0}

        def fake_infer(model, crop, imgsz, conf, iou):
            calls["n"] += 1
            if calls["n"] == 1:  # full frame: нічого
                return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
            if calls["n"] == 2:  # верхній лівий тайл: бокс впирається у правий внутрішній розріз
                tw = windows[0][2] - windows[0][0]
                return np.array([[tw - 40.0, 100.0, tw - 1.0, 140.0]], np.float32), np.array([0.9], np.float32)
            if calls["n"] == 3:  # верхній правий тайл: бокс у зовнішнього правого краю кадру — валідний
                tw = windows[1][2] - windows[1][0]
                return np.array([[tw - 40.0, 100.0, tw - 1.0, 140.0]], np.float32), np.array([0.8], np.float32)
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32)

        with patch.object(al, "run_inference_on_crop", side_effect=fake_infer):
            merged = al.detect_vehicles_multiscale(MagicMock(), img)

        assert len(merged) == 1
        assert merged[0][0] == pytest.approx(windows[1][0] + windows[1][2] - windows[1][0] - 40.0)

    def test_run_inference_on_crop_handles_no_results(self, al: ModuleType) -> None:
        model = MagicMock()
        model.predict.return_value = []
        b, s = al.run_inference_on_crop(model, np.zeros((10, 10, 3), np.uint8), 640, 0.2, 0.45)
        assert b.shape == (0, 4) and s.shape == (0,)

    def test_run_inference_on_crop_handles_empty_boxes(self, al: ModuleType) -> None:
        res = MagicMock()
        res.boxes.xyxy.detach.return_value.cpu.return_value.numpy.return_value = np.empty((0, 4))
        res.boxes.conf.detach.return_value.cpu.return_value.numpy.return_value = np.empty((0,))
        model = MagicMock()
        model.predict.return_value = [res]
        b, s = al.run_inference_on_crop(model, np.zeros((10, 10, 3), np.uint8), 640, 0.2, 0.45)
        assert len(b) == 0 and len(s) == 0

    def test_format_yolo_label_lines_and_clamping(self, al: ModuleType) -> None:
        lines = al.format_yolo_label_lines(
            np.array([[-10.0, -10.0, 50.0, 50.0], [10.0, 10.0, 10.0, 30.0]]), img_w=100, img_h=100
        )
        assert lines == ["0 0.250000 0.250000 0.500000 0.500000"]

    def test_tile_window_and_overlap_validation(self, al: ModuleType) -> None:
        with pytest.raises(ValueError):
            al.compute_tile_windows(100, 100, grid_rows=0)
        with pytest.raises(ValueError):
            al.compute_tile_windows(100, 100, overlap_ratio=0.6)

    def test_small_frame_skips_tiling(self, al: ModuleType) -> None:
        with patch.object(
            al, "run_inference_on_crop", return_value=(np.empty((0, 4), np.float32), np.empty(0, np.float32))
        ) as m:
            out = al.detect_vehicles_multiscale(MagicMock(), np.zeros((480, 640, 3), np.uint8))
        assert m.call_count == 1 and out.shape == (0, 4)


# ----------------------------------------------------------------------------
# 03_audit_labels
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def au() -> ModuleType:
    return load_script_module("03_audit_labels.py")


class TestAuditHardening:
    def test_threshold_masking_beats_plain_hungarian(self, au: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        IoU = [[0.35, 0.29], [0.29, 0.00]]. Без маски угорський алгоритм обрав би (A-D, B-C) із сумою 0.58,
        але обидві пари < 0.30 => 0 валідних збігів. З маскою обирається (A-C) = 0.35 => 1 збіг.
        """
        monkeypatch.setattr(au, "compute_iou_matrix", lambda a, b: np.array([[0.35, 0.29], [0.29, 0.0]]))
        dummy = np.zeros((2, 4))
        diff = au.compare_frame_labels(dummy, dummy)
        assert diff["fixed"] == 1 and diff["unchanged"] == 0
        assert diff["removed"] == 1 and diff["added"] == 1

    def test_empty_side_shortcuts(self, au: ModuleType) -> None:
        e, b = np.empty((0, 4)), np.array([[0.5, 0.5, 0.1, 0.1]])
        assert au.compare_frame_labels(e, e)["raw_total"] == 0
        assert au.compare_frame_labels(e, b)["added"] == 1
        assert au.compare_frame_labels(b, e)["removed"] == 1

    def test_iou_matrix_empty_and_identity(self, au: ModuleType) -> None:
        b = np.array([[0.5, 0.5, 0.2, 0.2]])
        assert au.compute_iou_matrix(b, np.empty((0, 4))).shape == (1, 0)
        assert au.compute_iou_matrix(b, b)[0, 0] == pytest.approx(1.0)
        assert au.yolo_norm_to_xyxy(np.empty((0, 4))).shape == (0, 4)

    @pytest.mark.parametrize("missing", ["raw", "corrected"])
    def test_missing_label_file_raises(self, au: ModuleType, tmp_path: Path, missing: str) -> None:
        raw, corr = tmp_path / "raw", tmp_path / "corr"
        raw.mkdir()
        corr.mkdir()
        if missing != "raw":
            (raw / "f1.txt").write_text("", encoding="utf-8")
        if missing != "corrected":
            (corr / "f1.txt").write_text("", encoding="utf-8")
        df = pd.DataFrame([_row("f1", "111", "train_A", "train")])
        from src.optics import CameraOpticsConfig

        with pytest.raises(FileNotFoundError, match="f1"):
            au.run_dataset_audit(df, raw, corr, CameraOpticsConfig())

    def test_empty_label_file_is_valid_empty_frame(self, au: ModuleType, tmp_path: Path) -> None:
        raw, corr = tmp_path / "raw", tmp_path / "corr"
        raw.mkdir()
        corr.mkdir()
        (raw / "f1.txt").write_text("", encoding="utf-8")
        (corr / "f1.txt").write_text("", encoding="utf-8")
        df = pd.DataFrame([_row("f1", "111", "train_A", "train")])
        from src.optics import CameraOpticsConfig

        report, dists = au.run_dataset_audit(df, raw, corr, CameraOpticsConfig())
        assert dists == [] and report["rule4_eval_band_check"]["both_bands_populated"] is False

    def test_parse_is_reexported(self, au: ModuleType) -> None:
        assert au.parse_yolo_label_file is parse_yolo_label_file

    def test_plot_and_main_end_to_end(self, au: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import yaml

        from src.manifest_manager import save_manifest as sm

        (tmp_path / "data" / "labels_raw").mkdir(parents=True)
        (tmp_path / "data" / "labels").mkdir(parents=True)
        df = _two_frame_df()
        sm(df, tmp_path / "data" / "manifest.csv")
        for fid, line in (("111_f00000", "0.5 0.5 0.05 0.05"), ("32179597_f00000", "0.5 0.5 0.01 0.02")):
            (tmp_path / "data" / "labels_raw" / f"{fid}.txt").write_text(f"0 {line}\n", encoding="utf-8")
            (tmp_path / "data" / "labels" / f"{fid}.txt").write_text(f"0 {line}\n", encoding="utf-8")
        cfg = {
            "paths": {
                "manifest_path": "data/manifest.csv",
                "labels_raw_dir": "data/labels_raw",
                "labels_dir": "data/labels",
                "audit_report_path": "data/audit_report.json",
                "assets_dir": "assets",
            },
            "optics": {
                "fov_long_deg": 84.0,
                "car_length_m": 4.5,
                "car_width_m": 1.8,
                "estimation_method": "max_side",
            },
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setattr(au, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["03", "--config", str(cfg_path)])
        au.main()
        assert (tmp_path / "data" / "audit_report.json").stat().st_size > 0
        assert (tmp_path / "assets" / "distance_distribution.png").exists()
        au.save_distance_distribution_plot([], tmp_path / "assets" / "empty.png")


# ----------------------------------------------------------------------------
# 01_extract_frames
# ----------------------------------------------------------------------------
class TestExtractFramesFailFast:
    def _cap(self, mod, read_results, grab_ok=True):
        cap = MagicMock()
        cap.isOpened.return_value = True
        props = {
            mod.cv2.CAP_PROP_FPS: 10.0,
            mod.cv2.CAP_PROP_FRAME_COUNT: 30,
            mod.cv2.CAP_PROP_FRAME_WIDTH: 64,
            mod.cv2.CAP_PROP_FRAME_HEIGHT: 36,
        }
        cap.get.side_effect = lambda p: props[p]
        cap.read.side_effect = read_results
        cap.grab.return_value = grab_ok
        return cap

    def _call(self, mod, tmp_path: Path, cap):
        (tmp_path / "v.mp4").write_bytes(b"x")
        with (
            patch.object(mod.cv2, "VideoCapture", return_value=cap),
            patch.object(mod.cv2, "imwrite", return_value=True),
        ):
            return mod.extract_video_frames(
                video_id="1",
                video_meta={"filename": "v.mp4", "role": "train_A", "is_eval": False},
                raw_videos_dir=tmp_path,
                images_dir=tmp_path,
                target_fps=1.0,
                train_ratio=0.8,
                jpeg_quality=90,
            )

    def test_decode_failure_raises_and_releases_capture(self, extract_frames_mod: ModuleType, tmp_path: Path) -> None:
        frame = np.zeros((36, 64, 3), np.uint8)
        cap = self._cap(extract_frames_mod, [(True, frame), (False, None)])
        with pytest.raises(RuntimeError, match="Failed to decode frame 10"):
            self._call(extract_frames_mod, tmp_path, cap)
        cap.release.assert_called_once()

    def test_grab_failure_raises_and_releases_capture(self, extract_frames_mod: ModuleType, tmp_path: Path) -> None:
        frame = np.zeros((36, 64, 3), np.uint8)
        cap = self._cap(extract_frames_mod, [(True, frame)] * 3, grab_ok=False)
        with pytest.raises(RuntimeError, match="Failed to grab frame"):
            self._call(extract_frames_mod, tmp_path, cap)
        cap.release.assert_called_once()

    def test_missing_video_file_raises(self, extract_frames_mod: ModuleType, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_frames_mod.extract_video_frames(
                "1", {"filename": "nope.mp4", "role": "r", "is_eval": False}, tmp_path, tmp_path, 1.0, 0.8, 90
            )

    def test_sampling_edge_cases(self, extract_frames_mod: ModuleType) -> None:
        assert extract_frames_mod.compute_sample_frame_indices(0, 30.0, 1.0) == []
        assert extract_frames_mod.compute_sample_frame_indices(10, -1.0, 1.0) == []
        assert extract_frames_mod.assign_temporal_splits(1, False) == ["train"]
        assert extract_frames_mod.assign_temporal_splits(2, False) == ["train", "val"]
        assert extract_frames_mod.assign_temporal_splits(3, True) == ["test"] * 3

    def test_summary_prints_config_driven_optics(self, extract_frames_mod: ModuleType, capsys) -> None:
        df = _two_frame_df()
        cfg = {
            "optics": {
                "fov_long_deg": 70.0,
                "car_length_m": 5.0,
                "car_width_m": 2.0,
                "estimation_method": "max_side",
            }
        }
        extract_frames_mod.print_dataset_and_optics_summary(df, cfg)
        out = capsys.readouterr().out
        assert "FOV_long = 70.0" in out and "L_car = 5.0m" in out and "W_car = 2.0m" in out


# ----------------------------------------------------------------------------
# 02_auto_label.main()
# ----------------------------------------------------------------------------
class TestAutoLabelMain:
    def _setup(self, tmp_path: Path):
        import yaml

        save_manifest(_two_frame_df(), tmp_path / "data" / "manifest.csv")
        cfg = {
            "paths": {
                "manifest_path": "data/manifest.csv",
                "images_dir": "data/images",
                "labels_raw_dir": "data/labels_raw",
                "labels_dir": "data/labels",
                "images_bundle_zip": "data/images_bundle.zip",
            },
            "remote_data": {"images_bundle_url": URL},
            "auto_label": {
                "model_weights": "w.pt",
                "prompt_classes": ["car"],
                "conf_threshold": 0.2,
                "iou_threshold": 0.45,
                "full_frame_imgsz": 640,
                "use_tiling_4k": False,
                "tile_grid": [2, 2],
                "tile_overlap_ratio": 0.15,
                "tile_imgsz": 640,
            },
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return cfg_path

    def _run(self, al: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_path: Path, *extra: str):
        boxes = np.array([[100.0, 100.0, 300.0, 200.0]], dtype=np.float32)
        monkeypatch.setattr(al, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(al, "ensure_images_available", lambda **kw: None)
        monkeypatch.setattr(al, "load_yoloe_model", lambda w, c: MagicMock())
        monkeypatch.setattr(al, "detect_vehicles_multiscale", lambda **kw: boxes)
        monkeypatch.setattr(al.cv2, "imread", lambda p: np.zeros((1080, 1920, 3), np.uint8))
        monkeypatch.setattr(sys, "argv", ["02", "--config", str(cfg_path), *extra])
        al.main()

    def test_main_writes_raw_and_initial_labels_and_keeps_human_edits(
        self, al: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = self._setup(tmp_path)
        self._run(al, tmp_path, monkeypatch, cfg_path)

        raw = tmp_path / "data" / "labels_raw" / "111_f00000.txt"
        corr = tmp_path / "data" / "labels" / "111_f00000.txt"
        assert raw.read_text(encoding="utf-8") == corr.read_text(encoding="utf-8")
        assert raw.read_text(encoding="utf-8").startswith("0 0.104167 0.138889")

        corr.write_text("", encoding="utf-8")  # людина видалила бокс
        self._run(al, tmp_path, monkeypatch, cfg_path)
        assert corr.read_text(encoding="utf-8") == ""  # не перезаписано

        self._run(al, tmp_path, monkeypatch, cfg_path, "--overwrite-corrected")
        assert corr.read_text(encoding="utf-8") == raw.read_text(encoding="utf-8")

    def test_main_raises_on_unreadable_image(
        self, al: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = self._setup(tmp_path)
        monkeypatch.setattr(al, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(al, "ensure_images_available", lambda **kw: None)
        monkeypatch.setattr(al, "load_yoloe_model", lambda w, c: MagicMock())
        monkeypatch.setattr(al.cv2, "imread", lambda p: None)
        monkeypatch.setattr(sys, "argv", ["02", "--config", str(cfg_path)])
        with pytest.raises(RuntimeError, match="Failed to read image"):
            al.main()
