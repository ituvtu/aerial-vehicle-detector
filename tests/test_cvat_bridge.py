from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.cvat_bridge import (
    compute_labels_tree_hash,
    create_labels_raw_lock,
    ensure_working_dirs_saved,
    export_yolo_1_1_for_cvat,
    import_yolo_1_1_from_cvat,
    load_ledger,
    materialize_version,
    pack_signed_raw_bundle,
    register_snapshot,
    unpack_signed_raw_bundle,
    verify_labels_lock,
)
from src.manifest_manager import save_manifest


def _make_dummy_manifest(manifest_path: Path, frame_ids: list[str]) -> pd.DataFrame:
    records = []
    for i, fid in enumerate(frame_ids):
        is_eval = fid.startswith("32179597")
        records.append(
            {
                "frame_id": fid,
                "video_id": "32179597" if is_eval else "8968356",
                "role": "eval" if is_eval else "train_A",
                "split": "test" if is_eval else "train",
                "is_active": True,
                "frame_idx": i * 30,
                "timestamp_sec": float(i),
                "width": 2160 if is_eval else 1920,
                "height": 3840 if is_eval else 1080,
                "fps": 30.0,
                "image_path": f"data/images/{fid}.jpg",
            }
        )
    df = pd.DataFrame(records)
    save_manifest(df, manifest_path)
    return df


def _populate_labels(labels_dir: Path, frame_ids: list[str], box_line: str = "0 0.500000 0.500000 0.100000 0.200000\n") -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids:
        (labels_dir / f"{fid}.txt").write_text(box_line, encoding="utf-8", newline="\n")


@pytest.fixture
def cvat_env(tmp_path: Path):
    manifest_path = tmp_path / "data" / "manifest.csv"
    frame_ids = ["8968356_f00000", "8968356_f00030", "32179597_f00000"]
    _make_dummy_manifest(manifest_path, frame_ids)

    labels_raw_dir = tmp_path / "data" / "labels_raw"
    labels_curated_dir = tmp_path / "data" / "labels"
    _populate_labels(labels_raw_dir, frame_ids)
    _populate_labels(labels_curated_dir, frame_ids)

    return {
        "root": tmp_path,
        "manifest": manifest_path,
        "frame_ids": frame_ids,
        "labels_raw": labels_raw_dir,
        "labels_curated": labels_curated_dir,
        "registry": tmp_path / "data" / "registry",
        "lock": tmp_path / "data" / "labels_raw_lock.json",
        "audit": tmp_path / "data" / "audit_report.json",
        "artifacts": tmp_path / "artifacts",
        "cvat": tmp_path / "data" / "cvat",
    }


# -----------------------------------------------------------------------------
# 1. Merkle Tree Hash & Tamper Detection
# -----------------------------------------------------------------------------
def test_merkle_tree_hash_and_tamper_detection(cvat_env):
    labels_raw = cvat_env["labels_raw"]
    manifest = cvat_env["manifest"]
    lock_file = cvat_env["lock"]

    info = create_labels_raw_lock(labels_raw, manifest, lock_file)
    assert info["num_frames"] == 3
    assert info["total_boxes"] == 3
    assert verify_labels_lock(labels_raw, lock_file, manifest) is True

    # Tamper 1 character
    target_file = labels_raw / "8968356_f00000.txt"
    orig_content = target_file.read_text(encoding="utf-8")
    target_file.write_text("0 0.510000 0.500000 0.100000 0.200000\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="Labels lock mismatch"):
        verify_labels_lock(labels_raw, lock_file, manifest)

    # Missing file
    target_file.unlink()
    with pytest.raises(FileNotFoundError, match="Missing 1 label files"):
        compute_labels_tree_hash(labels_raw, manifest)

    # Extra file
    target_file.write_text(orig_content, encoding="utf-8", newline="\n")
    (labels_raw / "extra.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Found unexpected .txt files"):
        compute_labels_tree_hash(labels_raw, manifest)


# -----------------------------------------------------------------------------
# 2. Append-Only Non-Overwriting Snapshots
# -----------------------------------------------------------------------------
def test_append_only_non_overwriting_snapshots(cvat_env):
    manifest = cvat_env["manifest"]
    registry = cvat_env["registry"]
    labels_raw = cvat_env["labels_raw"]
    labels_curated = cvat_env["labels_curated"]
    frame_ids = cvat_env["frame_ids"]

    # Register raw v001
    s1 = register_snapshot(labels_raw, manifest, registry, kind="raw")
    assert s1["seq"] == 1
    assert "v001_raw" in s1["version_id"]

    # Modify curated v1 & register
    (labels_curated / f"{frame_ids[0]}.txt").write_text("0 0.600000 0.600000 0.200000 0.300000\n", encoding="utf-8")
    s2 = register_snapshot(labels_curated, manifest, registry, kind="curated")
    assert s2["seq"] == 2
    assert "v002_curated" in s2["version_id"]

    # Modify curated v2 (add an object) & register
    (labels_curated / f"{frame_ids[1]}.txt").write_text(
        "0 0.500000 0.500000 0.100000 0.200000\n0 0.100000 0.100000 0.050000 0.050000\n", encoding="utf-8"
    )
    s3 = register_snapshot(labels_curated, manifest, registry, kind="curated")
    assert s3["seq"] == 3
    assert "v003_curated" in s3["version_id"]

    snapshots_dir = registry / "snapshots"
    d1 = snapshots_dir / s1["version_id"]
    d2 = snapshots_dir / s2["version_id"]
    d3 = snapshots_dir / s3["version_id"]

    for d in (d1, d2, d3):
        assert d.exists()
        assert (d / "labels").is_dir()
        assert (d / "lock.json").exists()
        assert (d / "audit_report.json").exists()

    ledger = load_ledger(registry)
    assert len(ledger["versions"]) == 3
    assert ledger["active_raw_version"] == s1["version_id"]
    assert ledger["active_curated_version"] == s3["version_id"]


# -----------------------------------------------------------------------------
# 3. Dirty-State Auto-Freeze Guard
# -----------------------------------------------------------------------------
def test_dirty_state_autofreeze_guard(cvat_env):
    manifest = cvat_env["manifest"]
    registry = cvat_env["registry"]
    labels_raw = cvat_env["labels_raw"]
    labels_curated = cvat_env["labels_curated"]
    lock_file = cvat_env["lock"]
    audit_file = cvat_env["audit"]
    frame_ids = cvat_env["frame_ids"]

    # Register initial raw
    s_raw = register_snapshot(labels_raw, manifest, registry, kind="raw")

    # User manually edits labels on disk without registering!
    unregistered_edit = labels_curated / f"{frame_ids[0]}.txt"
    unregistered_edit.write_text("0 0.900000 0.900000 0.050000 0.050000\n", encoding="utf-8")

    # Trigger materialize_version of raw: ensure_working_dirs_saved MUST freeze dirty edits!
    materialize_version(
        version_id=s_raw["version_id"],
        registry_dir=registry,
        manifest_path=manifest,
        labels_raw_dir=labels_raw,
        labels_curated_dir=labels_curated,
        lock_path=lock_file,
        audit_report_path=audit_file,
    )

    ledger = load_ledger(registry)
    autosave_entries = [v for v in ledger["versions"] if v["kind"] == "autosave_curated"]
    assert len(autosave_entries) == 1
    autosave_vid = autosave_entries[0]["version_id"]

    # Verify that the manual edit exists inside the autosaved snapshot
    saved_txt = registry / "snapshots" / autosave_vid / "labels" / f"{frame_ids[0]}.txt"
    assert "0.900000" in saved_txt.read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# 4. Checkout Time-Travel
# -----------------------------------------------------------------------------
def test_checkout_time_travel(cvat_env):
    manifest = cvat_env["manifest"]
    registry = cvat_env["registry"]
    labels_raw = cvat_env["labels_raw"]
    labels_curated = cvat_env["labels_curated"]
    lock_file = cvat_env["lock"]
    audit_file = cvat_env["audit"]
    frame_ids = cvat_env["frame_ids"]

    register_snapshot(labels_raw, manifest, registry, kind="raw")

    (labels_curated / f"{frame_ids[0]}.txt").write_text("0 0.111111 0.111111 0.100000 0.100000\n", encoding="utf-8")
    s2 = register_snapshot(labels_curated, manifest, registry, kind="curated")

    (labels_curated / f"{frame_ids[0]}.txt").write_text("0 0.222222 0.222222 0.100000 0.100000\n", encoding="utf-8")
    s3 = register_snapshot(labels_curated, manifest, registry, kind="curated")

    # Time travel back to s2
    materialize_version(s2["version_id"], registry, manifest, labels_raw, labels_curated, lock_file, audit_file)
    assert "0.111111" in (labels_curated / f"{frame_ids[0]}.txt").read_text(encoding="utf-8")

    # Time travel forward to s3
    materialize_version(s3["version_id"], registry, manifest, labels_raw, labels_curated, lock_file, audit_file)
    assert "0.222222" in (labels_curated / f"{frame_ids[0]}.txt").read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# 5. CVAT YOLO 1.1 Format Strictness
# -----------------------------------------------------------------------------
def test_cvat_yolo_1_1_format_strictness(cvat_env):
    manifest = cvat_env["manifest"]
    labels_curated = cvat_env["labels_curated"]
    output_zip = cvat_env["root"] / "export_cvat.zip"
    frame_ids = cvat_env["frame_ids"]

    # Frame 2 has 0 boxes
    (labels_curated / f"{frame_ids[2]}.txt").write_text("", encoding="utf-8")

    export_yolo_1_1_for_cvat(labels_curated, manifest, output_zip)
    assert output_zip.exists()

    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        assert "obj.names" in names
        assert "obj.data" in names
        assert "train.txt" in names
        assert f"obj_train_data/{frame_ids[0]}.txt" in names

        # Check line endings strictly LF, no CRLF
        for name in names:
            data = zf.read(name)
            assert b"\r" not in data, f"Found CR in {name}"

        obj_names = zf.read("obj.names").decode("utf-8")
        assert obj_names == "vehicle\n"

        train_txt = zf.read("train.txt").decode("utf-8")
        assert f"data/obj_train_data/{frame_ids[0]}.jpg\n" in train_txt

        f2_data = zf.read(f"obj_train_data/{frame_ids[2]}.txt").decode("utf-8")
        assert f2_data == ""


# -----------------------------------------------------------------------------
# 6. CLI Mock Tests
# -----------------------------------------------------------------------------
def test_cvat_cli_pipeline_subcommands(cvat_env, monkeypatch):
    import scripts.cvat_cli_pipeline as cli

    root = cvat_env["root"]
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)

    # 1. pack-raw-artifact
    monkeypatch.setattr(sys, "argv", ["cvat_cli_pipeline.py", "pack-raw-artifact"])
    cli.main()
    packed = list((root / "artifacts").glob("raw_autolabel_*.zip"))
    assert len(packed) == 1

    # 2. import-to-cvat with --skip-cli
    monkeypatch.setattr(sys, "argv", ["cvat_cli_pipeline.py", "import-to-cvat", "--skip-cli"])
    cli.main()
    assert (root / "data" / "cvat" / "exports").is_dir()

    # import-to-cvat running mocked cvat-cli subprocess
    mock_run = MagicMock()
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Created task ID: 42\n")
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(cli, "ensure_images_available", lambda **kw: None)

    monkeypatch.setattr(
        sys, "argv", ["cvat_cli_pipeline.py", "import-to-cvat", "--auth", "user:pass"]
    )
    cli.main()
    assert mock_run.call_count == 2
    state = json.loads((root / "data" / "cvat" / "cvat_state.json").read_text(encoding="utf-8"))
    assert state["task_id"] == 42

    # 3. pull-from-cvat with --from-zip
    dummy_cvat_zip = root / "dummy_cvat.zip"
    export_yolo_1_1_for_cvat(cvat_env["labels_curated"], cvat_env["manifest"], dummy_cvat_zip)

    monkeypatch.setattr(
        sys, "argv", ["cvat_cli_pipeline.py", "pull-from-cvat", "--from-zip", str(dummy_cvat_zip)]
    )
    cli.main()
    ledger = load_ledger(root / "data" / "registry")
    assert any("curated" in v["version_id"] for v in ledger["versions"])

    # 4. list-versions
    monkeypatch.setattr(sys, "argv", ["cvat_cli_pipeline.py", "list-versions"])
    cli.main()

    # 5. checkout
    target_vid = ledger["versions"][0]["version_id"]
    monkeypatch.setattr(sys, "argv", ["cvat_cli_pipeline.py", "checkout", "--version", target_vid])
    cli.main()

