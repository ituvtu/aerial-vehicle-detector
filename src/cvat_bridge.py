from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.manifest_manager import get_active_frames, load_manifest, parse_yolo_label_file
from src.optics import CameraOpticsConfig


def compute_labels_tree_hash(labels_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """
    Computes deterministic Merkle tree hash across all expected manifest frame label files.
    Validates that:
    1. Every expected frame_id has a corresponding labels_dir / f"{frame_id}.txt"
    2. No extra .txt files exist in labels_dir
    3. Every box in every file passes parse_yolo_label_file
    """
    labels_dir = Path(labels_dir)
    manifest_path = Path(manifest_path)

    active_df = get_active_frames(manifest_path)
    sorted_frame_ids = sorted(active_df["frame_id"].astype(str).tolist())
    expected_files = {f"{fid}.txt" for fid in sorted_frame_ids}

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    existing_txt_files = {p.name for p in labels_dir.glob("*.txt")}
    missing_files = sorted(expected_files - existing_txt_files)
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} label files in {labels_dir}: {missing_files[:5]}")

    extra_files = sorted(existing_txt_files - expected_files)
    if extra_files:
        raise ValueError(f"Found unexpected .txt files in {labels_dir}: {extra_files[:5]}")

    files_info: dict[str, dict[str, Any]] = {}
    total_boxes = 0
    tree_lines: list[str] = []

    for frame_id in sorted_frame_ids:
        txt_path = labels_dir / f"{frame_id}.txt"
        boxes = parse_yolo_label_file(txt_path)
        num_boxes = len(boxes)
        total_boxes += num_boxes

        raw_bytes = txt_path.read_bytes()
        file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        files_info[frame_id] = {
            "sha256": file_sha256,
            "num_boxes": num_boxes,
        }
        tree_lines.append(f"{frame_id}:{file_sha256}\n")

    tree_sha256 = hashlib.sha256("".join(tree_lines).encode("utf-8")).hexdigest()

    return {
        "schema_version": "1.0",
        "tree_sha256": tree_sha256,
        "short_hash": tree_sha256[:8],
        "num_frames": len(sorted_frame_ids),
        "total_boxes": total_boxes,
        "files": files_info,
    }


def create_labels_raw_lock(
    labels_raw_dir: Path,
    manifest_path: Path,
    lock_path: Path,
    model_name: str = "yoloe-26x-seg.pt",
) -> dict[str, Any]:
    """Generates deterministic lockfile for raw model-generated labels."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    info = compute_labels_tree_hash(labels_raw_dir, manifest_path)
    info["model_name"] = model_name

    json_str = json.dumps(info, sort_keys=True, indent=2) + "\n"
    lock_path.write_text(json_str, encoding="utf-8", newline="\n")
    return info


def verify_labels_lock(labels_dir: Path, lock_path: Path, manifest_path: Path) -> bool:
    """Verifies that labels_dir strictly matches lock_path specification."""
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise FileNotFoundError(f"Lock file does not exist: {lock_path}")

    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    current_info = compute_labels_tree_hash(labels_dir, manifest_path)

    expected_tree = lock_data.get("tree_sha256")
    actual_tree = current_info.get("tree_sha256")

    if expected_tree != actual_tree:
        mismatched_files = []
        expected_files = lock_data.get("files", {})
        for fid, finfo in current_info.get("files", {}).items():
            if fid not in expected_files or expected_files[fid]["sha256"] != finfo["sha256"]:
                mismatched_files.append(f"{fid}.txt")
        raise ValueError(
            f"Labels lock mismatch for {labels_dir} (expected {expected_tree[:8]}, got {actual_tree[:8]}). "
            f"Mismatched files: {mismatched_files[:10]}"
        )

    return True


def load_ledger(registry_dir: Path) -> dict[str, Any]:
    """Loads data/registry/ledger.json or initializes empty schema."""
    registry_dir = Path(registry_dir)
    ledger_path = registry_dir / "ledger.json"
    if not ledger_path.exists():
        return {
            "schema_version": "1.0",
            "active_raw_version": None,
            "active_curated_version": None,
            "versions": [],
        }
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def _save_ledger(registry_dir: Path, ledger: dict[str, Any]) -> None:
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = registry_dir / "ledger.json"
    temp_path = ledger_path.with_suffix(".json.tmp")
    content = json.dumps(ledger, sort_keys=True, indent=2) + "\n"
    temp_path.write_text(content, encoding="utf-8", newline="\n")
    temp_path.replace(ledger_path)


def register_snapshot(
    labels_source_dir: Path,
    manifest_path: Path,
    registry_dir: Path,
    kind: str,
    source_metadata: dict[str, Any] | None = None,
    parent_version_id: str | None = None,
    parent_raw_version_id: str | None = None,
    timestamp_utc: str | None = None,
    allow_duplicate_hash: bool = False,
) -> dict[str, Any]:
    """Registers an immutable, append-only snapshot under registry_dir / snapshots / <version_id>."""
    labels_source_dir = Path(labels_source_dir)
    manifest_path = Path(manifest_path)
    registry_dir = Path(registry_dir)

    valid_kinds = ("raw", "curated", "autosave_raw", "autosave_curated")
    if kind not in valid_kinds:
        raise ValueError(f"Invalid snapshot kind: {kind}. Allowed: {valid_kinds}")

    ledger = load_ledger(registry_dir)
    tree_info = compute_labels_tree_hash(labels_source_dir, manifest_path)
    tree_sha256 = tree_info["tree_sha256"]

    if not allow_duplicate_hash:
        for entry in ledger.get("versions", []):
            if entry.get("tree_sha256") == tree_sha256 and entry.get("kind") == kind:
                return entry

    seq = len(ledger.get("versions", [])) + 1
    ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_id = f"v{seq:03d}_{kind}_{ts}_{tree_info['short_hash']}"

    snapshots_dir = registry_dir / "snapshots"
    snapshot_dir = snapshots_dir / version_id
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot directory already exists: {snapshot_dir}")

    dest_labels_dir = snapshot_dir / "labels"
    dest_labels_dir.mkdir(parents=True, exist_ok=False)

    active_df = get_active_frames(manifest_path)
    for fid in active_df["frame_id"].astype(str):
        src_file = labels_source_dir / f"{fid}.txt"
        shutil.copy2(src_file, dest_labels_dir / f"{fid}.txt")

    # Resolve parent_raw_version_id
    if kind in ("raw", "autosave_raw"):
        resolved_parent_raw_id = version_id
        parent_raw_labels_dir = dest_labels_dir
    else:
        resolved_parent_raw_id = parent_raw_version_id or ledger.get("active_raw_version")
        if resolved_parent_raw_id:
            parent_raw_labels_dir = snapshots_dir / resolved_parent_raw_id / "labels"
        else:
            parent_raw_labels_dir = dest_labels_dir

    # Run audit dataset
    import importlib.util
    import sys

    mod_name = "scripts_03_audit_labels"
    if mod_name in sys.modules:
        audit_mod = sys.modules[mod_name]
    else:
        root_dir = registry_dir.parents[1] if len(registry_dir.parents) >= 2 else Path.cwd()
        script_path = root_dir / "scripts" / "03_audit_labels.py"
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load audit module from {script_path}")
        audit_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = audit_mod
        spec.loader.exec_module(audit_mod)

    run_dataset_audit = audit_mod.run_dataset_audit

    optics_cfg = CameraOpticsConfig()
    audit_report, _ = run_dataset_audit(
        active_df=active_df,
        labels_raw_dir=parent_raw_labels_dir,
        labels_dir=dest_labels_dir,
        optics_cfg=optics_cfg,
    )

    audit_path = snapshot_dir / "audit_report.json"
    audit_path.write_text(json.dumps(audit_report, indent=2) + "\n", encoding="utf-8", newline="\n")

    lock_data = {
        "version_id": version_id,
        "seq": seq,
        "kind": kind,
        "created_at_utc": ts,
        "parent_version_id": parent_version_id,
        "parent_raw_version_id": resolved_parent_raw_id,
        "source_metadata": source_metadata or {},
        **tree_info,
    }
    lock_path = snapshot_dir / "lock.json"
    lock_path.write_text(json.dumps(lock_data, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")

    rule2_summary = audit_report.get("rule2_label_correction_summary", {})
    entry = {
        "version_id": version_id,
        "seq": seq,
        "kind": kind,
        "created_at_utc": ts,
        "tree_sha256": tree_sha256,
        "short_hash": tree_info["short_hash"],
        "num_frames": tree_info["num_frames"],
        "total_boxes": tree_info["total_boxes"],
        "parent_version_id": parent_version_id,
        "parent_raw_version_id": resolved_parent_raw_id,
        "rule2_summary": rule2_summary,
    }

    ledger["versions"].append(entry)

    if kind == "raw":
        ledger["active_raw_version"] = version_id
        if ledger.get("active_curated_version") is None:
            ledger["active_curated_version"] = version_id
    elif kind == "curated":
        ledger["active_curated_version"] = version_id

    _save_ledger(registry_dir, ledger)
    return entry


def ensure_working_dirs_saved(
    labels_raw_dir: Path,
    labels_curated_dir: Path,
    manifest_path: Path,
    registry_dir: Path,
) -> list[str]:
    """
    Dirty-State Auto-Freeze Guard:
    Checks if labels_raw_dir and labels_curated_dir contain un-snapshotted files.
    If so, registers them as autosave snapshots before any destructive workspace write.
    """
    labels_raw_dir = Path(labels_raw_dir)
    labels_curated_dir = Path(labels_curated_dir)
    manifest_path = Path(manifest_path)
    registry_dir = Path(registry_dir)

    ledger = load_ledger(registry_dir)
    known_hashes = {v["tree_sha256"] for v in ledger.get("versions", [])}
    autosaved: list[str] = []

    for wdir, kind in ((labels_raw_dir, "autosave_raw"), (labels_curated_dir, "autosave_curated")):
        if not wdir.exists():
            continue
        try:
            tree_info = compute_labels_tree_hash(wdir, manifest_path)
        except (FileNotFoundError, ValueError):
            continue

        if tree_info["tree_sha256"] not in known_hashes:
            snap = register_snapshot(
                labels_source_dir=wdir,
                manifest_path=manifest_path,
                registry_dir=registry_dir,
                kind=kind,
                source_metadata={"reason": "dirty_workspace_autofreeze", "source_dir": str(wdir)},
            )
            autosaved.append(snap["version_id"])
            known_hashes.add(tree_info["tree_sha256"])

    return autosaved


def materialize_version(
    version_id: str,
    registry_dir: Path,
    manifest_path: Path,
    labels_raw_dir: Path,
    labels_curated_dir: Path,
    lock_path: Path,
    audit_report_path: Path,
) -> dict[str, Any]:
    """Safely materializes active versions into workspace directories."""
    registry_dir = Path(registry_dir)
    manifest_path = Path(manifest_path)
    labels_raw_dir = Path(labels_raw_dir)
    labels_curated_dir = Path(labels_curated_dir)
    lock_path = Path(lock_path)
    audit_report_path = Path(audit_report_path)

    ensure_working_dirs_saved(labels_raw_dir, labels_curated_dir, manifest_path, registry_dir)

    ledger = load_ledger(registry_dir)
    versions_by_id = {v["version_id"]: v for v in ledger.get("versions", [])}
    if version_id not in versions_by_id:
        raise KeyError(f"Version '{version_id}' not found in registry ledger.")

    target_entry = versions_by_id[version_id]
    snapshots_dir = registry_dir / "snapshots"
    target_snap_dir = snapshots_dir / version_id
    verify_labels_lock(target_snap_dir / "labels", target_snap_dir / "lock.json", manifest_path)

    kind = target_entry["kind"]
    if kind in ("raw", "autosave_raw"):
        ledger["active_raw_version"] = version_id
    elif kind in ("curated", "autosave_curated"):
        ledger["active_curated_version"] = version_id

    active_raw_id = ledger.get("active_raw_version")
    active_curated_id = ledger.get("active_curated_version") or active_raw_id

    if active_raw_id:
        raw_snap = snapshots_dir / active_raw_id
        labels_raw_dir.mkdir(parents=True, exist_ok=True)
        for p in (raw_snap / "labels").glob("*.txt"):
            shutil.copy2(p, labels_raw_dir / p.name)
        shutil.copy2(raw_snap / "lock.json", lock_path)

    if active_curated_id:
        curated_snap = snapshots_dir / active_curated_id
        labels_curated_dir.mkdir(parents=True, exist_ok=True)
        for p in (curated_snap / "labels").glob("*.txt"):
            shutil.copy2(p, labels_curated_dir / p.name)
        if (curated_snap / "audit_report.json").exists():
            shutil.copy2(curated_snap / "audit_report.json", audit_report_path)

    _save_ledger(registry_dir, ledger)
    return {
        "active_raw_version": ledger["active_raw_version"],
        "active_curated_version": ledger["active_curated_version"],
        "materialized_version_id": version_id,
    }


def pack_signed_raw_bundle(
    labels_raw_dir: Path,
    manifest_path: Path,
    lock_path: Path,
    artifacts_dir: Path,
    model_name: str = "yoloe-26x-seg.pt",
    timestamp_utc: str | None = None,
) -> Path:
    """Packs raw labels and lockfile into non-overwriting artifacts archive."""
    labels_raw_dir = Path(labels_raw_dir)
    manifest_path = Path(manifest_path)
    lock_path = Path(lock_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    info = create_labels_raw_lock(labels_raw_dir, manifest_path, lock_path, model_name)
    ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_hash = info["short_hash"]
    bundle_path = artifacts_dir / f"raw_autolabel_{ts}_{short_hash}.zip"

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(lock_path, arcname="labels_raw_lock.json")
        active_df = get_active_frames(manifest_path)
        for fid in active_df["frame_id"].astype(str):
            txt_file = labels_raw_dir / f"{fid}.txt"
            zf.write(txt_file, arcname=f"labels_raw/{fid}.txt")

    with zipfile.ZipFile(bundle_path, "r") as zf:
        if zf.testzip() is not None:
            raise zipfile.BadZipFile(f"Corrupted zip created: {bundle_path}")

    return bundle_path


def unpack_signed_raw_bundle(
    bundle_zip_path: Path,
    manifest_path: Path,
    registry_dir: Path,
    labels_raw_dir: Path,
    labels_curated_dir: Path,
    lock_path: Path,
    audit_report_path: Path,
) -> dict[str, Any]:
    """Verifies and unzips raw bundle into an immutable snapshot, then materializes it."""
    bundle_zip_path = Path(bundle_zip_path)
    if not zipfile.is_zipfile(bundle_zip_path):
        raise zipfile.BadZipFile(f"Not a valid zip: {bundle_zip_path}")

    with zipfile.ZipFile(bundle_zip_path, "r") as zf:
        if zf.testzip() is not None:
            raise zipfile.BadZipFile(f"Zip corruption detected in {bundle_zip_path}")
        for name in zf.namelist():
            if ".." in name or Path(name).is_absolute():
                raise ValueError(f"Zip-slip detected in {bundle_zip_path}: {name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            zf.extractall(tmp_root)
            tmp_labels_raw = tmp_root / "labels_raw"
            tmp_lock = tmp_root / "labels_raw_lock.json"

            verify_labels_lock(tmp_labels_raw, tmp_lock, manifest_path)
            ensure_working_dirs_saved(labels_raw_dir, labels_curated_dir, manifest_path, registry_dir)

            lock_data = json.loads(tmp_lock.read_text(encoding="utf-8"))
            snap_entry = register_snapshot(
                labels_source_dir=tmp_labels_raw,
                manifest_path=manifest_path,
                registry_dir=registry_dir,
                kind="raw",
                source_metadata={"origin_bundle": bundle_zip_path.name, "model_name": lock_data.get("model_name")},
            )

            mat_result = materialize_version(
                version_id=snap_entry["version_id"],
                registry_dir=registry_dir,
                manifest_path=manifest_path,
                labels_raw_dir=labels_raw_dir,
                labels_curated_dir=labels_curated_dir,
                lock_path=lock_path,
                audit_report_path=audit_report_path,
            )
            return {"snapshot": snap_entry, "materialization": mat_result}


def export_yolo_1_1_for_cvat(labels_dir: Path, manifest_path: Path, output_zip_path: Path) -> Path:
    """Exports dataset labels in strict CVAT YOLO 1.1 format with LF line endings."""
    labels_dir = Path(labels_dir)
    manifest_path = Path(manifest_path)
    output_zip_path = Path(output_zip_path)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    active_df = get_active_frames(manifest_path)
    sorted_rows = active_df.sort_values("frame_id").reset_index(drop=True)

    obj_names_content = "vehicle\n"
    obj_data_content = "classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n"

    train_lines = [f"data/obj_train_data/{row['frame_id']}.jpg\n" for _, row in sorted_rows.iterrows()]
    train_content = "".join(train_lines)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", obj_names_content.encode("utf-8"))
        zf.writestr("obj.data", obj_data_content.encode("utf-8"))
        zf.writestr("train.txt", train_content.encode("utf-8"))

        for _, row in sorted_rows.iterrows():
            fid = str(row["frame_id"])
            src_lbl = labels_dir / f"{fid}.txt"
            if src_lbl.exists():
                boxes = parse_yolo_label_file(src_lbl)
                lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in boxes]
                lbl_content = "".join(lines)
            else:
                lbl_content = ""
            zf.writestr(f"obj_train_data/{fid}.txt", lbl_content.encode("utf-8"))

    return output_zip_path


def import_yolo_1_1_from_cvat(
    cvat_zip_path: Path,
    manifest_path: Path,
    registry_dir: Path,
    labels_raw_dir: Path,
    labels_curated_dir: Path,
    lock_path: Path,
    audit_report_path: Path,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Imports CVAT YOLO 1.1 export zip into an immutable curated snapshot and materializes it."""
    cvat_zip_path = Path(cvat_zip_path)
    manifest_path = Path(manifest_path)
    if not zipfile.is_zipfile(cvat_zip_path):
        raise zipfile.BadZipFile(f"Not a valid zip: {cvat_zip_path}")

    active_df = get_active_frames(manifest_path)
    expected_fids = set(active_df["frame_id"].astype(str).tolist())

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        with zipfile.ZipFile(cvat_zip_path, "r") as zf:
            if zf.testzip() is not None:
                raise zipfile.BadZipFile(f"Zip corrupted: {cvat_zip_path}")
            for member in zf.namelist():
                if ".." in member or Path(member).is_absolute():
                    raise ValueError(f"Zip-slip detected: {member}")
            zf.extractall(tmp_root)

        staging_dir = tmp_root / "staging_labels"
        staging_dir.mkdir(parents=True, exist_ok=True)

        found_txts = {p.stem: p for p in tmp_root.rglob("*.txt") if p.name not in ("train.txt", "obj.names", "obj.data")}

        for fid in expected_fids:
            dest_file = staging_dir / f"{fid}.txt"
            if fid in found_txts:
                boxes = parse_yolo_label_file(found_txts[fid])
                lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in boxes]
                dest_file.write_text("".join(lines), encoding="utf-8", newline="\n")
            else:
                dest_file.write_text("", encoding="utf-8", newline="\n")

        ensure_working_dirs_saved(labels_raw_dir, labels_curated_dir, manifest_path, registry_dir)

        snap = register_snapshot(
            labels_source_dir=staging_dir,
            manifest_path=manifest_path,
            registry_dir=registry_dir,
            kind="curated",
            source_metadata={"origin_zip": cvat_zip_path.name, **(source_metadata or {})},
        )

        mat_res = materialize_version(
            version_id=snap["version_id"],
            registry_dir=registry_dir,
            manifest_path=manifest_path,
            labels_raw_dir=labels_raw_dir,
            labels_curated_dir=labels_curated_dir,
            lock_path=lock_path,
            audit_report_path=audit_report_path,
        )

        return {"snapshot": snap, "materialization": mat_res}
