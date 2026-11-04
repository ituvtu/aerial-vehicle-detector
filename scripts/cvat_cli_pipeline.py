#!/usr/bin/env python3
"""
CLI Pipeline for CVAT Task Synchronization, Raw Artifact Bundling,
and Append-Only Merkle Registry Management.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cvat_bridge import (
    export_yolo_1_1_for_cvat,
    import_yolo_1_1_from_cvat,
    load_ledger,
    materialize_version,
    pack_signed_raw_bundle,
    register_snapshot,
    unpack_signed_raw_bundle,
)
from src.manifest_manager import ensure_images_available, get_active_frames


def _get_cvat_cli_cmd() -> list[str]:
    """Resolves cvat-cli executable cross-platform (Windows & Linux)."""
    import shutil

    exe = shutil.which("cvat-cli")
    if exe:
        return [exe]
    return [sys.executable, "-m", "cvat_cli"]


def _load_env_auth() -> str | None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("CVAT_AUTH="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.getenv("CVAT_AUTH")


def cmd_pack_raw_artifact(args: argparse.Namespace) -> None:
    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    labels_raw_dir = PROJECT_ROOT / "data" / "labels_raw"
    lock_path = PROJECT_ROOT / "data" / "labels_raw_lock.json"
    artifacts_dir = PROJECT_ROOT / "artifacts"

    bundle_path = pack_signed_raw_bundle(
        labels_raw_dir=labels_raw_dir,
        manifest_path=manifest_path,
        lock_path=lock_path,
        artifacts_dir=artifacts_dir,
    )
    print(f"[Artifact] Packed raw bundle: {bundle_path.resolve()}")


def cmd_import_to_cvat(args: argparse.Namespace) -> None:
    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    registry_dir = PROJECT_ROOT / "data" / "registry"
    labels_raw_dir = PROJECT_ROOT / "data" / "labels_raw"
    labels_curated_dir = PROJECT_ROOT / "data" / "labels"
    lock_path = PROJECT_ROOT / "data" / "labels_raw_lock.json"
    audit_report_path = PROJECT_ROOT / "data" / "audit_report.json"
    cvat_dir = PROJECT_ROOT / "data" / "cvat"
    exports_dir = cvat_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    if args.artifact:
        artifact_path = Path(args.artifact)
        unpack_res = unpack_signed_raw_bundle(
            bundle_zip_path=artifact_path,
            manifest_path=manifest_path,
            registry_dir=registry_dir,
            labels_raw_dir=labels_raw_dir,
            labels_curated_dir=labels_curated_dir,
            lock_path=lock_path,
            audit_report_path=audit_report_path,
        )
        version_id = unpack_res["snapshot"]["version_id"]
    else:
        snap_res = register_snapshot(
            labels_source_dir=labels_raw_dir,
            manifest_path=manifest_path,
            registry_dir=registry_dir,
            kind="raw",
            source_metadata={"origin": "local_working_dir"},
        )
        version_id = snap_res["version_id"]

    cvat_export_zip = exports_dir / f"cvat_yolo11_{version_id}.zip"
    export_yolo_1_1_for_cvat(
        labels_dir=labels_raw_dir,
        manifest_path=manifest_path,
        output_zip_path=cvat_export_zip,
    )
    print(f"[CVAT Bridge] Exported YOLO 1.1 dataset archive: {cvat_export_zip}")

    if args.skip_cli:
        print("[CVAT Bridge] --skip-cli specified. Task creation bypassed.")
        return

    ensure_images_available(
        manifest_path=manifest_path,
        images_dir=PROJECT_ROOT / "data" / "images",
    )

    auth = args.auth or _load_env_auth()
    if not auth:
        user = input("CVAT Username: ")
        pw = getpass.getpass("CVAT Password: ")
        auth = f"{user}:{pw}"

    server_url = f"{args.server_host}:{args.server_port}"
    active_df = get_active_frames(manifest_path)
    sorted_rows = active_df.sort_values("frame_id").reset_index(drop=True)
    image_paths = [str((PROJECT_ROOT / row["image_path"]).resolve()) for _, row in sorted_rows.iterrows()]

    cli_base = _get_cvat_cli_cmd()
    labels_json = json.dumps([{"name": "vehicle"}])
    create_cmd = [
        *cli_base,
        "--server-host", args.server_host,
        "--server-port", str(args.server_port),
        "--auth", auth,
        "task", "create", args.task_name,
        "--labels", labels_json,
        "--sorting-method", "lexicographical",
        "--image_quality", "95",
        "local",
        *image_paths,
    ]
    print(f"[CVAT CLI] Creating task '{args.task_name}' on {server_url} ...")
    res = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
    task_id_str = res.stdout.strip().splitlines()[-1]
    task_id = int("".join(filter(str.isdigit, task_id_str)))

    cvat_state = {
        "task_id": task_id,
        "server_host": args.server_host,
        "server_port": args.server_port,
        "raw_version_id": version_id,
        "task_name": args.task_name,
    }
    state_file = cvat_dir / "cvat_state.json"
    state_file.write_text(json.dumps(cvat_state, indent=2) + "\n", encoding="utf-8")
    print(f"[CVAT CLI] Task created: ID {task_id}. State saved to {state_file}")

    import_cmd = [
        *cli_base,
        "--server-host", args.server_host,
        "--server-port", str(args.server_port),
        "--auth", auth,
        "task", "import-dataset", str(task_id),
        str(cvat_export_zip.resolve()),
        "--format", "YOLO 1.1",
    ]
    print(f"[CVAT CLI] Importing initial raw labels into task {task_id} ...")
    subprocess.run(import_cmd, check=True)
    print(f"[CVAT CLI] Setup complete! Open: {args.server_host}:{args.server_port}/tasks/{task_id}")


def cmd_pull_from_cvat(args: argparse.Namespace) -> None:
    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    registry_dir = PROJECT_ROOT / "data" / "registry"
    labels_raw_dir = PROJECT_ROOT / "data" / "labels_raw"
    labels_curated_dir = PROJECT_ROOT / "data" / "labels"
    lock_path = PROJECT_ROOT / "data" / "labels_raw_lock.json"
    audit_report_path = PROJECT_ROOT / "data" / "audit_report.json"
    cvat_dir = PROJECT_ROOT / "data" / "cvat"

    if args.from_zip:
        export_zip = Path(args.from_zip)
    else:
        task_id = args.task_id
        if task_id is None:
            state_file = cvat_dir / "cvat_state.json"
            if not state_file.exists():
                raise ValueError("No --task-id specified and data/cvat/cvat_state.json does not exist.")
            state_data = json.loads(state_file.read_text(encoding="utf-8"))
            task_id = state_data["task_id"]

        auth = args.auth or _load_env_auth()
        if not auth:
            user = input("CVAT Username: ")
            pw = getpass.getpass("CVAT Password: ")
            auth = f"{user}:{pw}"

        server_host = args.server_host
        server_port = args.server_port
        exports_dir = cvat_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        export_zip = exports_dir / f"cvat_export_task{task_id}.zip"

        cli_base = _get_cvat_cli_cmd()
        export_cmd = [
            *cli_base,
            "--server-host", server_host,
            "--server-port", str(server_port),
            "--auth", auth,
            "task", "export-dataset", str(task_id),
            str(export_zip.resolve()),
            "--format", "YOLO 1.1",
            "--with-images", "no",
        ]
        print(f"[CVAT CLI] Exporting annotations for task {task_id} ...")
        subprocess.run(export_cmd, check=True)

        if not args.no_backup_task:
            backups_dir = cvat_dir / "backups"
            backups_dir.mkdir(parents=True, exist_ok=True)
            backup_zip = backups_dir / f"cvat_backup_task{task_id}.zip"
            backup_cmd = [
                *cli_base,
                "--server-host", server_host,
                "--server-port", str(server_port),
                "--auth", auth,
                "task", "backup", str(task_id),
                str(backup_zip.resolve()),
            ]
            print(f"[CVAT CLI] Creating task backup: {backup_zip} ...")
            subprocess.run(backup_cmd, check=True)

    result = import_yolo_1_1_from_cvat(
        cvat_zip_path=export_zip,
        manifest_path=manifest_path,
        registry_dir=registry_dir,
        labels_raw_dir=labels_raw_dir,
        labels_curated_dir=labels_curated_dir,
        lock_path=lock_path,
        audit_report_path=audit_report_path,
    )
    snap = result["snapshot"]
    print(f"\n[Registry] Successfully imported curated snapshot: {snap['version_id']}")
    r2 = snap.get("rule2_summary", {})
    print(f"Rule 2 Audit (Raw vs Curated):")
    print(f"  Raw: {r2.get('raw_total')} | Curated: {r2.get('corrected_total')}")
    print(f"  Unchanged: {r2.get('unchanged')} | Fixed: {r2.get('fixed')}")
    print(f"  Removed (FP): {r2.get('removed')} | Added (FN): {r2.get('added')}")


def cmd_list_versions(args: argparse.Namespace) -> None:
    registry_dir = PROJECT_ROOT / "data" / "registry"
    ledger = load_ledger(registry_dir)
    versions = ledger.get("versions", [])
    active_raw = ledger.get("active_raw_version")
    active_curated = ledger.get("active_curated_version")

    print("\n" + "=" * 92)
    print(f"{'VERSION ID':<35} {'KIND':<14} {'BOXES':<6} {'R2 (U/F/R/A)':<18} {'ACTIVE'}")
    print("-" * 92)
    for v in versions:
        vid = v["version_id"]
        kind = v["kind"]
        boxes = v["total_boxes"]
        r2 = v.get("rule2_summary", {})
        r2_str = f"{r2.get('unchanged',0)}/{r2.get('fixed',0)}/{r2.get('removed',0)}/{r2.get('added',0)}"
        status = []
        if vid == active_raw:
            status.append("RAW*")
        if vid == active_curated:
            status.append("CURATED*")
        active_str = " ".join(status)
        print(f"{vid:<35} {kind:<14} {boxes:<6} {r2_str:<18} {active_str}")
    print("=" * 92 + "\n")


def cmd_checkout(args: argparse.Namespace) -> None:
    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    registry_dir = PROJECT_ROOT / "data" / "registry"
    labels_raw_dir = PROJECT_ROOT / "data" / "labels_raw"
    labels_curated_dir = PROJECT_ROOT / "data" / "labels"
    lock_path = PROJECT_ROOT / "data" / "labels_raw_lock.json"
    audit_report_path = PROJECT_ROOT / "data" / "audit_report.json"

    res = materialize_version(
        version_id=args.version,
        registry_dir=registry_dir,
        manifest_path=manifest_path,
        labels_raw_dir=labels_raw_dir,
        labels_curated_dir=labels_curated_dir,
        lock_path=lock_path,
        audit_report_path=audit_report_path,
    )
    print(f"[Registry] Checked out version '{args.version}' successfully.")
    print(f"Active state: {res}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CVAT Bridge & Label Snapshot Registry Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. pack-raw-artifact
    subparsers.add_parser("pack-raw-artifact", help="Pack labels_raw into a signed artifact zip")

    # 2. import-to-cvat
    p_imp = subparsers.add_parser("import-to-cvat", help="Create CVAT task and import YOLO 1.1 raw labels")
    p_imp.add_argument("--artifact", type=Path, default=None, help="Path to signed raw artifact zip")
    p_imp.add_argument("--server-host", default="http://localhost", help="CVAT server host")
    p_imp.add_argument("--server-port", type=int, default=8080, help="CVAT server port")
    p_imp.add_argument("--auth", default=None, help="CVAT username:password")
    p_imp.add_argument("--task-name", default="aerial-vehicle-detector", help="CVAT task name")
    p_imp.add_argument("--skip-cli", action="store_true", help="Bypass running cvat-cli commands")

    # 3. pull-from-cvat
    p_pull = subparsers.add_parser("pull-from-cvat", help="Export curated labels from CVAT into registry")
    p_pull.add_argument("--task-id", type=int, default=None, help="CVAT task id")
    p_pull.add_argument("--from-zip", type=Path, default=None, help="Offline fallback zip exported from CVAT")
    p_pull.add_argument("--server-host", default="http://localhost", help="CVAT server host")
    p_pull.add_argument("--server-port", type=int, default=8080, help="CVAT server port")
    p_pull.add_argument("--auth", default=None, help="CVAT username:password")
    p_pull.add_argument("--no-backup-task", action="store_true", help="Skip task backup")

    # 4. list-versions
    subparsers.add_parser("list-versions", help="List all versions in registry ledger")

    # 5. checkout
    p_chk = subparsers.add_parser("checkout", help="Materialize a specific version from registry")
    p_chk.add_argument("--version", required=True, help="Version ID (e.g. v001_raw_...)")

    args = parser.parse_args()

    if args.command == "pack-raw-artifact":
        cmd_pack_raw_artifact(args)
    elif args.command == "import-to-cvat":
        cmd_import_to_cvat(args)
    elif args.command == "pull-from-cvat":
        cmd_pull_from_cvat(args)
    elif args.command == "list-versions":
        cmd_list_versions(args)
    elif args.command == "checkout":
        cmd_checkout(args)


if __name__ == "__main__":
    main()

