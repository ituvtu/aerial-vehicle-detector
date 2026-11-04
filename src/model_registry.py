from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    """
    Append-only model registry with Champion Guard protection against overwriting best.pt.
    Stores immutable model runs in weights/registry/m{NNN}_{exp_name}_{UTC_TS}_{SHA8}/.
    """

    def __init__(self, weights_dir: Path | str = "weights"):
        self.weights_dir = Path(weights_dir)
        self.registry_dir = self.weights_dir / "registry"
        self.ledger_path = self.registry_dir / "model_ledger.json"
        self._ensure_init()

    def _ensure_init(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists():
            initial_ledger = {
                "schema_version": "1.0",
                "active_champion_id": None,
                "models": [],
            }
            self._save_ledger(initial_ledger)

    def load_ledger(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            self._ensure_init()
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def _save_ledger(self, data: dict[str, Any]) -> None:
        temp_file = self.ledger_path.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp_file.replace(self.ledger_path)

    def auto_freeze_unregistered_root_weights(self) -> str | None:
        """
        If weights/best.pt exists and its sha256 is not registered in model_ledger.json,
        archives it as m000_legacy_{ts}_{sha8} so prior work is never lost.
        """
        root_best = self.weights_dir / "best.pt"
        if not root_best.exists():
            return None

        ledger = self.load_ledger()
        root_sha = compute_file_sha256(root_best)
        registered_shas = {m["weights_sha256"] for m in ledger.get("models", [])}

        if root_sha in registered_shas:
            return None

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_sha = root_sha[:8]
        model_id = "m000"
        folder_name = f"{model_id}_legacy_{ts}_{short_sha}"
        target_dir = self.registry_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy best.pt
        shutil.copy2(root_best, target_dir / "best.pt")

        # Copy calibration if exists
        root_calib = self.weights_dir / "calibration_val.json"
        calib_data = {}
        if root_calib.exists():
            shutil.copy2(root_calib, target_dir / "calibration_val.json")
            try:
                calib_data = json.loads(root_calib.read_text(encoding="utf-8"))
            except Exception:
                calib_data = {}

        manifest = {
            "model_id": model_id,
            "exp_name": "legacy_preexisting",
            "created_at_utc": ts,
            "folder_name": folder_name,
            "weights_sha256": root_sha,
            "val_selection_score": 0.0,
            "val_macro_f1": 0.0,
            "val_map50": 0.0,
            "best_conf_threshold": calib_data.get("best_conf_threshold", 0.25),
            "is_champion": False,
        }
        (target_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        ledger["models"].insert(0, manifest)
        if ledger.get("active_champion_id") is None:
            ledger["active_champion_id"] = model_id
            manifest["is_champion"] = True
        self._save_ledger(ledger)
        print(f"[Model Registry] Auto-froze unregistered root weights into: {folder_name}")
        return model_id

    def get_next_model_id(self) -> str:
        ledger = self.load_ledger()
        existing_ids = [m["model_id"] for m in ledger.get("models", [])]
        numbers = []
        for mid in existing_ids:
            if mid.startswith("m") and mid[1:].isdigit():
                numbers.append(int(mid[1:]))
        next_num = max(numbers, default=0) + 1
        return f"m{next_num:03d}"

    def register_model_run(
        self,
        exp_name: str,
        best_pt_path: Path | str,
        calibration_data: dict[str, Any],
        val_macro_f1: float,
        val_map50: float,
        train_history_csv: Path | str | None = None,
        training_curves_png: Path | str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        dataset_snapshot_id: str | None = None,
        unfused_params: int = 0,
        fused_params: int = 0,
        force_champion: bool = False,
    ) -> dict[str, Any]:
        """
        Registers a newly trained model run. Applies Champion Guard:
        val_selection_score = 0.5 * val_macro_f1 + 0.5 * val_map50
        If new_score > champion_score (or force_champion=True):
            Promotes to active_champion_id and updates root weights/best.pt & weights/calibration_val.json.
        Else:
            Saves in registry without overwriting root weights/best.pt.
        """
        best_pt = Path(best_pt_path)
        if not best_pt.exists():
            raise FileNotFoundError(f"Weights file not found: {best_pt}")

        self.auto_freeze_unregistered_root_weights()

        ledger = self.load_ledger()
        model_id = self.get_next_model_id()
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        weights_sha = compute_file_sha256(best_pt)
        short_sha = weights_sha[:8]

        folder_name = f"{model_id}_{exp_name}_{ts}_{short_sha}"
        target_dir = self.registry_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy weights
        dest_best_pt = target_dir / "best.pt"
        shutil.copy2(best_pt, dest_best_pt)

        # 2. Write calibration
        dest_calib = target_dir / "calibration_val.json"
        dest_calib.write_text(json.dumps(calibration_data, indent=2) + "\n", encoding="utf-8")

        # 3. Copy optional training history and curves
        if train_history_csv and Path(train_history_csv).exists():
            shutil.copy2(train_history_csv, target_dir / "train_history.csv")
        if training_curves_png and Path(training_curves_png).exists():
            shutil.copy2(training_curves_png, target_dir / "training_curves.png")

        # 4. Compute selection score
        selection_score = round(0.5 * float(val_macro_f1) + 0.5 * float(val_map50), 4)

        # 5. Champion Guard logic
        current_champion_id = ledger.get("active_champion_id")
        current_champion_score = -1.0

        if current_champion_id:
            for m in ledger.get("models", []):
                if m["model_id"] == current_champion_id:
                    current_champion_score = float(m.get("val_selection_score", 0.0))
                    break

        is_new_champion = force_champion or (selection_score > current_champion_score)

        manifest = {
            "model_id": model_id,
            "exp_name": exp_name,
            "created_at_utc": ts,
            "folder_name": folder_name,
            "weights_sha256": weights_sha,
            "val_selection_score": selection_score,
            "val_macro_f1": round(float(val_macro_f1), 4),
            "val_map50": round(float(val_map50), 4),
            "best_conf_threshold": calibration_data.get("best_conf_threshold", 0.25),
            "unfused_params": unfused_params,
            "fused_params": fused_params,
            "dataset_snapshot_id": dataset_snapshot_id,
            "hyperparameters": hyperparameters or {},
            "is_champion": is_new_champion,
        }
        (target_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        # Update ledger
        if is_new_champion:
            for m in ledger["models"]:
                m["is_champion"] = False
            ledger["active_champion_id"] = model_id
            # Materialize to root weights/
            shutil.copy2(dest_best_pt, self.weights_dir / "best.pt")
            shutil.copy2(dest_calib, self.weights_dir / "calibration_val.json")
            print(
                f"[Champion Guard] PROMOTED {model_id} ({exp_name}) as active champion! "
                f"(score={selection_score:.4f} > prior={current_champion_score:.4f}). "
                f"Updated weights/best.pt."
            )
        else:
            print(
                f"[Champion Guard] Keeping existing champion {current_champion_id} "
                f"(score={current_champion_score:.4f} >= new {model_id} score={selection_score:.4f}). "
                f"New model safely archived as {folder_name} without overwriting weights/best.pt."
            )

        ledger["models"].append(manifest)
        self._save_ledger(ledger)
        return manifest

    def promote_model(self, model_id: str) -> dict[str, Any]:
        """Manually promotes a specific model_id to active champion."""
        ledger = self.load_ledger()
        target_model = None
        for m in ledger.get("models", []):
            if m["model_id"] == model_id:
                target_model = m
                break

        if not target_model:
            raise KeyError(f"Model ID '{model_id}' not found in registry ledger.")

        folder_dir = self.registry_dir / target_model["folder_name"]
        src_pt = folder_dir / "best.pt"
        src_calib = folder_dir / "calibration_val.json"

        if not src_pt.exists():
            raise FileNotFoundError(f"Model checkpoint missing at {src_pt}")

        for m in ledger["models"]:
            m["is_champion"] = m["model_id"] == model_id

        ledger["active_champion_id"] = model_id
        self._save_ledger(ledger)

        # Copy to root weights/
        shutil.copy2(src_pt, self.weights_dir / "best.pt")
        if src_calib.exists():
            shutil.copy2(src_calib, self.weights_dir / "calibration_val.json")

        print(f"[Champion Guard] Manually promoted {model_id} ({target_model['exp_name']}) to champion.")
        return target_model

    def resolve_model(self, model_id: str | None = None) -> tuple[Path, Path, str, str]:
        """
        Resolves (weights_path, calibration_path, model_id, sha256) for a given model_id,
        or falls back to active_champion_id / weights/best.pt.
        """
        ledger = self.load_ledger()
        target_id = model_id or ledger.get("active_champion_id")

        if target_id:
            for m in ledger.get("models", []):
                if m["model_id"] == target_id:
                    folder = self.registry_dir / m["folder_name"]
                    pt = folder / "best.pt"
                    calib = folder / "calibration_val.json"
                    if pt.exists():
                        return pt, calib, m["model_id"], m.get("weights_sha256", compute_file_sha256(pt))

        # Fallback to root weights/best.pt
        root_best = self.weights_dir / "best.pt"
        root_calib = self.weights_dir / "calibration_val.json"
        if not root_best.exists():
            raise FileNotFoundError(f"No checkpoint available at {root_best}")
        sha = compute_file_sha256(root_best)
        return root_best, root_calib, target_id or "untracked", sha

    def list_models_table(self) -> str:
        """Renders formatted table of all registered models."""
        ledger = self.load_ledger()
        models = ledger.get("models", [])
        if not models:
            return "No models registered in weights/registry/ yet."

        champ = ledger.get("active_champion_id")
        lines = [
            f"Active Champion: {champ}",
            "",
            f"{'ID':<6} {'Exp Name':<22} {'Macro-F1':<10} {'mAP@0.5':<10} {'Score':<8} {'Conf':<6} {'Params(F)':<10} {'Champion'}",
            "-" * 86,
        ]
        for m in models:
            is_c = "★ YES" if m["model_id"] == champ else "  no"
            params_str = f"{m.get('fused_params', 0):,}" if m.get('fused_params') else "N/A"
            lines.append(
                f"{m['model_id']:<6} {m['exp_name'][:21]:<22} {m.get('val_macro_f1', 0.0):<10.4f} "
                f"{m.get('val_map50', 0.0):<10.4f} {m.get('val_selection_score', 0.0):<8.4f} "
                f"{m.get('best_conf_threshold', 0.25):<6.2f} {params_str:<10} {is_c}"
            )
        return "\n".join(lines)
