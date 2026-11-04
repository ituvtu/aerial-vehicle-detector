from __future__ import annotations

from pathlib import Path

import pytest

from src.model_registry import create_pretrained_pico_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "yolo26n.pt").exists(), reason="yolo26n.pt not available locally")
def test_create_pretrained_pico_checkpoint(tmp_path):
    out = tmp_path / "pico_p2_pretrained_init.pt"
    info = create_pretrained_pico_checkpoint(
        yaml_path=ROOT / "configs" / "yolo26_pico_p2.yaml",
        backbone_source=ROOT / "yolo26n.pt",
        output_path=out,
        report_path=tmp_path / "report.json",
    )
    assert info["transferred_items"] > 0
    assert info["transferred_items"] <= info["total_items"]
    assert out.exists()

    from ultralytics import YOLO

    reloaded = YOLO(str(out))
    assert reloaded.model is not None
