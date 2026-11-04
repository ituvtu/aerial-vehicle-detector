from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(script_name: str) -> ModuleType:
    """
    Динамічно імпортує скрипт із папки scripts/ (наприклад, '01_extract_frames.py'),
    оскільки імена файлів із числовим префіксом не можна імпортувати через стандартний 'import'.
    """
    script_path = PROJECT_ROOT / "scripts" / script_name
    module_name = f"scripts_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def extract_frames_mod() -> ModuleType:
    """Повертає завантажений модуль scripts/01_extract_frames.py."""
    return load_script_module("01_extract_frames.py")


def create_synthetic_mp4(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
) -> None:
    """
    Створює реальний міні-відеофайл .mp4 із закодованим номером кадру у пікселях
    для інтеграційного тестування без залежності від важких 4K відео.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}")

    for idx in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(idx * 7) % 255, dtype=np.uint8)
        writer.write(frame)

    writer.release()