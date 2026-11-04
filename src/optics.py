from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

EstimationMethod = Literal["max_side", "geom_mean"]
DistanceBand = Literal["0-200m", "200-400m", ">400m"]


@dataclass(frozen=True)
class CameraOpticsConfig:
    """Фізичні константи Pinhole-моделі камери та еталонного авто."""

    fov_long_deg: float = 84.0
    car_length_m: float = 4.5
    car_width_m: float = 1.8
    estimation_method: EstimationMethod = "max_side"
    near_band_max_m: float = 200.0
    far_band_max_m: float = 400.0

    @property
    def tan_half_fov(self) -> float:
        return math.tan(math.radians(self.fov_long_deg / 2.0))

    @property
    def car_geom_mean_m(self) -> float:
        return math.sqrt(self.car_length_m * self.car_width_m)


def compute_focal_length_px(
    width_px: int,
    height_px: int,
    fov_long_deg: float = 84.0,
) -> float:
    """
    Обчислює фокусну відстань у пікселях вздовж довшої сторони кадру:
        f_px = max(W, H) / (2 * tan(FOV_long / 2))

    Для 4K (3840x2160 або 2160x3840) при FOV=84°: f_px ≈ 2132.38 px.
    Для 1080p (1920x1080) при FOV=84°: f_px ≈ 1066.19 px.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Invalid frame dimensions: {width_px}x{height_px}")
    long_side_px = float(max(width_px, height_px))
    tan_half = math.tan(math.radians(fov_long_deg / 2.0))
    return long_side_px / (2.0 * tan_half)


def estimate_distance_m_from_px(
    box_w_px: float,
    box_h_px: float,
    img_w_px: int,
    img_h_px: int,
    config: CameraOpticsConfig = CameraOpticsConfig(),
) -> float:
    """
    Оцінює фізичну відстань D (у метрах) від камери до автомобіля за розміром
    bounding box у пікселях оригінального кадру.

    Метод 'max_side' (за замовчуванням):
        D = (f_px * L_car) / max(w_px, h_px)
        Інваріантний до орієнтації кадру (16:9 vs 9:16) і стійкий до діагонального
        руху авто (максимальна кутова похибка AABB < 7.7%).

    Метод 'geom_mean':
        D = (f_px * sqrt(L_car * W_car)) / sqrt(w_px * h_px)
    """
    if box_w_px <= 0.0 or box_h_px <= 0.0:
        return float("inf")

    f_px = compute_focal_length_px(img_w_px, img_h_px, config.fov_long_deg)

    if config.estimation_method == "max_side":
        ref_size_m = config.car_length_m
        measured_px = max(box_w_px, box_h_px)
    elif config.estimation_method == "geom_mean":
        ref_size_m = config.car_geom_mean_m
        measured_px = math.sqrt(box_w_px * box_h_px)
    else:
        raise ValueError(f"Unknown estimation_method: {config.estimation_method}")

    return (f_px * ref_size_m) / measured_px


def estimate_distance_m_from_yolo_norm(
    w_norm: float,
    h_norm: float,
    img_w_px: int,
    img_h_px: int,
    config: CameraOpticsConfig = CameraOpticsConfig(),
) -> float:
    """
    Оцінює відстань (у метрах) безпосередньо з нормалізованих YOLO-координат (w_norm, h_norm in [0, 1]).
    Важливо: w_norm множиться на img_w_px, а h_norm — на img_h_px, що коректно обробляє
    як горизонтальні (3840x2160), так і вертикальні (2160x3840) кадри.
    """
    box_w_px = w_norm * float(img_w_px)
    box_h_px = h_norm * float(img_h_px)
    return estimate_distance_m_from_px(
        box_w_px=box_w_px,
        box_h_px=box_h_px,
        img_w_px=img_w_px,
        img_h_px=img_h_px,
        config=config,
    )


def assign_distance_band(
    distance_m: float,
    config: CameraOpticsConfig = CameraOpticsConfig(),
) -> DistanceBand:
    """Класифікує відстань у цільові діапазони завдання."""
    if distance_m <= config.near_band_max_m:
        return "0-200m"
    if distance_m <= config.far_band_max_m:
        return "200-400m"
    return ">400m"


def expected_box_px_at_distance(
    distance_m: float,
    img_w_px: int,
    img_h_px: int,
    config: CameraOpticsConfig = CameraOpticsConfig(),
) -> tuple[float, float]:
    """
    Повертає очікувані (довжина_px, ширина_px) автомобіля в оригінальних пікселях кадру
    на заданій дистанції distance_m (для верифікації та візуалізації).
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    f_px = compute_focal_length_px(img_w_px, img_h_px, config.fov_long_deg)
    length_px = (f_px * config.car_length_m) / distance_m
    width_px = (f_px * config.car_width_m) / distance_m
    return length_px, width_px