from __future__ import annotations

import math
import pytest

from src.optics import (
    CameraOpticsConfig,
    assign_distance_band,
    compute_focal_length_px,
    estimate_distance_m_from_px,
    estimate_distance_m_from_yolo_norm,
    expected_box_px_at_distance,
)


class TestFocalLengthSpecification:
    """Перевірка математичної специфікації фокусної відстані Pinhole-камери (Розділ 3.2)."""

    def test_focal_length_4k_horizontal_and_vertical_symmetry(self) -> None:
        """
        Умова: f_px залежить від довшої сторони кадру max(W, H).
        Отже, горизонтальний 4K (3840x2160) та вертикальний 4K (2160x3840)
        повинні мати абсолютно однакову фокусну відстань: 3840 / (2 * tan(42°)) ≈ 2132.376 px.
        """
        expected_f = 3840.0 / (2.0 * math.tan(math.radians(42.0)))
        f_horiz = compute_focal_length_px(3840, 2160, fov_long_deg=84.0)
        f_vert = compute_focal_length_px(2160, 3840, fov_long_deg=84.0)

        assert f_horiz == pytest.approx(expected_f, rel=1e-6)
        assert f_vert == pytest.approx(expected_f, rel=1e-6)
        assert f_horiz == pytest.approx(2132.376, abs=1e-2)

    def test_focal_length_1080p_is_exact_half_of_4k(self) -> None:
        """Для 1920x1080 (train A) фокусна відстань рівно вдвічі менша за 4K: ≈ 1066.188 px."""
        f_4k = compute_focal_length_px(3840, 2160, fov_long_deg=84.0)
        f_1080p = compute_focal_length_px(1920, 1080, fov_long_deg=84.0)

        assert f_1080p == pytest.approx(f_4k / 2.0, rel=1e-9)
        assert f_1080p == pytest.approx(1066.188, abs=1e-2)

    @pytest.mark.parametrize("w,h", [(0, 1080), (1920, 0), (-100, 100), (0, 0)])
    def test_invalid_dimensions_raise_value_error(self, w: int, h: int) -> None:
        with pytest.raises(ValueError, match="Invalid frame dimensions"):
            compute_focal_length_px(w, h)


class TestExpectedBoxAndDistanceInversion:
    """Перевірка еталонних розмірів авто на 200м і 400м та оборотності формул."""

    def test_expected_car_size_in_4k_at_200m_and_400m(self) -> None:
        """
        Специфікація Розділу 3.2:
        - На 200 м у 4K: довжина ≈ 48.0 px (47.98 px), ширина ≈ 19.2 px (19.19 px).
        - На 400 м у 4K: довжина ≈ 24.0 px (23.99 px), ширина ≈ 9.6 px (9.60 px).
        """
        cfg = CameraOpticsConfig(fov_long_deg=84.0, car_length_m=4.5, car_width_m=1.8)
        l_200, w_200 = expected_box_px_at_distance(200.0, 3840, 2160, cfg)
        l_400, w_400 = expected_box_px_at_distance(400.0, 3840, 2160, cfg)

        assert l_200 == pytest.approx(47.978, abs=0.05)
        assert w_200 == pytest.approx(19.191, abs=0.05)
        assert l_400 == pytest.approx(23.989, abs=0.05)
        assert w_400 == pytest.approx(9.596, abs=0.05)

    @pytest.mark.parametrize("target_dist", [50.0, 120.0, 200.0, 275.5, 400.0])
    @pytest.mark.parametrize("method", ["max_side", "geom_mean"])
    def test_distance_estimation_inverts_expected_box_px(
        self, target_dist: float, method: str
    ) -> None:
        """Якщо взяти ідеальний бокс авто на відстані D, оцінка відстані має повернути рівно D."""
        cfg = CameraOpticsConfig(estimation_method=method)  # type: ignore[arg-type]
        l_px, w_px = expected_box_px_at_distance(target_dist, 3840, 2160, cfg)

        estimated_d = estimate_distance_m_from_px(l_px, w_px, 3840, 2160, cfg)
        assert estimated_d == pytest.approx(target_dist, rel=1e-6)

    def test_yolo_normalized_coordinates_16_9_vs_9_16_consistency(self) -> None:
        """
        Критичний тест (16:9 vs 9:16):
        Автомобіль розміром 48x19.2 px у горизонтальному кадрі (3840x2160) має
        w_norm = 48 / 3840, h_norm = 19.2 / 2160.
        Той самий автомобіль, повернутий у вертикальному кадрі eval (2160x3840), має
        w_norm = 19.2 / 2160, h_norm = 48 / 3840.
        Функція estimate_distance_m_from_yolo_norm ПОВИННА дати однакову відстань (~200 м)!
        """
        cfg = CameraOpticsConfig(estimation_method="max_side")
        d_horiz = estimate_distance_m_from_yolo_norm(
            w_norm=48.0 / 3840.0,
            h_norm=19.2 / 2160.0,
            img_w_px=3840,
            img_h_px=2160,
            config=cfg,
        )
        d_vert = estimate_distance_m_from_yolo_norm(
            w_norm=19.2 / 2160.0,
            h_norm=48.0 / 3840.0,
            img_w_px=2160,
            img_h_px=3840,
            config=cfg,
        )
        assert d_horiz == pytest.approx(d_vert, rel=1e-9)
        assert d_horiz == pytest.approx(199.91, abs=0.2)


class TestRotatedVehicleStabilityProof:
    """
    Математичний тест-доказ: чому max_side стійкий до кута повороту дороги,
    а geom_mean ламає класифікацію діапазонів на діагональних розв'язках.
    """

    @pytest.mark.parametrize("angle_deg", [0, 15, 21.8, 30, 45, 60, 75, 90])
    def test_max_side_keeps_rotated_car_at_250m_in_far_band(self, angle_deg: float) -> None:
        """
        Автомобіль на реальній дистанції D = 250 м (діапазон '200-400m') рухається
        під кутом angle_deg. Його AABB має розміри:
            w_aabb = L*|cos(a)| + W*|sin(a)|
            h_aabb = L*|sin(a)| + W*|cos(a)|
        Метод 'max_side' повинен на ВСІХ кутах класифікувати авто в '200-400m'
        із похибкою дистанції менше ніж 8%.
        """
        true_dist_m = 250.0
        cfg = CameraOpticsConfig(estimation_method="max_side")
        l_px, w_px = expected_box_px_at_distance(true_dist_m, 3840, 2160, cfg)

        rad = math.radians(angle_deg)
        aabb_w_px = l_px * abs(math.cos(rad)) + w_px * abs(math.sin(rad))
        aabb_h_px = l_px * abs(math.sin(rad)) + w_px * abs(math.cos(rad))

        est_dist_m = estimate_distance_m_from_px(aabb_w_px, aabb_h_px, 3840, 2160, cfg)
        band = assign_distance_band(est_dist_m, cfg)

        assert band == "200-400m", f"Failed at {angle_deg}°: estimated {est_dist_m:.1f}m"
        assert 232.0 <= est_dist_m <= 253.0

    def test_geom_mean_fails_on_45_degree_diagonal_road(self) -> None:
        """
        Демонстрація компромісу: на тій самій машині (D = 250 м) при русі під 45°
        метод 'geom_mean' занижує дистанцію до ~159.7 м і помилково кидає її в '0-200m'.
        """
        true_dist_m = 250.0
        cfg_geom = CameraOpticsConfig(estimation_method="geom_mean")
        l_px, w_px = expected_box_px_at_distance(true_dist_m, 3840, 2160, cfg_geom)

        rad = math.radians(45.0)
        aabb_w_px = l_px * math.cos(rad) + w_px * math.sin(rad)
        aabb_h_px = l_px * math.sin(rad) + w_px * math.cos(rad)

        est_dist_geom = estimate_distance_m_from_px(aabb_w_px, aabb_h_px, 3840, 2160, cfg_geom)
        assert est_dist_geom == pytest.approx(159.7, abs=0.5)
        assert assign_distance_band(est_dist_geom, cfg_geom) == "0-200m"


class TestDistanceBandBoundariesAndEdgeCases:
    """Перевірка граничних умов розбиття на діапазони (0-200m, 200-400m, >400m)."""

    def test_exact_band_boundaries(self) -> None:
        cfg = CameraOpticsConfig()
        assert assign_distance_band(0.0, cfg) == "0-200m"
        assert assign_distance_band(199.999, cfg) == "0-200m"
        assert assign_distance_band(200.0, cfg) == "0-200m"
        assert assign_distance_band(200.001, cfg) == "200-400m"
        assert assign_distance_band(400.0, cfg) == "200-400m"
        assert assign_distance_band(400.001, cfg) == ">400m"

    @pytest.mark.parametrize("box_w,box_h", [(0.0, 10.0), (10.0, 0.0), (-5.0, 10.0)])
    def test_zero_or_negative_box_returns_infinity(self, box_w: float, box_h: float) -> None:
        dist = estimate_distance_m_from_px(box_w, box_h, 3840, 2160)
        assert math.isinf(dist)
        assert assign_distance_band(dist) == ">400m"