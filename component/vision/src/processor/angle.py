from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_FOV_DEG = 73.0
DEFAULT_DISTORTION = {
    "k1": -0.08,
    "k2": 0.02,
    "p1": 0.0,
    "p2": 0.0,
    "k3": 0.0,
}


def clamp_fov_deg(value: float | None) -> float:
    if value is None:
        return DEFAULT_FOV_DEG
    return min(179.0, max(1.0, float(value)))


def normalize_distortion(distortion: Mapping[str, float] | None = None) -> dict[str, float]:
    merged = dict(DEFAULT_DISTORTION)
    if distortion:
        for key in merged:
            raw = distortion.get(key)
            if raw is not None:
                merged[key] = float(raw)
    return merged


@dataclass(frozen=True)
class CameraCalibration:
    fov_deg: float = DEFAULT_FOV_DEG
    k1: float = DEFAULT_DISTORTION["k1"]
    k2: float = DEFAULT_DISTORTION["k2"]
    p1: float = DEFAULT_DISTORTION["p1"]
    p2: float = DEFAULT_DISTORTION["p2"]
    k3: float = DEFAULT_DISTORTION["k3"]

    @classmethod
    def from_values(
        cls,
        fov_deg: float | None = None,
        distortion: Mapping[str, float] | None = None,
    ) -> "CameraCalibration":
        values = normalize_distortion(distortion)
        return cls(
            fov_deg=clamp_fov_deg(fov_deg),
            k1=values["k1"],
            k2=values["k2"],
            p1=values["p1"],
            p2=values["p2"],
            k3=values["k3"],
        )

    def focal_length_px(self, image_width: int) -> float:
        return float(image_width) / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    def camera_matrix(self, image_width: int, image_height: int) -> np.ndarray:
        fx = self.focal_length_px(image_width)
        fy = fx
        cx = image_width / 2.0
        cy = image_height / 2.0
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def distortion_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)

    def undistort_pixel(
        self,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        dist_coeffs = self.distortion_coeffs()
        if cv2 is None or not np.any(np.abs(dist_coeffs) > 1e-12):
            return float(x_px), float(y_px)
        camera_matrix = self.camera_matrix(image_width, image_height)
        points = np.array([[[float(x_px), float(y_px)]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
        u, v = undistorted[0, 0]
        return float(u), float(v)

    def normalized_ray(
        self,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        u, v = self.undistort_pixel(x_px, y_px, image_width, image_height)
        fx = self.focal_length_px(image_width)
        fy = fx
        cx = image_width / 2.0
        cy = image_height / 2.0
        xn = (u - cx) / max(fx, 1e-6)
        yn = (v - cy) / max(fy, 1e-6)
        return float(xn), float(yn)

    def bearing_angle_deg(
        self,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
    ) -> float:
        xn, _ = self.normalized_ray(x_px, y_px, image_width, image_height)
        return math.degrees(math.atan(xn))

    def range_from_z_depth_m(
        self,
        z_depth_m: float | None,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
    ) -> float | None:
        if z_depth_m is None:
            return None
        xn, yn = self.normalized_ray(x_px, y_px, image_width, image_height)
        return float(z_depth_m * math.sqrt(1.0 + (xn * xn) + (yn * yn)))


DEFAULT_CAMERA_CALIBRATION = CameraCalibration.from_values()


def detection_center(detection: Mapping[str, float]) -> tuple[float, float]:
    x = float(detection.get("x", 0.0))
    y = float(detection.get("y", 0.0))
    w = max(0.0, float(detection.get("w", 0.0)))
    h = max(0.0, float(detection.get("h", 0.0)))
    return x + (w / 2.0), y + (h / 2.0)


def bearing_angle_for_detection(
    calibration: CameraCalibration,
    detection: Mapping[str, float],
    image_width: int,
    image_height: int,
) -> float:
    center_x, center_y = detection_center(detection)
    return calibration.bearing_angle_deg(center_x, center_y, image_width, image_height)
