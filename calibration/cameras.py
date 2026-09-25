"""OpenCV pinhole camera convention: world XY ground, +Z up; pixel +Y down."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.models import Intrinsics


def _matrix(value: object, shape: tuple[int, ...]) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"expected finite matrix with shape {shape}")
    return matrix


def validate_intrinsics(
    intrinsics: Intrinsics, image_size: tuple[int, int],
) -> None:
    """Reject implausible or non-invertible radial profiles over this image."""
    width, height = image_size
    limit = max(image_size)
    if (
        not 0.2 * limit < intrinsics.fx < 8 * limit
        or not 0.2 * limit < intrinsics.fy < 8 * limit
        or not 0 <= intrinsics.cx < width
        or not 0 <= intrinsics.cy < height
    ):
        raise ValueError("intrinsic focal lengths or principal point are implausible")
    coeffs = intrinsics.distortion
    if len(coeffs) not in (0, 4, 5, 8):
        raise ValueError("unsupported OpenCV distortion vector")
    padded = list(coeffs) + [0.0] * (8 - len(coeffs))
    k1, k2, _, _, k3, k4, k5, k6 = padded
    radius = max(
        np.hypot((x - intrinsics.cx) / intrinsics.fx,
                 (y - intrinsics.cy) / intrinsics.fy)
        for x in (0, width)
        for y in (0, height)
    )
    samples = np.linspace(0, radius, 128)
    r2 = samples * samples
    numerator = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    denominator = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("radial distortion has nonpositive denominator")
    distorted = samples * numerator / denominator
    if not np.isfinite(distorted).all() or np.any(np.diff(distorted) <= 0):
        raise ValueError("radial distortion is nonmonotonic over the image")


@dataclass(frozen=True)
class IntrinsicProfile:
    """Intrinsics apply only to this exact decoded image geometry and camera."""

    camera_id: str
    width_px: int
    height_px: int
    crop_xywh: tuple[int, int, int, int]
    rotation_cw: int
    intrinsics: Intrinsics

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop_xywh", tuple(self.crop_xywh))
        if len(self.crop_xywh) != 4:
            raise ValueError("profile crop must be x,y,width,height")
        x, y, width, height = self.crop_xywh
        if not self.camera_id or min(self.width_px, self.height_px, width, height) <= 0:
            raise ValueError("profile requires camera and positive image/crop sizes")
        if x < 0 or y < 0 or x + width > self.width_px or y + height > self.height_px:
            raise ValueError("profile crop exceeds source image")
        if self.rotation_cw not in (0, 90, 180, 270):
            raise ValueError("rotation_cw must be 0, 90, 180 or 270")
        decoded_width, decoded_height = (
            (height, width) if self.rotation_cw in (90, 270) else (width, height)
        )
        validate_intrinsics(self.intrinsics, (decoded_width, decoded_height))

    def require_compatible(
        self, camera_id: str, image_size: tuple[int, int],
        crop_xywh: tuple[int, int, int, int], rotation_cw: int,
    ) -> None:
        if (
            camera_id != self.camera_id
            or image_size != (self.width_px, self.height_px)
            or crop_xywh != self.crop_xywh
            or rotation_cw != self.rotation_cw
        ):
            raise ValueError(
                "incompatible intrinsic profile: camera, size, crop or rotation"
            )


@dataclass(frozen=True)
class CameraModel:
    """Row-major world-to-camera transform; camera +Z is optical forward."""

    intrinsics: Intrinsics
    world_to_camera: NDArray[np.float64]

    def __post_init__(self) -> None:
        transform = _matrix(self.world_to_camera, (4, 4))
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("world_to_camera must be homogeneous")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("world_to_camera rotation must be orthonormal")
        if abs(np.linalg.det(rotation) - 1) > 1e-6:
            raise ValueError("world_to_camera rotation must be right-handed")
        owned = transform.copy()
        owned.setflags(write=False)
        object.__setattr__(self, "world_to_camera", owned)

    @property
    def intrinsic_matrix(self) -> NDArray[np.float64]:
        i = self.intrinsics
        return np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1]], dtype=np.float64)

    @property
    def distortion(self) -> NDArray[np.float64]:
        return np.asarray(self.intrinsics.distortion, dtype=np.float64)

    def project(self, points_world: object) -> NDArray[np.float64]:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_world must be finite Nx3")
        camera = points @ self.world_to_camera[:3, :3].T + self.world_to_camera[:3, 3]
        if np.any(camera[:, 2] <= 0):
            raise ValueError("point behind camera or on optical plane")
        rotation_vector, _ = cv2.Rodrigues(self.world_to_camera[:3, :3])
        pixels, _ = cv2.projectPoints(
            points, rotation_vector, self.world_to_camera[:3, 3],
            self.intrinsic_matrix, self.distortion,
        )
        return np.asarray(pixels, dtype=np.float64).reshape(-1, 2)

    def undistort(self, pixels: object) -> NDArray[np.float64]:
        """Return ideal pixel coordinates in the same intrinsic matrix."""
        coordinates = np.asarray(pixels, dtype=np.float64)
        if (
            coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("pixels must be finite Nx2")
        result = cv2.undistortPoints(
            coordinates.reshape(-1, 1, 2), self.intrinsic_matrix,
            self.distortion, P=self.intrinsic_matrix,
        )
        return np.asarray(result, dtype=np.float64).reshape(-1, 2)

    def rays(self, pixels: object) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ideal = self.undistort(pixels)
        x = (ideal[:, 0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (ideal[:, 1] - self.intrinsics.cy) / self.intrinsics.fy
        camera_rays = np.column_stack((x, y, np.ones_like(x)))
        world_rays = camera_rays @ self.world_to_camera[:3, :3]
        world_rays /= np.linalg.norm(world_rays, axis=1, keepdims=True)
        center = -self.world_to_camera[:3, :3].T @ self.world_to_camera[:3, 3]
        return np.broadcast_to(center, world_rays.shape).copy(), world_rays


def back_project_ray(
    camera: CameraModel, pixel: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    centers, directions = camera.rays([pixel])
    return centers[0], directions[0]
