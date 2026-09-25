"""Invert explicit image preprocessing; MMPose's topdown API already does this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PixelTransform:
    """Source -> crop -> clockwise rotation -> scaled padded model image.

    Coordinates use pixel-center conventions. The production topdown API accepts
    the original image and returns source coordinates, so it uses identity here.
    This transform is for adapters that supply externally preprocessed arrays.
    """

    crop_x: float = 0
    crop_y: float = 0
    crop_width: float = 1
    crop_height: float = 1
    clockwise: Literal[0, 90, 180, 270] = 0
    scale_x: float = 1
    scale_y: float = 1
    pad_x: float = 0
    pad_y: float = 0
    mirror_x: bool = False

    def __post_init__(self) -> None:
        if min(self.crop_width, self.crop_height, self.scale_x, self.scale_y) <= 0:
            raise ValueError("crop extent and resize scales must be positive")

    def to_model(self, xy: tuple[float, float]) -> tuple[float, float]:
        x, y = xy[0] - self.crop_x, xy[1] - self.crop_y
        if self.clockwise == 90:
            x, y = self.crop_height - 1 - y, x
        elif self.clockwise == 180:
            x, y = self.crop_width - 1 - x, self.crop_height - 1 - y
        elif self.clockwise == 270:
            x, y = y, self.crop_width - 1 - x
        rotated_width = (
            self.crop_height if self.clockwise in (90, 270) else self.crop_width
        )
        if self.mirror_x:
            x = rotated_width - 1 - x
        return x * self.scale_x + self.pad_x, y * self.scale_y + self.pad_y

    def to_source(self, xy: tuple[float, float]) -> tuple[float, float]:
        x = (xy[0] - self.pad_x) / self.scale_x
        y = (xy[1] - self.pad_y) / self.scale_y
        rotated_width = (
            self.crop_height if self.clockwise in (90, 270) else self.crop_width
        )
        if self.mirror_x:
            x = rotated_width - 1 - x
        if self.clockwise == 90:
            x, y = y, self.crop_height - 1 - x
        elif self.clockwise == 180:
            x, y = self.crop_width - 1 - x, self.crop_height - 1 - y
        elif self.clockwise == 270:
            x, y = self.crop_width - 1 - y, x
        return x + self.crop_x, y + self.crop_y
