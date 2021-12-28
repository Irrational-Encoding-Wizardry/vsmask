
from __future__ import annotations

from typing import Optional, Protocol, Sequence

from vapoursynth import VideoNode


class MorphoFunc(Protocol):
    def __call__(self, clip: VideoNode, planes: int | Sequence[int] | None = ...,
                 threshold: Optional[int] = ..., coordinates: int | Sequence[int] | None = ...) -> VideoNode:
        ...


class ZResizer(Protocol):
    def __call__(self, clip: VideoNode, width: Optional[int] = ..., height: Optional[int] = ...,
                 format: Optional[int] = ...) -> VideoNode:
        ...
