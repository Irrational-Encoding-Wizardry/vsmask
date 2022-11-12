from __future__ import annotations

from typing import Protocol

from vstools import PlanesT, StrArrOpt, VideoNode


class MorphoFunc(Protocol):
    def __call__(
        self, clip: VideoNode, planes: PlanesT = ..., threshold: int | None = ..., coordinates: StrArrOpt[int] = ...
    ) -> VideoNode:
        ...
