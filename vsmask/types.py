from __future__ import annotations

from typing import Protocol

from vstools import PlanesT, SingleOrArrOpt, vs


class MorphoFunc(Protocol):
    def __call__(
        self, clip: vs.VideoNode, planes: PlanesT = ...,
        threshold: int | None = ..., coordinates: SingleOrArrOpt[int] = ...
    ) -> vs.VideoNode:
        ...
