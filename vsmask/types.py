
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Sequence

from vapoursynth import VideoFormat, VideoNode

if TYPE_CHECKING:
    class _VideoNode(VideoNode):
        format: VideoFormat

    def ensure_format(clip: VideoNode) -> _VideoNode:
        return _VideoNode()
else:
    def ensure_format(clip: VideoNode) -> VideoNode:
        return clip


class MorphoFunc(Protocol):
    def __call__(self, clip: VideoNode, planes: int | Sequence[int] | None = ...,
                 threshold: Optional[int] = ..., coordinates: int | Sequence[int] | None = ...) -> VideoNode:
        ...


class ZResizer(Protocol):
    def __call__(self, clip: VideoNode, width: Optional[int] = ..., height: Optional[int] = ...,
                 format: Optional[int] = ...) -> VideoNode:
        ...
