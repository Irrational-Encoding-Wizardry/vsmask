from typing import List, cast

import vapoursynth as vs
from vsutil import disallow_variable_format

from .types import ensure_format

core = vs.core


def join(planes: List[vs.VideoNode], /, family: vs.ColorFamily = vs.YUV) -> vs.VideoNode:
    return planes[0] if len(planes) == 1 and family == vs.GRAY \
        else core.std.ShufflePlanes(planes[:3], [0, 0, 0], family)


@disallow_variable_format
def split(clip: vs.VideoNode, /) -> List[vs.VideoNode]:
    return [clip] if ensure_format(clip).format.num_planes == 1 else cast(List[vs.VideoNode], clip.std.SplitPlanes())
