from typing import List, cast

import vapoursynth as vs

core = vs.core


def join(planes: List[vs.VideoNode], /, family: vs.ColorFamily = vs.YUV) -> vs.VideoNode:
    return planes[0] if len(planes) == 1 and family == vs.GRAY \
        else core.std.ShufflePlanes(clips=planes[:3], planes=[0, 0, 0], colorfamily=family)


def split(clip: vs.VideoNode, /) -> List[vs.VideoNode]:
    if clip.format is None:
        raise ValueError('split: Variable format not allowed!')
    return [clip] if clip.format.num_planes == 1 else cast(List[vs.VideoNode], clip.std.SplitPlanes())
