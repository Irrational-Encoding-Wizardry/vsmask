from typing import List, cast

import vapoursynth as vs

core = vs.core


def join(planes: List[vs.VideoNode], /, family: vs.ColorFamily = vs.YUV) -> vs.VideoNode:
    if len(planes) == 1 and family == vs.GRAY:
        return planes[0]
    return core.std.ShufflePlanes(clips=planes[:3], planes=[0, 0, 0], colorfamily=family)


def split(clip: vs.VideoNode, /) -> List[vs.VideoNode]:
    assert clip.format
    if clip.format.num_planes == 1:
        return [clip]
    return list(cast(List[vs.VideoNode], clip.std.SplitPlanes()))
