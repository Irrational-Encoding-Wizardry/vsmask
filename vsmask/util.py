from __future__ import annotations

from functools import partial
from itertools import islice, zip_longest
from typing import Iterable, List, Optional, Sequence

from vskernels import Bilinear, Kernel, KernelT
from vstools import EXPR_VARS, CustomEnum, check_variable_format, core, disallow_variable_format, flatten, split, vs

from .types import MorphoFunc


def max_expr(n: int) -> str:
    """
    Dynamic variable max string to be integrated in std.Expr.

    :param n:           Number of elements.
    :return:            Expression
    """
    return ' '.join(var for var in EXPR_VARS[:n]) + ' max' * (n - 1)


class XxpandMode(CustomEnum):
    """Expand/inpand mode"""
    RECTANGLE = object()
    """Rectangular shape"""
    ELLIPSE = object()
    """Elliptical shape"""
    LOSANGE = object()
    """Diamond shape"""

    def __repr__(self) -> str:
        return '<%s.%s>' % (self.__class__.__name__, self.name)


def morpho_transfo(clip: vs.VideoNode, func: MorphoFunc, sw: int, sh: Optional[int] = None,
                   mode: XxpandMode = XxpandMode.RECTANGLE, thr: Optional[int] = None,
                   planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    """
    Calls a morphological function in order to grow or shrink a clip from the desired width and height.

    :param clip:        Source clip.
    :param func:        Morphological function.
    :param sw:          Growing/shrinking shape width.
    :param sh:          Growing/shrinking shape height. If not specified, default to sw.
    :param mode:        Shape form. Ellipses are combinations of rectangles and losanges
                        and look more like octogons.
                        Losanges are truncated (not scaled) when sw and sh are not equal.
    :param thr:         Allows to limit how much pixels are changed.
                        Output pixels will not become less than ``input - threshold``.
                        The default is no limit.
    :param planes:      Specifies which planes will be processed. Any unprocessed planes will be simply copied.
    :return:            Transformed clip
    """
    if sh is None:
        sh = sw
    for (wi, hi) in zip_longest(range(sw, -1, -1), range(sh, -1, -1), fillvalue=0):
        if wi > 0 and hi > 0:
            if mode == XxpandMode.LOSANGE or (mode == XxpandMode.ELLIPSE and wi % 3 != 1):
                coordinates = [0, 1, 0, 1, 1, 0, 1, 0]
            else:
                coordinates = [1] * 8
        elif wi > 0:
            coordinates = [0, 0, 0, 1, 1, 0, 0, 0]
        elif hi > 0:
            coordinates = [0, 1, 0, 0, 0, 0, 1, 0]
        else:
            break
        clip = func(clip, planes, thr, coordinates)
    return clip


def expand(clip: vs.VideoNode, sw: int, sh: Optional[int] = None, mode: XxpandMode = XxpandMode.RECTANGLE,
           thr: Optional[int] = None, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    """
    Calls std.Maximum in order to grow each pixel with the largest value in its 3x3 neighbourhood
    from the desired width and height.

    :param clip:        Source clip.
    :param sw:          Growing shape width.
    :param sh:          Growing shape height. If not specified, default to sw.
    :param mode:        Shape form. Ellipses are combinations of rectangles and losanges
                        and look more like octogons.
                        Losanges are truncated (not scaled) when sw and sh are not equal.
    :param thr:         Allows to limit how much pixels are changed.
                        Output pixels will not become less than ``input - threshold``.
                        The default is no limit.
    :param planes:      Specifies which planes will be processed. Any unprocessed planes will be simply copied.
    :return:            Transformed clip
    """
    return morpho_transfo(clip, core.std.Maximum, sw, sh, mode, thr, planes)


def inpand(clip: vs.VideoNode, sw: int, sh: Optional[int] = None, mode: XxpandMode = XxpandMode.RECTANGLE,
           thr: Optional[int] = None, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    """
    Calls std.Minimum in order to shrink each pixel with the smallest value in its 3x3 neighbourhood
    from the desired width and height.

    :param clip:        Source clip.
    :param sw:          Shrinking shape width.
    :param sh:          Shrinking shape height. If not specified, default to sw.
    :param mode:        Shape form. Ellipses are combinations of rectangles and losanges
                        and look more like octogons.
                        Losanges are truncated (not scaled) when sw and sh are not equal.
    :param thr:         Allows to limit how much pixels are changed.
                        Output pixels will not become less than ``input - threshold``.
                        The default is no limit.
    :param planes:      Specifies which planes will be processed. Any unprocessed planes will be simply copied.
    :return:            Transformed clip
    """
    return morpho_transfo(clip, core.std.Minimum, sw, sh, mode, thr, planes)


@disallow_variable_format
def max_planes(*clips: vs.VideoNode, resizer: KernelT = Bilinear) -> vs.VideoNode:
    """
    Set max value of all the planes of all the clips

    Output clip format is a GRAY clip with the same bitdepth as the first clip

    :param clips:       Source clips.
    :param resizer:     Resizer used for converting the clips to the same width, height and to 444.
    :return:            Maxed clip
    """
    resizer = Kernel.ensure_obj(resizer, max_planes)

    model = clips[0]
    assert check_variable_format(model, max_planes)

    width, height, format_target = model.width, model.height, model.format

    format_target = format_target.replace(subsampling_w=0, subsampling_h=0)

    planes = list[vs.VideoNode](flatten(  # type: ignore[arg-type]
        split(resizer.scale(clip, width, height, format=format_target)) for clip in clips
    ))

    def _max_clips(p: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(p, max_expr(len(p)))

    def _recursive_max(p: List[vs.VideoNode]) -> vs.VideoNode:
        if len(p) < 27:
            return _max_clips(p)

        p_iter = iter(p)
        return _recursive_max([
            _max_clips(chunked)
            for chunked in iter(lambda: tuple(islice(p_iter, 26)), ())
        ])

    return _recursive_max(planes)


def region_mask(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    """
    Alias for :py:func:`region_rel_mask`

    Region relatively the clip with the desired numbers of pixels

    :param clip:        Source clip
    :param left:        Left side
    :param right:       Right side
    :param top:         Top side
    :param bottom:      Bottom side
    :return:            Regionned mask
    """
    return region_rel_mask(clip, left, right, top, bottom)


def region_rel_mask(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    """
    Region relatively the clip with the desired numbers of pixels

    :param clip:        Source clip
    :param left:        Left side
    :param right:       Right side
    :param top:         Top side
    :param bottom:      Bottom side
    :return:            Regionned mask
    """
    return clip.std.Crop(left, right, top, bottom).std.AddBorders(left, right, top, bottom)


def region_abs_mask(clip: vs.VideoNode, width: int, height: int, left: int = 0, top: int = 0) -> vs.VideoNode:
    """
    Region the clip with absolute desired dimensions

    :param clip:        Source clip
    :param width:       Width of the box
    :param height:      Height of the box
    :param left:        Shift from the left, AKA x parameter
    :param top:         Shift from the top, AKA y parameter
    :return:            Regionned mask
    """
    def _crop(c: vs.VideoNode, w: int, h: int) -> vs.VideoNode:
        return c.std.CropAbs(width, height, left, top).std.AddBorders(
            left, w - width - left, top, h - height - top
        )

    if 0 in {clip.width, clip.height}:
        def _region(n: int, f: vs.VideoFrame, c: vs.VideoNode) -> vs.VideoNode:
            return _crop(c, f.width, f.height)
        return clip.std.FrameEval(partial(_region, c=clip), clip)
    return _crop(clip, clip.width, clip.height)
