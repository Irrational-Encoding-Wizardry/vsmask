from __future__ import annotations

from enum import Enum, auto
from functools import partial
from itertools import zip_longest
from typing import Any, Callable, Optional, Protocol, Sequence, Union

import vapoursynth as vs
from vsutil import EXPR_VARS

core = vs.core


def pick_px_op(
    use_expr: bool,
    expr: str,
    lut: Union[int, float, Sequence[int], Sequence[float], Callable[..., Any]],
) -> Callable[..., vs.VideoNode]:
    """
    Pick either std.Lut or std.Expr

    :param use_expr: [description]

    :param expr: [description]
    :param lut: [description]

    :return: Callable[..., vs.VideoNode]
    """
    if use_expr:
        func = partial(core.std.Expr, expr=expr)
    else:
        if callable(lut):
            func = partial(core.std.Lut, function=lut)
        elif isinstance(lut, Sequence):
            if all(isinstance(x, int) for x in lut):
                func = partial(core.std.Lut, lut=lut)
            elif all(isinstance(x, float) for x in lut):
                func = partial(core.std.Lut, lutf=lut)
            else:
                raise ValueError('pick_px_operation: operations[1] is not a valid type!')
        elif isinstance(lut, int):
            func = partial(core.std.Lut, lut=lut)
        elif isinstance(lut, float):
            func = partial(core.std.Lut, lutf=lut)
        else:
            raise ValueError('pick_px_operation: operations[1] is not a valid type!')
    return func


def max_expr(n: int) -> str:
    """
    Dynamic variable max string to be integrated in std.Expr.

    :param n:           Number of elements.
    :return:            Expression
    """
    names = ' '.join(EXPR_VARS[:n])
    maxes = 'max ' * (n-1)
    return f'{names} {maxes}'[:-1]


class XxpandMode(Enum):
    """Expand/inpand mode"""
    RECTANGLE = auto()
    ELLIPSE = auto()
    LOSANGE = auto()


class _MorphoFunc(Protocol):
    def __call__(self, clip: vs.VideoNode, planes: int | Sequence[int] | None = ...,
                 threshold: Optional[int] = ..., coordinates: int | Sequence[int] | None = ...) -> vs.VideoNode:
        ...


def morpho_transfo(clip: vs.VideoNode, func: _MorphoFunc, sw: int, sh: Optional[int] = None,
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


# def max_planes(*clips: vs.VideoNode, resizer)
