from __future__ import annotations

from typing import Any, Iterable

from vsexprtools import ExprOp, aka_expr_available, norm_expr
from vskernels import Bilinear, Kernel, KernelT
from vsrgtools import box_blur, gauss_blur
from vstools import (
    CustomValueError, FrameRangeN, FrameRangesN, FuncExceptT, check_variable, check_variable_format, flatten,
    get_peak_value, insert_clip, replace_ranges, split, vs, depth
)

from .edge import EdgeDetect, RidgeDetect
from .types import GenericMaskT
from .abstract import GeneralMask

__all__ = [
    'max_planes',

    'region_rel_mask', 'region_abs_mask',

    'squaremask', 'replace_squaremask', 'freeze_replace_squaremask',

    'normalize_mask'
]


def max_planes(*_clips: vs.VideoNode | Iterable[vs.VideoNode], resizer: KernelT = Bilinear) -> vs.VideoNode:
    clips = list[vs.VideoNode](flatten(_clips))  # type: ignore

    assert check_variable_format((model := clips[0]), max_planes)

    resizer = Kernel.ensure_obj(resizer, max_planes)

    width, height, fmt = model.width, model.height, model.format.replace(subsampling_w=0, subsampling_h=0)

    return ExprOp.MAX.combine(
        split(resizer.scale(clip, width, height, format=fmt)) for clip in clips
    )


def _get_region_expr(
    clip: vs.VideoNode | vs.VideoFrame, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
    replace: str | int = 0, rel: bool = False
) -> str:
    right, bottom = right + 1, bottom + 1

    if isinstance(replace, int):
        replace = f'x {replace}'

    if rel:
        return f'X {left} < X {right} > or Y {top} < Y {bottom} > or or {replace} ?'

    return f'X {left} < X {clip.width - right} > or Y {top} < Y {clip.height - bottom} > or or {replace} ?'


def region_rel_mask(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if aka_expr_available:
        return norm_expr(clip, _get_region_expr(clip, left, right, top, bottom, 0), force_akarin=region_rel_mask)

    return clip.std.Crop(left, right, top, bottom).std.AddBorders(left, right, top, bottom)


def region_abs_mask(clip: vs.VideoNode, width: int, height: int, left: int = 0, top: int = 0) -> vs.VideoNode:
    def _crop(w: int, h: int) -> vs.VideoNode:
        return clip.std.CropAbs(width, height, left, top).std.AddBorders(
            left, w - width - left, top, h - height - top
        )

    if 0 in {clip.width, clip.height}:
        if aka_expr_available:
            return norm_expr(
                clip, _get_region_expr(clip, left, left + width, top, top + height, 0, True),
                force_akarin=region_rel_mask
            )

        return clip.std.FrameEval(lambda f, n: _crop(f.width, f.height), clip)

    return region_rel_mask(clip, left, clip.width - width - left, top, clip.height - height - top)


def squaremask(
    clip: vs.VideoNode, width: int, height: int, offset_x: int, offset_y: int, invert: bool = False,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or squaremask

    assert check_variable(clip, func)

    mask_format = clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)

    if offset_x + width > clip.width or offset_y + height > clip.height:
        raise CustomValueError('mask exceeds clip size!')

    base_clip = clip.std.BlankClip(
        width, height, mask_format.id, 1, color=0 if aka_expr_available else get_peak_value(clip), keep=True
    )

    if aka_expr_available:
        mask = norm_expr(
            base_clip, _get_region_expr(
                clip, offset_x, clip.width - width - offset_x, offset_y, clip.height - height - offset_y,
                'range_max x' if invert else 'x range_max'
            ), force_akarin=func
        )
    else:
        mask = base_clip.std.AddBorders(
            offset_x, clip.width - width - offset_x, offset_y, clip.height - height - offset_y
        )
        if invert:
            mask = mask.std.Invert()

    if clip.num_frames == 1:
        return mask

    return mask.std.Loop(clip.num_frames)


def replace_squaremask(
    clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: tuple[int, int, int, int],
    ranges: FrameRangeN | FrameRangesN | None = None, blur_sigma: int | float | None = None,
    invert: bool = False, func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or replace_squaremask

    assert check_variable(clipa, func) and check_variable(clipb, func)

    mask = squaremask(clipb[0], *mask_params, invert, func)

    if isinstance(blur_sigma, int):
        mask = box_blur(mask, blur_sigma)
    elif isinstance(blur_sigma, float):
        mask = gauss_blur(mask, blur_sigma)

    merge = clipa.std.MaskedMerge(clipb, mask.std.Loop(clipa.num_frames))

    return replace_ranges(clipa, merge, ranges)


def freeze_replace_squaremask(
    mask: vs.VideoNode, insert: vs.VideoNode, mask_params: tuple[int, int, int, int],
    frame: int, frame_range: tuple[int, int]
) -> vs.VideoNode:
    start, end = frame_range

    masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)

    return insert_clip(mask, masked_insert * (end - start + 1), start)


def normalize_mask(
    mask: GenericMaskT, clip: vs.VideoNode, ref: vs.VideoNode | None = None,
    *, ridge: bool = False, **kwargs: Any
) -> vs.VideoNode:
    if isinstance(mask, str):
        mask = EdgeDetect.ensure_obj(mask)

    if isinstance(mask, type):
        mask = mask()

    if isinstance(mask, RidgeDetect) and ridge:
        mask = mask.ridgemask(clip, **kwargs)

    if isinstance(mask, EdgeDetect):
        mask = mask.edgemask(clip, **kwargs)

    if isinstance(mask, GeneralMask):
        mask = mask.get_mask(clip, ref)

    if callable(mask):
        if ref is None:
            raise CustomValueError('This mask function requires a ref to be specified!')

        mask = mask(clip, ref)

    return depth(mask, clip)
