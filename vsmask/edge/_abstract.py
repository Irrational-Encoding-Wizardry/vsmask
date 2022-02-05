from __future__ import annotations

__all__ = ['EdgeDetect', 'MatrixEdgeDetect', 'SingleMatrix', 'EuclidianDistance', 'Max', 'RidgeDetect']

import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import ClassVar, List, NoReturn, Optional, Sequence, Tuple, cast

import vapoursynth as vs

from ..util import max_expr

core = vs.core


class _Feature(Enum):
    EDGE = auto()
    RIDGE = auto()


class EdgeDetect(ABC):
    """Abstract edge detection interface."""
    _bits: int

    def edgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0,
        clamp: bool | Tuple[float, float] | List[Tuple[float, float]] = False
    ) -> vs.VideoNode:
        """
        Makes edge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding
        :param clamp:           Clamp to TV or full range if True or specified range `(low, high)`

        :return:                Mask clip
        """
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.EDGE)

    def get_mask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0
    ) -> vs.VideoNode:
        """
        This method is deprecated, pleasee use `edgemask` instead.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding

        :return:                Mask clip
        """
        warnings.warn(f'{self.__class__.__name__}: `get_mask` is deprecated, please use `edgemask` instead', DeprecationWarning)
        return self._mask(clip, lthr, hthr, multi, False, _Feature.EDGE)

    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0,
        clamp: bool | Tuple[float, float] | List[Tuple[float, float]] = False
    ) -> vs.VideoNode | NoReturn:
        """
        Makes ridge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding
        :param clamp:           Clamp to TV or full range if True or specified range `(low, high)`

        :return:                Mask clip
        """
        raise NotImplementedError

    def _mask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0,
        clamp: bool | Tuple[float, float] | List[Tuple[float, float]] = False,
        feature: _Feature = _Feature.EDGE
    ) -> vs.VideoNode:
        if not clip.format:
            raise ValueError('Variable format not allowed!')

        self._bits = clip.format.bits_per_sample
        is_float = clip.format.sample_type == vs.FLOAT
        peak = 1.0 if is_float else (1 << self._bits) - 1
        hthr = peak if hthr is None else hthr

        clip_p = self._preprocess(clip)
        if feature == _Feature.EDGE:
            mask = self._compute_edge_mask(clip_p)
        elif feature == _Feature.RIDGE:
            mask = self._compute_ridge_mask(clip_p)
        mask = self._postprocess(mask)

        if multi != 1:
            if is_float:
                mask = mask.std.Expr(f'x {multi} *')
            else:
                def _multi_func(x: int) -> int:
                    return round(max(min(x * multi, peak), 0))
                mask = mask.std.Lut(function=_multi_func)

        if lthr > 0 or hthr < peak:
            if is_float:
                mask = mask.std.Expr(f'x {hthr} > {peak} x {lthr} <= 0 x ? ?')
            else:
                def _thr_func(x: int) -> int | float:
                    return peak if x > hthr else 0 if x <= lthr else x  # type: ignore[operator]
                mask = mask.std.Lut(function=_thr_func)

        if clamp:
            if isinstance(clamp, list):
                mask = core.std.Expr(mask, ['x {} max {} min'.format(*c) for c in clamp])
            if isinstance(clamp, tuple):
                mask = core.std.Expr(mask, 'x {} max {} min'.format(*clamp))
            else:
                assert mask.format
                if is_float:
                    clamp_vals = [(0., 1.), (-0.5, 0.5), (-0.5, 0.5)]
                else:
                    with mask.get_frame(0) as f:
                        crange = cast(int, f.props['_ColorRange'])
                    clamp_vals = [(0, peak)] * 3 if crange == 0 else [
                        (16 << self._bits - 8, 235 << self._bits - 8),
                        (16 << self._bits - 8, 240 << self._bits - 8),
                        (16 << self._bits - 8, 240 << self._bits - 8)
                    ]

                mask = core.std.Expr(mask, ['x {} max {} min'.format(*c) for c in clamp_vals[:mask.format.num_planes]])

        return mask

    @abstractmethod
    def _compute_edge_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        raise NotImplementedError

    @abstractmethod
    def _compute_ridge_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        raise NotImplementedError

    def _preprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip


class MatrixEdgeDetect(EdgeDetect, ABC):
    matrices: ClassVar[Sequence[Sequence[float]]]
    divisors: ClassVar[Optional[Sequence[float]]] = None
    mode_types: ClassVar[Optional[Sequence[str]]] = None

    def _compute_edge_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self._merge_edge([
            clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
            for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())
        ])

    def _compute_ridge_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        def _x(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[0], divisor=self._get_divisors()[0])

        def _y(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[1], divisor=self._get_divisors()[1])

        x = _x(clip)
        y = _y(clip)
        xx = _x(x)
        yy = _y(y)
        xy = _x(x)
        return self._merge_ridge([xx, yy, xy])

    @abstractmethod
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        raise NotImplementedError

    @abstractmethod
    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.matrices

    def _get_divisors(self) -> Sequence[float]:
        return self.divisors if self.divisors else [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> Sequence[str]:
        return self.mode_types if self.mode_types else ['s'] * len(self._get_matrices())

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        if len(self.matrices[0]) > 9 or (self.mode_types and self.mode_types[0] != 's'):
            clip = clip.std.Crop(
                right=clip.format.subsampling_w * 2 if clip.format and clip.format.subsampling_w != 0 else 2
            ).resize.Point(clip.width, src_width=clip.width)
        return clip


class RidgeDetect(MatrixEdgeDetect):
    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0,
        clamp: bool | Tuple[float, float] | List[Tuple[float, float]] = False,
    ) -> vs.VideoNode:
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.RIDGE)

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        # return core.std.Expr(clips, 'x dup * z dup * 4 * + x y * 2 * - y dup * + sqrt x y + +')
        return core.std.Expr(clips, 'x y * 2 * -1 * x dup * z dup * 4 * + y dup * + + sqrt x y + +')


class SingleMatrix(MatrixEdgeDetect, ABC):
    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0,
        clamp: bool | Tuple[float, float] | List[Tuple[float, float]] = False
    ) -> vs.VideoNode | NoReturn:
        raise NotImplementedError

    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return clips[0]

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


class EuclidianDistance(MatrixEdgeDetect, ABC):
    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0,
        clamp: bool | Tuple[float, float] | List[Tuple[float, float]] = False
    ) -> vs.VideoNode | NoReturn:
        raise NotImplementedError

    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, 'x x * y y * + sqrt')

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


class Max(MatrixEdgeDetect, ABC):
    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0,
        clamp: bool | Tuple[float, float] | List[Tuple[float, float]] = False
    ) -> vs.VideoNode | NoReturn:
        raise NotImplementedError

    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, max_expr(len(clips)))

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError
