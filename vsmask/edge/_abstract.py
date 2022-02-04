__all__ = ['EdgeDetect', 'MatrixEdgeDetect', 'SingleMatrixDetect', 'EuclidianDistanceMatrixDetect', 'MaxDetect']

from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Sequence

import vapoursynth as vs

from ..util import _pick_px_op, max_expr

core = vs.core


class EdgeDetect(ABC):
    """Abstract edge detection interface."""
    _bits: int

    def get_mask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: Optional[float] = None,
        multi: float = 1.0
    ) -> vs.VideoNode:
        """
        Makes edge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding

        :return:                Mask clip
        """

        if clip.format is None:
            raise ValueError('get_mask: Variable format not allowed!')

        self._bits = clip.format.bits_per_sample
        is_float = clip.format.sample_type == vs.FLOAT
        peak = 1.0 if is_float else (1 << self._bits) - 1
        hthr = peak if hthr is None else hthr

        clip_p = self._preprocess(clip)
        mask = self._compute_mask(clip_p)
        mask = self._postprocess(mask)

        if multi != 1:
            mask = _pick_px_op(
                use_expr=is_float,
                expr=f'x {multi} *',
                lut=lambda x: round(max(min(x * multi, peak), 0))
            )(mask)

        if lthr > 0 or hthr < peak:
            mask = _pick_px_op(
                use_expr=is_float,
                expr=f'x {hthr} > {peak} x {lthr} <= 0 x ? ?',
                lut=lambda x: peak if x > hthr else 0 if x <= lthr else x
            )(mask)

        return mask

    @abstractmethod
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        raise NotImplementedError

    def _preprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip


class MatrixEdgeDetect(EdgeDetect, ABC):
    matrices: ClassVar[Sequence[Sequence[float]]]
    divisors: ClassVar[Optional[Sequence[float]]] = None
    mode_types: ClassVar[Optional[Sequence[str]]] = None

    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self._merge([
            clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
            for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())
        ])

    @abstractmethod
    def _merge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
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


class SingleMatrix(MatrixEdgeDetect, ABC):
    def _merge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return clips[0]


class EuclidianDistance(MatrixEdgeDetect, ABC):
    def _merge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, 'x x * y y * + sqrt')


class Max(MatrixEdgeDetect, ABC):
    def _merge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, max_expr(len(clips)))
