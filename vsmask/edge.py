
import math
from abc import ABC, abstractmethod
from typing import List, Optional

import vapoursynth as vs
from vsutil import Range, depth

from .util import pick_px_op, max_expr

core = vs.core


class EdgeDetect(ABC):
    """Abstract edge detection interface."""

    def get_mask(self, clip: vs.VideoNode,
                 lthr: float = 0.0, hthr: Optional[float] = None, multi: float = 1.0) -> vs.VideoNode:
        """
        Makes edge mask based on convolution kernel.
        he resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding

        :return:                Mask clip
        """
        if clip.format is None:
            raise ValueError('get_mask: Variable format not allowed!')

        bits = clip.format.bits_per_sample
        is_float = clip.format.sample_type == vs.FLOAT
        peak = 1.0 if is_float else (1 << bits) - 1
        hthr = peak if hthr is None else hthr


        clip_p = self._preprocess(clip)
        mask = self._compute_mask(clip_p)

        mask = depth(mask, bits, range=Range.FULL, range_in=Range.FULL)


        if multi != 1:
            mask = pick_px_op(
                use_expr=is_float,
                expr=f'x {multi} *',
                lut=lambda x: round(max(min(x * multi, peak), 0))
            )(mask)


        if lthr > 0 or hthr < peak:
            mask = pick_px_op(
                use_expr=is_float,
                expr=f'x {hthr} > {peak} x {lthr} <= 0 x ? ?',
                lut=lambda x: peak if x > hthr else 0 if x <= lthr else x
            )(mask)


        return mask

    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        masks = [clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
                 for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())]

        expr = self._get_expr()
        mask = core.std.Expr(masks, expr) if expr else masks[0]

        return mask

    def _get_divisors(self) -> List[float]:
        return [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> List[str]:
        return ['s'] * len(self._get_matrices())

    @staticmethod
    def _get_expr() -> Optional[str]:
        return None

    @staticmethod
    def _preprocess(clip: vs.VideoNode) -> vs.VideoNode:
        return clip

    @staticmethod
    @abstractmethod
    def _get_matrices() -> List[List[float]]:
        pass


class Laplacian1(EdgeDetect):
    """Pierre-Simon de Laplace operator 1st implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, -1, 0, -1, 4, -1, 0, -1, 0]]


class Laplacian2(EdgeDetect):
    """Pierre-Simon de Laplace operator 2nd implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, -2, 1, -2, 4, -2, 1, -2, 1]]


class Laplacian3(EdgeDetect):
    """Pierre-Simon de Laplace operator 3rd implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[2, -1, 2, -1, -4, -1, 2, -1, 2]]


class Laplacian4(EdgeDetect):
    """Pierre-Simon de Laplace operator 4th implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-1, -1, -1, -1, 8, -1, -1, -1, -1]]


class ExLaplacian1(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 1st implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, 8, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]]


class ExLaplacian2(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 2nd implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 1, -1, 1, 0, 1, 1, -4, 1, 1, -1, -4, 8, -4, -1, 1, 1, -4, 1, 1, 0, 1, -1, 1, 0]]


class ExLaplacian3(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 3rd implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-1, 1, -1, 1, -1, 1, 2, -4, 2, 1, -1, -4, 8, -4, -1, 1, 2, -4, 2, 1, -1, 1, -1, 1, -1]]


class ExLaplacian4(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 4th implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


class Kayyali(EdgeDetect):
    """Kayyali operator. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[6, 0, -6, 0, 0, 0, -6, 0, 6]]


class LoG(EdgeDetect):
    """Laplacian of Gaussian. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0]]


class Roberts(EdgeDetect):
    """Lawrence Roberts operator. 2x2 matrices computed in 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 0, 0, 0, 1, 0, 0, 0, -1],
                [0, 1, 0, -1, 0, 0, 0, 0, 0]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Prewitt(EdgeDetect):
    """Judith M. S. Prewitt operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 0, -1, 1, 0, -1, 1, 0, -1],
                [1, 1, 1, 0, 0, 0, -1, -1, -1]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class PrewittStd(EdgeDetect):
    """Judith M. S. Prewitt Vapoursynth plugin operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[]]

    @staticmethod
    def _compute_mask(clip: vs.VideoNode) -> vs.VideoNode:
        return core.std.Prewitt(clip)


class ExPrewitt(EdgeDetect):
    """Extended Judith M. S. Prewitt operator. 5x5 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
                [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Sobel(EdgeDetect):
    """Sobel–Feldman operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 0, -1, 2, 0, -2, 1, 0, -1],
                [1, 2, 1, 0, 0, 0, -1, -2, -1]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class SobelStd(EdgeDetect):
    """Sobel–Feldman Vapoursynth plugin operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[]]

    @staticmethod
    def _compute_mask(clip: vs.VideoNode) -> vs.VideoNode:
        return core.std.Sobel(clip)


class ExSobel(EdgeDetect):
    """Extended Sobel–Feldman operator. 5x5 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 4, 2, 0, -2, -4, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
                [2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -2, -2, -4, -2, -2]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Scharr(EdgeDetect):
    """H. Scharr optimized operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-3, 0, 3, -10, 0, 10, -3, 0, 3],
                [-3, -10, -3, 0, 0, 0, 3, 10, 3]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class FDOG(EdgeDetect):
    """Flow-based Difference Of Gaussian operator. 3x3 matrices from G41Fun."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1],
                [1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, -1, -2, -3, -2, -1, -1, -2, -3, -2, -1]]

    @staticmethod
    def _get_divisors() -> List[float]:
        return [2, 2]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Kroon(EdgeDetect):
    """Dirk-Jan Kroon operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-17, 0, 17, -61, 0, 61, -17, 0, 17],
                [-17, -61, -17, 0, 0, 0, 17, 61, 17]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class FreyChen(EdgeDetect):
    """Chen Frei operator. 3x3 matrices properly implemented."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        sqrt2 = math.sqrt(2)
        return [[1, sqrt2, 1, 0, 0, 0, -1, -sqrt2, -1],
                [1, 0, -1, sqrt2, 0, -sqrt2, 1, 0, -1],
                [0, -1, sqrt2, 1, 0, -1, -sqrt2, 1, 0],
                [sqrt2, -1, 0, -1, 0, 1, 0, 1, -sqrt2],
                [0, 1, 0, -1, 0, -1, 0, 1, 0],
                [-1, 0, 1, 0, 0, 0, 1, 0, -1],
                [1, -2, 1, -2, 4, -2, 1, -2, 1],
                [-2, 1, -2, 1, 4, 1, -2, 1, -2],
                [1, 1, 1, 1, 1, 1, 1, 1, 1]]

    @staticmethod
    def _get_divisors() -> List[float]:
        sqrt2 = math.sqrt(2)
        return [2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2, 2, 6, 6, 3]

    @staticmethod
    def _get_expr() -> Optional[str]:
        M = 'x x * y y * + z z * + a a * +'
        S = f'b b * c c * + d d * + e e * + f f * + {M} +'
        return f'{M} {S} / sqrt'

    @staticmethod
    def _preprocess(clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, 32)


class FreyChenG41(EdgeDetect):
    """"Chen Frei" operator. 3x3 matrices from G41Fun."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-7, 0, 7, -10, 0, 10, -7, 0, 7],
                [-7, -10, -7, 0, 0, 0, 7, 10, 7]]

    @staticmethod
    def _get_divisors() -> List[float]:
        return [7, 7]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class TEdge(EdgeDetect):
    """(TEdgeMasktype=2) Avisynth plugin. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[12, -74, 0, 74, -12],
                [-12, 74, 0, -74, 12]]

    @staticmethod
    def _get_divisors() -> List[float]:
        return [62, 62]

    @staticmethod
    def _get_mode_types() -> List[str]:
        return ['h', 'v']

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class TEdgeTedgemask(EdgeDetect):
    """(tedgemask.TEdgeMask(threshold=0.0, type=2)) Vapoursynth plugin. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[]]

    @staticmethod
    def _compute_mask(clip: vs.VideoNode) -> vs.VideoNode:
        return core.tedgemask.TEdgeMask(clip, threshold=0, type=2)


class Robinson3(EdgeDetect):
    """Robinson compass operator level 3. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 1, 1, 0, 0, 0, -1, -1, -1],
                [1, 1, 0, 1, 0, -1, 0, -1, -1],
                [1, 0, -1, 1, 0, -1, 1, 0, -1],
                [0, -1, -1, 1, 0, -1, 1, 1, 0]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(4)


class Robinson5(EdgeDetect):
    """Robinson compass operator level 5. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 2, 1, 0, 0, 0, -1, -2, -1],
                [2, 1, 0, 1, 0, -1, 0, -1, -2],
                [1, 0, -1, 2, 0, -2, 1, 0, -1],
                [0, -1, -2, 1, 0, -1, 2, 1, 0]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(4)


class Kirsch(EdgeDetect):
    """Russell Kirsch compass operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[5, 5, 5, -3, 0, -3, -3, -3, -3],
                [5, 5, -3, 5, 0, -3, -3, -3, -3],
                [5, -3, -3, 5, 0, -3, 5, -3, -3],
                [-3, -3, -3, 5, 0, -3, 5, 5, -3],
                [-3, -3, -3, -3, 0, -3, 5, 5, 5],
                [-3, -3, -3, -3, 0, 5, -3, 5, 5],
                [-3, -3, 5, -3, 0, 5, -3, -3, 5],
                [-3, 5, 5, -3, 0, 5, -3, -3, -3]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(8)


class ExKirsch(EdgeDetect):
    """Extended Russell Kirsch compass operator. 5x5 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[9, 9, 9, 9, 9, 9, 5, 5, 5, 9, -7, -3, 0, -3, -7, -7, -3, -3, -3, -7, -7, -7, -7, -7, -7],
                [9, 9, 9, 9, -7, 9, 5, 5, -3, -7, 9, 5, 0, -3, -7, 9, -3, -3, -3, -7, -7, -7, -7, -7, -7],
                [9, 9, -7, -7, -7, 9, 5, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, -3, -3, -7, 9, 9, -7, -7, -7],
                [-7, -7, -7, -7, -7, 9, -3, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, 5, -3, -7, 9, 9, 9, 9, -7],
                [-7, -7, -7, -7, -7, -7, -3, -3, -3, -7, -7, -3, 0, -3, -7, 9, 5, 5, 5, 9, 9, 9, 9, 9, 9],
                [-7, -7, -7, -7, -7, -7, -3, -3, -3, 9, -7, -3, 0, 5, 9, -7, -3, 5, 5, 9, -7, 9, 9, 9, 9],
                [-7, -7, -7, 9, 9, -7, -3, -3, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, 5, 9, -7, -7, -7, 9, 9],
                [-7, 9, 9, 9, 9, -7, -3, 5, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, -3, 9, -7, -7, -7, -7, -7]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(8)


def get_all_edge_detects(clip: vs.VideoNode, **kwargs) -> List[vs.VideoNode]:
    masks = [
        edge_detect().get_mask(clip, **kwargs).text.Text(edge_detect.__name__)
        for edge_detect in EdgeDetect.__subclasses__()
    ]
    return masks
