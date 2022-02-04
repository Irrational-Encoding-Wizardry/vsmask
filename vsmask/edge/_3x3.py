__all__ = [
    'Matrix3x3',
    # Single matrix
    'Laplacian1', 'Laplacian2', 'Laplacian3', 'Laplacian4',
    'Kayyali',
    # Euclidian Distance
    'Tritical', 'TriticalTCanny',
    'Cross',
    'Prewitt', 'PrewittStd', 'PrewittTCanny',
    'Sobel', 'SobelStd', 'SobelTCanny', 'ASobel',
    'Scharr', 'RScharr', 'ScharrTCanny',
    'Kroon', 'KroonTCanny',
    'FreyChenG41', 'FreyChen',
    # Max
    'Robinson3', 'Robinson5', 'TheToof',
    'Kirsch', 'KirschTCanny',
    # Misc
    'MinMax'
]

import math
from abc import ABC
from typing import Sequence, Tuple

import vapoursynth as vs
from vsutil import get_depth, depth, Range

from ..better_vsutil import join, split
from ..util import XxpandMode, expand, inpand
from ._abstract import EdgeDetect, EuclidianDistanceMatrixDetect, MatrixEdgeDetect, MaxDetect, SingleMatrixDetect


class Matrix3x3(EdgeDetect, ABC):
    ...


# Single matrix
class Laplacian1(SingleMatrixDetect, Matrix3x3):
    """Pierre-Simon de Laplace operator 1st implementation."""
    matrices = [[0, -1, 0, -1, 4, -1, 0, -1, 0]]


class Laplacian2(SingleMatrixDetect, Matrix3x3):
    """Pierre-Simon de Laplace operator 2nd implementation."""
    matrices = [[1, -2, 1, -2, 4, -2, 1, -2, 1]]


class Laplacian3(SingleMatrixDetect, Matrix3x3):
    """Pierre-Simon de Laplace operator 3rd implementation."""
    matrices = [[2, -1, 2, -1, -4, -1, 2, -1, 2]]


class Laplacian4(SingleMatrixDetect, Matrix3x3):
    """Pierre-Simon de Laplace operator 4th implementation."""
    matrices = [[-1, -1, -1, -1, 8, -1, -1, -1, -1]]


class Kayyali(SingleMatrixDetect, Matrix3x3):
    """Kayyali operator."""
    matrices = [[6, 0, -6, 0, 0, 0, -6, 0, 6]]


# Euclidian Distance
class Tritical(EuclidianDistanceMatrixDetect, Matrix3x3):
    """
    Operator used in Tritical's original TCanny filter.
    Plain and simple orthogonal first order derivative.
    """
    matrices = [
        [0, 0, 0, -1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, -1, 0]
    ]


class TriticalTCanny(Matrix3x3, EdgeDetect):
    """
    Operator used in Tritical's original TCanny filter.
    Plain and simple orthogonal first order derivative.
    """
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(0, mode=1, op=0)


class Cross(EuclidianDistanceMatrixDetect, Matrix3x3):
    """
    "HotDoG" Operator from AVS ExTools by Dogway.
    Plain and simple cross first order derivative.
    """
    matrices = [
        [1, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, -1, 0, 0, 0, 1, 0, 0]
    ]


class Prewitt(EuclidianDistanceMatrixDetect, Matrix3x3):
    """Judith M. S. Prewitt operator."""
    matrices = [
        [1, 0, -1, 1, 0, -1, 1, 0, -1],
        [1, 1, 1, 0, 0, 0, -1, -1, -1]
    ]


class PrewittStd(Matrix3x3, EdgeDetect):
    """Judith M. S. Prewitt Vapoursynth plugin operator."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.std.Prewitt()


class PrewittTCanny(Matrix3x3, EdgeDetect):
    """Judith M. S. Prewitt TCanny plugin operator."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(0, mode=1, op=1, scale=2)


class Sobel(EuclidianDistanceMatrixDetect, Matrix3x3):
    """Sobel–Feldman operator."""
    matrices = [
        [1, 0, -1, 2, 0, -2, 1, 0, -1],
        [1, 2, 1, 0, 0, 0, -1, -2, -1]
    ]


class SobelStd(Matrix3x3, EdgeDetect):
    """Sobel–Feldman Vapoursynth plugin operator."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.std.Sobel()


class SobelTCanny(Matrix3x3, EdgeDetect):
    """Sobel–Feldman Vapoursynth plugin operator."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(0, mode=1, op=2, scale=2)


class ASobel(Matrix3x3, EdgeDetect):
    """Modified Sobel–Feldman operator from AWarpSharp."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        # warp.ASobel and warpsf.ASobel have different function signatures
        # so mypy set the ternary expression as Callable[..., Any]
        # which makes sense.
        # Since we're using ``warn_return_any = True`` in mypy config,
        # mypy warns us about not being able to call a function of unknown type
        # and returning Any from ``_compute_mask`` declared to return "VideoNode".
        # I could edit the stubs files but then, they will be wrong and adding more boilerplate code
        # for just satisfy mypy here doesn't seem to be very relevant.
        return (vs.core.warp.ASobel if get_depth(clip) < 32 else vs.core.warpsf.ASobel)(clip, 255)  # type: ignore


class Scharr(EuclidianDistanceMatrixDetect, Matrix3x3):
    """
    Original H. Scharr optimised operator which attempts
    to achieve the perfect rotational symmetry with coefficients 3 and 10.

    """
    matrices = [
        [-3, 0, 3, -10, 0, 10, -3, 0, 3],
        [-3, -10, -3, 0, 0, 0, 3, 10, 3]
    ]
    divisors = [3, 3]


class RScharr(EuclidianDistanceMatrixDetect, Matrix3x3):
    """
    Refined H. Scharr operator to more accurately calculate
    1st derivatives for a 3x3 kernel with coeffs 47 and 162.

    """
    matrices = [
        [-47, 0, 47, -162, 0, 162, -47, 0, 47],
        [-47, -162, -47, 0, 0, 0, 47, 162, 47]
    ]
    divisors = [47, 47]


class ScharrTCanny(Matrix3x3, EdgeDetect):
    """H. Scharr optimised TCanny Vapoursynth plugin operator."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(0, mode=1, op=2, scale=4 / 3)


class Kroon(EuclidianDistanceMatrixDetect, Matrix3x3):
    """Dirk-Jan Kroon operator."""
    matrices = [
        [-17, 0, 17, -61, 0, 61, -17, 0, 17],
        [-17, -61, -17, 0, 0, 0, 17, 61, 17]
    ]


class KroonTCanny(Matrix3x3, EdgeDetect):
    """Dirk-Jan Kroon TCanny Vapoursynth plugin operator."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(0, mode=1, op=4)


class FreyChen(MatrixEdgeDetect):
    """Chen Frei operator. 3x3 matrices properly implemented."""
    sqrt2 = math.sqrt(2)
    matrices = [
        [1, sqrt2, 1, 0, 0, 0, -1, -sqrt2, -1],
        [1, 0, -1, sqrt2, 0, -sqrt2, 1, 0, -1],
        [0, -1, sqrt2, 1, 0, -1, -sqrt2, 1, 0],
        [sqrt2, -1, 0, -1, 0, 1, 0, 1, -sqrt2],
        [0, 1, 0, -1, 0, -1, 0, 1, 0],
        [-1, 0, 1, 0, 0, 0, 1, 0, -1],
        [1, -2, 1, -2, 4, -2, 1, -2, 1],
        [-2, 1, -2, 1, 4, 1, -2, 1, -2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    divisors = [
        2 * sqrt2,
        2 * sqrt2,
        2 * sqrt2,
        2 * sqrt2,
        2,
        2,
        6,
        6,
        3
    ]

    def _preprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, 32)

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, self._bits, range=Range.FULL, range_in=Range.FULL)

    def _merge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        M = 'x x * y y * + z z * + a a * +'
        S = f'b b * c c * + d d * + e e * + f f * + {M} +'
        return vs.core.std.Expr(clips, f'{M} {S} / sqrt')


class FreyChenG41(EuclidianDistanceMatrixDetect, Matrix3x3):
    """"Chen Frei" operator. 3x3 matrices from G41Fun."""
    matrices = [
        [-7, 0, 7, -10, 0, 10, -7, 0, 7],
        [-7, -10, -7, 0, 0, 0, 7, 10, 7]
    ]
    divisors = [7, 7]


# Max
class Robinson3(MaxDetect, Matrix3x3):
    """Robinson compass operator level 3."""
    matrices = [
        [1, 1, 1, 0, 0, 0, -1, -1, -1],
        [1, 1, 0, 1, 0, -1, 0, -1, -1],
        [1, 0, -1, 1, 0, -1, 1, 0, -1],
        [0, -1, -1, 1, 0, -1, 1, 1, 0]
    ]


class Robinson5(MaxDetect, Matrix3x3):
    """Robinson compass operator level 5."""
    matrices = [
        [1, 2, 1, 0, 0, 0, -1, -2, -1],
        [2, 1, 0, 1, 0, -1, 0, -1, -2],
        [1, 0, -1, 2, 0, -2, 1, 0, -1],
        [0, -1, -2, 1, 0, -1, 2, 1, 0]
    ]


class TheToof(MaxDetect, Matrix3x3):
    """TheToof compass operator from SharpAAMCmod."""
    matrices = [
        [5, 10, 5, 0, 0, 0, -5, -10, -5],
        [10, 5, 0, 5, 0, -5, 0, -5, -10],
        [5, 0, -5, 10, 0, -10, 5, 0, -5],
        [0, -5, -10, 5, 0, -5, 10, 5, 0]
    ]
    divisors = [4] * 4


class Kirsch(MaxDetect, Matrix3x3):
    """Russell Kirsch compass operator."""
    matrices = [
        [5, 5, 5, -3, 0, -3, -3, -3, -3],
        [5, 5, -3, 5, 0, -3, -3, -3, -3],
        [5, -3, -3, 5, 0, -3, 5, -3, -3],
        [-3, -3, -3, 5, 0, -3, 5, 5, -3],
        [-3, -3, -3, -3, 0, -3, 5, 5, 5],
        [-3, -3, -3, -3, 0, 5, -3, 5, 5],
        [-3, -3, 5, -3, 0, 5, -3, -3, 5],
        [-3, 5, 5, -3, 0, 5, -3, -3, -3]
    ]


class KirschTCanny(Matrix3x3, EdgeDetect):
    """Russell Kirsch compass TCanny Vapoursynth plugin operator."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(0, mode=1, op=5)


# Misc
class MinMax(EdgeDetect):
    """Min/max mask with separate luma/chroma radii."""
    radii: Tuple[int, int, int]

    def __init__(self, rady: int = 2, radc: int = 0) -> None:
        """
        :param rady:    Luma radius
        :param radc:    Chroma radius
        """
        super().__init__()
        self.radii = (rady, radc, radc)

    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        assert clip.format
        planes = [
            vs.core.std.Expr(
                [expand(p, rad, rad, XxpandMode.ELLIPSE),
                 inpand(p, rad, rad, XxpandMode.ELLIPSE)],
                'x y -')
            for p, rad in zip(split(clip), self.radii)
        ]
        return join(planes, clip.format.color_family)
