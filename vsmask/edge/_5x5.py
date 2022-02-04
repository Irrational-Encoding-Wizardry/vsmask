__all__ = [
    'Matrix5x5',
    # Single matrix
    'ExLaplacian1', 'ExLaplacian2', 'ExLaplacian3', 'ExLaplacian4',
    'LoG',
    # Euclidian distance
    'ExPrewitt',
    'ExSobel',
    'FDoG', 'FDOG', 'FDoGTCanny', 'FDOGTCanny',
    'DoG',
    'Farid',
    # Max
    'ExKirsch'
]

from abc import ABC
from typing import Sequence

import vapoursynth as vs
from vsutil import Range, depth

from ._abstract import EdgeDetect, EuclidianDistance, Max, RidgeDetect, SingleMatrix


class Matrix5x5(EdgeDetect, ABC):
    ...


# Single matrix
class ExLaplacian1(SingleMatrix, Matrix5x5):
    """Extended Pierre-Simon de Laplace operator, 1st implementation."""
    matrices = [[0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, 8, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]]


class ExLaplacian2(SingleMatrix, Matrix5x5):
    """Extended Pierre-Simon de Laplace operator, 2nd implementation."""
    matrices = [[0, 1, -1, 1, 0, 1, 1, -4, 1, 1, -1, -4, 8, -4, -1, 1, 1, -4, 1, 1, 0, 1, -1, 1, 0]]


class ExLaplacian3(SingleMatrix, Matrix5x5):
    """Extended Pierre-Simon de Laplace operator, 3rd implementation."""
    matrices = [[-1, 1, -1, 1, -1, 1, 2, -4, 2, 1, -1, -4, 8, -4, -1, 1, 2, -4, 2, 1, -1, 1, -1, 1, -1]]


class ExLaplacian4(SingleMatrix, Matrix5x5):
    """Extended Pierre-Simon de Laplace operator, 4th implementation."""
    matrices = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


class LoG(SingleMatrix, Matrix5x5):
    """Laplacian of Gaussian operator."""
    matrices = [[0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0]]


# Euclidian distance
class ExPrewitt(RidgeDetect, EuclidianDistance, Matrix5x5):
    """Extended Judith M. S. Prewitt operator."""
    matrices = [
        [2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2]
    ]


class ExSobel(RidgeDetect, EuclidianDistance, Matrix5x5):
    """Extended Sobel–Feldman operator."""
    matrices = [
        [2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 4, 2, 0, -2, -4, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
        [2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -2, -2, -4, -2, -2]
    ]


class FDoG(RidgeDetect, EuclidianDistance, Matrix5x5):
    """Flow-based Difference of Gaussian"""
    matrices = [
        [1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1],
        [1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, -1, -2, -3, -2, -1, -1, -2, -3, -2, -1]
    ]
    divisors = [2, 2]


class FDOG(FDoG):
    ...


class FDoGTCanny(Matrix5x5, EdgeDetect):
    """Flow-based Difference of Gaussian TCanny Vapoursynth plugin."""
    def _compute_edge_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(0, mode=1, op=6, scale=0.5)


class FDOGTCanny(FDoGTCanny):
    ...


class DoG(EuclidianDistance, Matrix5x5):
    """Zero-cross (of the 2nd derivative) of a Difference of Gaussians"""
    matrices = [
        [0, 0, 5, 0, 0, 0, 5, 10, 5, 0, 5, 10, 20, 10, 5, 0, 5, 10, 5, 0, 0, 0, 5, 0, 0],
        [0, 25, 0, 25, 50, 25, 0, 25, 0],
    ]
    divisors = [4, 6]

    def _preprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, 32)

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, self._bits, range=Range.FULL, range_in=Range.FULL)

    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return vs.core.std.Expr(clips, 'x y -')


class Farid(RidgeDetect, EuclidianDistance, Matrix5x5):
    """Farid & Simoncelli operator."""
    matrices = [
        [0.004127602875174862, 0.027308149775363867, 0.04673225765917656, 0.027308149775363867, 0.004127602875174862,
         0.010419993699470744, 0.06893849946536831, 0.11797400212587895, 0.06893849946536831, 0.010419993699470744,
         0.0, 0.0, 0.0, 0.0, 0.0,
         -0.010419993699470744, -0.06893849946536831, -0.11797400212587895, -0.06893849946536831, -0.010419993699470744,
         -0.004127602875174862, -0.027308149775363867, -0.04673225765917656, -0.027308149775363867, -0.004127602875174862],
        [0.004127602875174862, 0.027308149775363867, 0.04673225765917656, 0.027308149775363867, 0.004127602875174862,
         0.010419993699470744, 0.06893849946536831, 0.11797400212587895, 0.06893849946536831, 0.010419993699470744,
         0.0, 0.0, 0.0, 0.0, 0.0,
         -0.010419993699470744, -0.06893849946536831, -0.11797400212587895, -0.06893849946536831, -0.010419993699470744,
         -0.004127602875174862, -0.027308149775363867, -0.04673225765917656, -0.027308149775363867, -0.004127602875174862]
    ]

    def _preprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, 32)

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, self._bits, range=Range.FULL, range_in=Range.FULL)


# Max
class ExKirsch(Max):
    """Extended Russell Kirsch compass operator. 5x5 matrices."""
    matrices = [
        [9, 9, 9, 9, 9, 9, 5, 5, 5, 9, -7, -3, 0, -3, -7, -7, -3, -3, -3, -7, -7, -7, -7, -7, -7],
        [9, 9, 9, 9, -7, 9, 5, 5, -3, -7, 9, 5, 0, -3, -7, 9, -3, -3, -3, -7, -7, -7, -7, -7, -7],
        [9, 9, -7, -7, -7, 9, 5, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, -3, -3, -7, 9, 9, -7, -7, -7],
        [-7, -7, -7, -7, -7, 9, -3, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, 5, -3, -7, 9, 9, 9, 9, -7],
        [-7, -7, -7, -7, -7, -7, -3, -3, -3, -7, -7, -3, 0, -3, -7, 9, 5, 5, 5, 9, 9, 9, 9, 9, 9],
        [-7, -7, -7, -7, -7, -7, -3, -3, -3, 9, -7, -3, 0, 5, 9, -7, -3, 5, 5, 9, -7, 9, 9, 9, 9],
        [-7, -7, -7, 9, 9, -7, -3, -3, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, 5, 9, -7, -7, -7, 9, 9],
        [-7, 9, 9, 9, 9, -7, -3, 5, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, -3, 9, -7, -7, -7, -7, -7]
    ]
