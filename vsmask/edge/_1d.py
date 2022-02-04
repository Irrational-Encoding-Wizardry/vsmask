__all__ = [
    'Matrix1D',
    'TEdge', 'TEdgeTedgemask',
]

from abc import ABC

import vapoursynth as vs

from ._abstract import EdgeDetect, EuclidianDistanceMatrixDetect


class Matrix1D(EdgeDetect, ABC):
    ...


class TEdge(EuclidianDistanceMatrixDetect, Matrix1D):
    """(TEdgeMasktype=2) Avisynth plugin."""
    matrices = [
        [12, -74, 0, 74, -12],
        [-12, 74, 0, -74, 12]
    ]
    divisors = [62, 62]
    mode_types = ['h', 'v']


class TEdgeTedgemask(Matrix1D, EdgeDetect):
    """(tedgemask.TEdgeMask(threshold=0.0, type=2)) Vapoursynth plugin."""
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tedgemask.TEdgeMask(threshold=0, type=2)
