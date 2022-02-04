__all__ = ['get_all_edge_detects']

from typing import List, Optional, Set, Type

import vapoursynth as vs

from ._abstract import EdgeDetect

core = vs.core


def get_all_edge_detects(
    clip: vs.VideoNode,
    lthr: float = 0.0, hthr: Optional[float] = None,
    multi: float = 1.0
) -> List[vs.VideoNode]:
    """
    Returns all the EdgeDetect subclasses

    :param clip:        Source clip
    :param lthr:        See :py:func:`EdgeDetect.get_mask`
    :param hthr:        See :py:func:`EdgeDetect.get_mask`
    :param multi:       See :py:func:`EdgeDetect.get_mask`
    :return:            A list edge masks
    """
    def _all_subclasses(cls: Type[EdgeDetect]) -> Set[Type[EdgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s for s in _all_subclasses(EdgeDetect)  # type: ignore
        if s.__name__ not in {
            'MatrixEdgeDetect', 'SingleMatrix', 'EuclidianDistance', 'Max',
            'Matrix1D', 'SavitzkyGolay',
            'Matrix2x2', 'Matrix3x3', 'Matrix5x5'
        }
    }
    return [
        edge_detect().get_mask(clip, lthr, hthr, multi).text.Text(edge_detect.__name__)  # type: ignore
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]
