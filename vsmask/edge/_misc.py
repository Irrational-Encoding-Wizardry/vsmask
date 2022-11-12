from __future__ import annotations

__all__ = ['get_all_edge_detects', 'get_all_ridge_detect']

import warnings
from abc import ABCMeta
from typing import Any, Callable, TypeVar, cast

from vstools import vs

from ._abstract import EdgeDetect, RidgeDetect


def get_all_edge_detects(
    clip: vs.VideoNode,
    lthr: float = 0.0, hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False
) -> list[vs.VideoNode]:
    """
    Returns all the EdgeDetect subclasses

    :param clip:        Source clip
    :param lthr:        See :py:func:`EdgeDetect.get_mask`
    :param hthr:        See :py:func:`EdgeDetect.get_mask`
    :param multi:       See :py:func:`EdgeDetect.get_mask`
    :param clamp:       Clamp to TV or full range if True or specified range `(low, high)`

    :return:            A list edge masks
    """
    def _all_subclasses(cls: type[EdgeDetect] = EdgeDetect) -> set[type[EdgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s for s in _all_subclasses()
        if s.__name__ not in {
            'MatrixEdgeDetect', 'RidgeDetect', 'SingleMatrix', 'EuclidianDistance', 'Max',
            'Matrix1D', 'SavitzkyGolay', 'SavitzkyGolayNormalise',
            'Matrix2x2', 'Matrix3x3', 'Matrix5x5'
        }
    }
    return [
        edge_detect().edgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]


def get_all_ridge_detect(
    clip: vs.VideoNode,
    lthr: float = 0.0, hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False
) -> list[vs.VideoNode]:
    """
    Returns all the RidgeDetect subclasses

    :param clip:        Source clip
    :param lthr:        See :py:func:`EdgeDetect.get_mask`
    :param hthr:        See :py:func:`EdgeDetect.get_mask`
    :param multi:       See :py:func:`EdgeDetect.get_mask`
    :param clamp:       Clamp to TV or full range if True or specified range `(low, high)`

    :return:            A list edge masks
    """
    def _all_subclasses(cls: type[RidgeDetect] = RidgeDetect) -> set[type[RidgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s for s in _all_subclasses()
        if s.__name__ not in {
            'MatrixEdgeDetect', 'RidgeDetect', 'SingleMatrix', 'EuclidianDistance', 'Max',
            'Matrix1D', 'SavitzkyGolay', 'SavitzkyGolayNormalise',
            'Matrix2x2', 'Matrix3x3', 'Matrix5x5'
        }
    }
    return [
        edge_detect().ridgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]


_T = TypeVar('_T')


def _deprecated(msg: str) -> Callable[[type[_T]], type[_T]]:
    class _DeprecatedMeta(ABCMeta):
        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecationWarning)
            return super().__call__(*args, **kwargs)

    def _make_cls(cls: type[_T]) -> type[_T]:
        return cast(type[_T], _DeprecatedMeta(cls.__name__, cls.__bases__, dict(cls.__dict__)))

    return _make_cls
