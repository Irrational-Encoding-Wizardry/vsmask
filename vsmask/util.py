
from functools import partial
from typing import Any, Callable, Sequence, Union

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
