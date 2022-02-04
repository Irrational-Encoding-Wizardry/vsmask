__all__ = [
    'Matrix1D',
    'TEdge', 'TEdgeTedgemask',
    'SavitzkyGolay',
    'SGD1Q5', 'SavitzkyGolayDerivative1Quad5',
    'SGD1Q7', 'SavitzkyGolayDerivative1Quad7',
    'SGD1Q9', 'SavitzkyGolayDerivative1Quad9',
    'SGD1Q11', 'SavitzkyGolayDerivative1Quad11',
    'SGD1Q13', 'SavitzkyGolayDerivative1Quad13',
    'SGD1Q15', 'SavitzkyGolayDerivative1Quad15',
    'SGD1Q17', 'SavitzkyGolayDerivative1Quad17',
    'SGD1Q19', 'SavitzkyGolayDerivative1Quad19',
    'SGD1Q21', 'SavitzkyGolayDerivative1Quad21',
    'SGD1Q23', 'SavitzkyGolayDerivative1Quad23',
    'SGD1Q25', 'SavitzkyGolayDerivative1Quad25',
    'SGD1C5', 'SavitzkyGolayDerivative1Cubic5',
    'SGD1C7', 'SavitzkyGolayDerivative1Cubic7',
    'SGD1C9', 'SavitzkyGolayDerivative1Cubic9',
    'SGD1C11', 'SavitzkyGolayDerivative1Cubic11',
    'SGD1C13', 'SavitzkyGolayDerivative1Cubic13',
    'SGD1C15', 'SavitzkyGolayDerivative1Cubic15',
    'SGD1C17', 'SavitzkyGolayDerivative1Cubic17',
    'SGD1C19', 'SavitzkyGolayDerivative1Cubic19',
    'SGD1C21', 'SavitzkyGolayDerivative1Cubic21',
    'SGD1C23', 'SavitzkyGolayDerivative1Cubic23',
    'SGD1C25', 'SavitzkyGolayDerivative1Cubic25',
    'SGD1S7', 'SavitzkyGolayDerivative1Sex7',
    'SGD1S9', 'SavitzkyGolayDerivative1Sex9',
    'SGD1S11', 'SavitzkyGolayDerivative1Sex11',
    'SGD1S13', 'SavitzkyGolayDerivative1Sex13',
    'SGD1S15', 'SavitzkyGolayDerivative1Sex15',
    'SGD1S17', 'SavitzkyGolayDerivative1Sex17',
    'SGD1S19', 'SavitzkyGolayDerivative1Sex19',
    'SGD1S21', 'SavitzkyGolayDerivative1Sex21',
    'SGD1S23', 'SavitzkyGolayDerivative1Sex23',
    'SGD1S25', 'SavitzkyGolayDerivative1Sex25',
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


class SavitzkyGolay(EuclidianDistanceMatrixDetect, Matrix1D, ABC):
    ...


class SavitzkyGolayDerivative1Quad5(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 5"""
    matrices = [[-2, -1, 0, 1, 2]] * 2
    divisors = [10] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad7(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 7"""
    matrices = [[-3, -2, -1, 0, 1, 2, 3]] * 2
    divisors = [28] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad9(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 9"""
    matrices = [[-4, -3, -2, -1, 0, 1, 2, 3, 4]] * 2
    divisors = [60] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad11(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 11"""
    matrices = [[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]] * 2
    divisors = [110] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad13(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 13"""
    matrices = [[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]] * 2
    divisors = [182] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad15(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 15"""
    matrices = [[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]] * 2
    divisors = [280] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad17(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 17"""
    matrices = [[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]] * 2
    divisors = [408] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad19(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 19"""
    matrices = [[-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2
    divisors = [570] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad21(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 21"""
    matrices = [[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 2
    divisors = [770] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad23(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 23"""
    matrices = [[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] * 2
    divisors = [1012] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Quad25(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 25"""
    matrices = [[-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] * 2
    divisors = [1300] * 2
    mode_types = ['h', 'v']


SGD1Q5 = SavitzkyGolayDerivative1Quad5
SGD1Q7 = SavitzkyGolayDerivative1Quad7
SGD1Q9 = SavitzkyGolayDerivative1Quad9
SGD1Q11 = SavitzkyGolayDerivative1Quad11
SGD1Q13 = SavitzkyGolayDerivative1Quad13
SGD1Q15 = SavitzkyGolayDerivative1Quad15
SGD1Q17 = SavitzkyGolayDerivative1Quad17
SGD1Q19 = SavitzkyGolayDerivative1Quad19
SGD1Q21 = SavitzkyGolayDerivative1Quad21
SGD1Q23 = SavitzkyGolayDerivative1Quad23
SGD1Q25 = SavitzkyGolayDerivative1Quad25


class SavitzkyGolayDerivative1Cubic5(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 5"""
    matrices = [[1, -8, 0, 8, -1]] * 2
    divisors = [12] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic7(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic derivative operator of size 7"""
    matrices = [[22, -67, -58, 0, 58, 67, -22]] * 2
    divisors = [252] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic9(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 9"""
    matrices = [[86, -142, -193, -126, 0, 126, 193, 142, -86]] * 2
    divisors = [1188] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic11(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 11"""
    matrices = [[300, -294, -532, -503, -296, 0, 296, 503, 532, 294, -300]] * 2
    divisors = [5148] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic13(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 13"""
    matrices = [[1133, -660, -1578, -1796, -1489, -832, 0, 832, 1489, 1796, 1578, 660, -1133]] * 2
    divisors = [24024] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic15(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 15"""
    matrices = [[12922, -4121, -14150, -18334, -17842, -13843, -7506,
                 0,
                 7506, 13843, 17842, 18334, 14150, 4121, -12922]] * 2
    divisors = [334152] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic17(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 17"""
    matrices = [[748, -98, -643, -930, -1002, -902, -673, -358, 0, 358, 673, 902, 1002, 930, 643, 98, -748]] * 2
    divisors = [23256] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic19(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 19"""
    matrices = [[6936, 68, -4648, -7481, -8700, -8574, -7372, -5363, -2816,
                 0,
                 2816, 5363, 7372, 8574, 8700, 7481, 4648, -68, -6936]] * 2
    divisors = [255816] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic21(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 21"""
    matrices = [[84075, 10032, -43284, -78176, -96947, -101900, -95338, -79564, -56881, -29592,
                 0,
                 29592, 56881, 79564, 95338, 101900, 96947, 78176, 43284, -10032, -84075]] * 2
    divisors = [3634092] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic23(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 23"""
    matrices = [[3938, 815, -1518, -3140, -4130, -4567, -4530, -4098, -3350, -2365, -1222,
                 0,
                 1222, 2365, 3350, 4098, 4530, 4567, 4130, 3140, 1518, -815, -3938]] * 2
    divisors = [197340] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Cubic25(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 25"""
    matrices = [[30866, 8602, -8525, -20982, -29236, -33754, -35003, -33450, -29562, -23806, -16649, -8558,
                 0,
                 8558, 16649, 23806, 29562, 33450, 35003, 33754, 29236, 20982, 8525, -8602, -30866]] * 2
    divisors = [1776060] * 2
    mode_types = ['h', 'v']


SGD1C5 = SavitzkyGolayDerivative1Cubic5
SGD1C7 = SavitzkyGolayDerivative1Cubic7
SGD1C9 = SavitzkyGolayDerivative1Cubic9
SGD1C11 = SavitzkyGolayDerivative1Cubic11
SGD1C13 = SavitzkyGolayDerivative1Cubic13
SGD1C15 = SavitzkyGolayDerivative1Cubic15
SGD1C17 = SavitzkyGolayDerivative1Cubic17
SGD1C19 = SavitzkyGolayDerivative1Cubic19
SGD1C21 = SavitzkyGolayDerivative1Cubic21
SGD1C23 = SavitzkyGolayDerivative1Cubic23
SGD1C25 = SavitzkyGolayDerivative1Cubic25


class SavitzkyGolayDerivative1Sex7(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 7"""
    matrices = [[-1, 9, -45, 0, 45, -9, 1]] * 2
    divisors = [60] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex9(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 9"""
    matrices = [[-254, 1381, -2269, -2879, 0, 2879, 2269, -1381, 254]] * 2
    divisors = [8580] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex11(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 11"""
    matrices = [[-573, 2166, -1249, -3774, -3084, 0, 3084, 3774, 1249, -2166, 573]] * 2
    divisors = [17160] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex13(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 13"""
    matrices = [[-9647, 27093, -12, -33511, -45741, -31380, 0, 31380, 45741, 33511, 12, -27093, 9647]] * 2
    divisors = [291720] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex15(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 15"""
    matrices = [[-78351, 169819, 65229, -130506, -266401, -279975, -175125,
                 0,
                 175125, 279975, 266401, 130506, -65229, -169819, 78351]] * 2
    divisors = [2519400] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex17(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 17"""
    matrices = [[-14404, 24661, 16679, -8671, -32306, -43973, -40483, -23945,
                 0,
                 23945, 40483, 43973, 32306, 8671, -16679, -24661, 14404]] * 2
    divisors = [503880] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex19(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 19"""
    matrices = [[-255102, 349928, 322378, 9473, -348823, -604484, -686099, -583549, -332684,
                 0,
                 332684, 583549, 686099, 604484, 348823, -9473, -322378, -349928, 255102]] * 2
    divisors = [9806280] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex21(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 21"""
    matrices = [[
        -15033066, 16649358, 19052988, 6402438, -10949942, -26040033, -34807914, -35613829, -28754154, -15977364,
        0,
        15977364, 28754154, 35613829, 34807914, 26040033, 10949942, -6402438, -19052988, -16649358, 15033066
    ]] * 2
    divisors = [637408200] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex23(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 23"""
    matrices = [[
        -400653, 359157, 489687, 265164, -106911, -478349, -752859, -878634, -840937, -654687, -357045,
        0,
        357045, 654687, 840937, 878634, 752859, 478349, 106911, -265164, -489687, -359157, 400653
    ]] * 2
    divisors = [18747300] * 2
    mode_types = ['h', 'v']


class SavitzkyGolayDerivative1Sex25(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 25"""
    matrices = [[
        -8322182, 6024183, 9604353, 6671883, 544668, -6301491, -12139321, -15896511, -17062146, -15593141, -11820675, -6356625,
        0,
        6356625, 11820675, 15593141, 17062146, 15896511, 12139321, 6301491, -544668, -6671883, -9604353, -6024183, 8322182
    ]] * 2
    divisors = [429214500] * 2
    mode_types = ['h', 'v']


SGD1S7 = SavitzkyGolayDerivative1Sex7
SGD1S9 = SavitzkyGolayDerivative1Sex9
SGD1S11 = SavitzkyGolayDerivative1Sex11
SGD1S13 = SavitzkyGolayDerivative1Sex13
SGD1S15 = SavitzkyGolayDerivative1Sex15
SGD1S17 = SavitzkyGolayDerivative1Sex17
SGD1S19 = SavitzkyGolayDerivative1Sex19
SGD1S21 = SavitzkyGolayDerivative1Sex21
SGD1S23 = SavitzkyGolayDerivative1Sex23
SGD1S25 = SavitzkyGolayDerivative1Sex25
