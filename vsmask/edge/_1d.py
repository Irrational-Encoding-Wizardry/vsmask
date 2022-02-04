__all__ = [
    'Matrix1D',
    'TEdge', 'TEdgeTedgemask',
    #
    'SavitzkyGolay',
    #
    'SavitzkyGolayDeriv1Quad5',
    'SavitzkyGolayDeriv1Quad7',
    'SavitzkyGolayDeriv1Quad9',
    'SavitzkyGolayDeriv1Quad11',
    'SavitzkyGolayDeriv1Quad13',
    'SavitzkyGolayDeriv1Quad15',
    'SavitzkyGolayDeriv1Quad17',
    'SavitzkyGolayDeriv1Quad19',
    'SavitzkyGolayDeriv1Quad21',
    'SavitzkyGolayDeriv1Quad23',
    'SavitzkyGolayDeriv1Quad25',
    #
    'SavitzkyGolayDeriv1Cubic5',
    'SavitzkyGolayDeriv1Cubic7',
    'SavitzkyGolayDeriv1Cubic9',
    'SavitzkyGolayDeriv1Cubic11',
    'SavitzkyGolayDeriv1Cubic13',
    'SavitzkyGolayDeriv1Cubic15',
    'SavitzkyGolayDeriv1Cubic17',
    'SavitzkyGolayDeriv1Cubic19',
    'SavitzkyGolayDeriv1Cubic21',
    'SavitzkyGolayDeriv1Cubic23',
    'SavitzkyGolayDeriv1Cubic25',
    #
    'SavitzkyGolayDeriv1Quint7',
    'SavitzkyGolayDeriv1Quint9',
    'SavitzkyGolayDeriv1Quint11',
    'SavitzkyGolayDeriv1Quint13',
    'SavitzkyGolayDeriv1Quint15',
    'SavitzkyGolayDeriv1Quint17',
    'SavitzkyGolayDeriv1Quint19',
    'SavitzkyGolayDeriv1Quint21',
    'SavitzkyGolayDeriv1Quint23',
    'SavitzkyGolayDeriv1Quint25',
    #
    'SavitzkyGolayDeriv2Quad5',
    'SavitzkyGolayDeriv2Quad7',
    'SavitzkyGolayDeriv2Quad9',
    'SavitzkyGolayDeriv2Quad11',
    'SavitzkyGolayDeriv2Quad13',
    'SavitzkyGolayDeriv2Quad15',
    'SavitzkyGolayDeriv2Quad17',
    'SavitzkyGolayDeriv2Quad19',
    'SavitzkyGolayDeriv2Quad21',
    'SavitzkyGolayDeriv2Quad23',
    'SavitzkyGolayDeriv2Quad25',
    'SavitzkyGolayDeriv2Quart7',
    'SavitzkyGolayDeriv2Quart9',
    'SavitzkyGolayDeriv2Quart11',
    'SavitzkyGolayDeriv2Quart13',
    'SavitzkyGolayDeriv2Quart15',
    'SavitzkyGolayDeriv2Quart17',
    'SavitzkyGolayDeriv2Quart19',
    'SavitzkyGolayDeriv2Quart21',
    'SavitzkyGolayDeriv2Quart23',
    'SavitzkyGolayDeriv2Quart25',
]

from abc import ABC

import vapoursynth as vs

from ._abstract import EdgeDetect, EuclidianDistance


class Matrix1D(EdgeDetect, ABC):
    ...


class TEdge(EuclidianDistance, Matrix1D):
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


class SavitzkyGolay(EuclidianDistance, Matrix1D):
    mode_types = ['h', 'v']


class SavitzkyGolayDeriv1Quad5(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 5"""
    matrices = [[-2, -1, 0, 1, 2]] * 2
    divisors = [10] * 2


class SavitzkyGolayDeriv1Quad7(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 7"""
    matrices = [[-3, -2, -1, 0, 1, 2, 3]] * 2
    divisors = [28] * 2


class SavitzkyGolayDeriv1Quad9(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 9"""
    matrices = [[-4, -3, -2, -1, 0, 1, 2, 3, 4]] * 2
    divisors = [60] * 2


class SavitzkyGolayDeriv1Quad11(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 11"""
    matrices = [[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]] * 2
    divisors = [110] * 2


class SavitzkyGolayDeriv1Quad13(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 13"""
    matrices = [[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]] * 2
    divisors = [182] * 2


class SavitzkyGolayDeriv1Quad15(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 15"""
    matrices = [[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]] * 2
    divisors = [280] * 2


class SavitzkyGolayDeriv1Quad17(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 17"""
    matrices = [[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]] * 2
    divisors = [408] * 2


class SavitzkyGolayDeriv1Quad19(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 19"""
    matrices = [[-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2
    divisors = [570] * 2


class SavitzkyGolayDeriv1Quad21(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 21"""
    matrices = [[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 2
    divisors = [770] * 2


class SavitzkyGolayDeriv1Quad23(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 23"""
    matrices = [[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] * 2
    divisors = [1012] * 2


class SavitzkyGolayDeriv1Quad25(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 25"""
    matrices = [[-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] * 2
    divisors = [1300] * 2


class SavitzkyGolayDeriv1Cubic5(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 5"""
    matrices = [[1, -8, 0, 8, -1]] * 2
    divisors = [12] * 2


class SavitzkyGolayDeriv1Cubic7(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic derivative operator of size 7"""
    matrices = [[22, -67, -58, 0, 58, 67, -22]] * 2
    divisors = [252] * 2


class SavitzkyGolayDeriv1Cubic9(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 9"""
    matrices = [[86, -142, -193, -126, 0, 126, 193, 142, -86]] * 2
    divisors = [1188] * 2


class SavitzkyGolayDeriv1Cubic11(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 11"""
    matrices = [[300, -294, -532, -503, -296, 0, 296, 503, 532, 294, -300]] * 2
    divisors = [5148] * 2


class SavitzkyGolayDeriv1Cubic13(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 13"""
    matrices = [[1133, -660, -1578, -1796, -1489, -832, 0, 832, 1489, 1796, 1578, 660, -1133]] * 2
    divisors = [24024] * 2


class SavitzkyGolayDeriv1Cubic15(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 15"""
    matrices = [[12922, -4121, -14150, -18334, -17842, -13843, -7506,
                 0,
                 7506, 13843, 17842, 18334, 14150, 4121, -12922]] * 2
    divisors = [334152] * 2


class SavitzkyGolayDeriv1Cubic17(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 17"""
    matrices = [[748, -98, -643, -930, -1002, -902, -673, -358, 0, 358, 673, 902, 1002, 930, 643, 98, -748]] * 2
    divisors = [23256] * 2


class SavitzkyGolayDeriv1Cubic19(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 19"""
    matrices = [[6936, 68, -4648, -7481, -8700, -8574, -7372, -5363, -2816,
                 0,
                 2816, 5363, 7372, 8574, 8700, 7481, 4648, -68, -6936]] * 2
    divisors = [255816] * 2


class SavitzkyGolayDeriv1Cubic21(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 21"""
    matrices = [[84075, 10032, -43284, -78176, -96947, -101900, -95338, -79564, -56881, -29592,
                 0,
                 29592, 56881, 79564, 95338, 101900, 96947, 78176, 43284, -10032, -84075]] * 2
    divisors = [3634092] * 2


class SavitzkyGolayDeriv1Cubic23(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 23"""
    matrices = [[3938, 815, -1518, -3140, -4130, -4567, -4530, -4098, -3350, -2365, -1222,
                 0,
                 1222, 2365, 3350, 4098, 4530, 4567, 4130, 3140, 1518, -815, -3938]] * 2
    divisors = [197340] * 2


class SavitzkyGolayDeriv1Cubic25(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 25"""
    matrices = [[30866, 8602, -8525, -20982, -29236, -33754, -35003, -33450, -29562, -23806, -16649, -8558,
                 0,
                 8558, 16649, 23806, 29562, 33450, 35003, 33754, 29236, 20982, 8525, -8602, -30866]] * 2
    divisors = [1776060] * 2


class SavitzkyGolayDeriv1Quint7(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 7"""
    matrices = [[-1, 9, -45, 0, 45, -9, 1]] * 2
    divisors = [60] * 2


class SavitzkyGolayDeriv1Quint9(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 9"""
    matrices = [[-254, 1381, -2269, -2879, 0, 2879, 2269, -1381, 254]] * 2
    divisors = [8580] * 2


class SavitzkyGolayDeriv1Quint11(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 11"""
    matrices = [[-573, 2166, -1249, -3774, -3084, 0, 3084, 3774, 1249, -2166, 573]] * 2
    divisors = [17160] * 2


class SavitzkyGolayDeriv1Quint13(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 13"""
    matrices = [[-9647, 27093, -12, -33511, -45741, -31380, 0, 31380, 45741, 33511, 12, -27093, 9647]] * 2
    divisors = [291720] * 2


class SavitzkyGolayDeriv1Quint15(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 15"""
    matrices = [[-78351, 169819, 65229, -130506, -266401, -279975, -175125,
                 0,
                 175125, 279975, 266401, 130506, -65229, -169819, 78351]] * 2
    divisors = [2519400] * 2


class SavitzkyGolayDeriv1Quint17(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 17"""
    matrices = [[-14404, 24661, 16679, -8671, -32306, -43973, -40483, -23945,
                 0,
                 23945, 40483, 43973, 32306, 8671, -16679, -24661, 14404]] * 2
    divisors = [503880] * 2


class SavitzkyGolayDeriv1Quint19(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 19"""
    matrices = [[-255102, 349928, 322378, 9473, -348823, -604484, -686099, -583549, -332684,
                 0,
                 332684, 583549, 686099, 604484, 348823, -9473, -322378, -349928, 255102]] * 2
    divisors = [9806280] * 2


class SavitzkyGolayDeriv1Quint21(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 21"""
    matrices = [[
        -15033066, 16649358, 19052988, 6402438, -10949942, -26040033, -34807914, -35613829, -28754154, -15977364,
        0,
        15977364, 28754154, 35613829, 34807914, 26040033, 10949942, -6402438, -19052988, -16649358, 15033066
    ]] * 2
    divisors = [637408200] * 2


class SavitzkyGolayDeriv1Quint23(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 23"""
    matrices = [[
        -400653, 359157, 489687, 265164, -106911, -478349, -752859, -878634, -840937, -654687, -357045,
        0,
        357045, 654687, 840937, 878634, 752859, 478349, 106911, -265164, -489687, -359157, 400653
    ]] * 2
    divisors = [18747300] * 2


class SavitzkyGolayDeriv1Quint25(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 25"""
    matrices = [[
        -8322182, 6024183, 9604353, 6671883, 544668, -6301491, -12139321, -15896511, -17062146, -15593141, -11820675, -6356625,
        0,
        6356625, 11820675, 15593141, 17062146, 15896511, 12139321, 6301491, -544668, -6671883, -9604353, -6024183, 8322182
    ]] * 2
    divisors = [429214500] * 2


class SavitzkyGolayDeriv2Quad5(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 5"""
    matrices = [[2, -1, -2, -1, 2]] * 2
    divisors = [7] * 2


class SavitzkyGolayDeriv2Quad7(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 7"""
    matrices = [[5, 0, -3, -4, -3, 0, 5]] * 2
    divisors = [42] * 2


class SavitzkyGolayDeriv2Quad9(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 9"""
    matrices = [[28, 7, -8, -17, -20, -17, -8, 7, 28]] * 2
    divisors = [462] * 2


class SavitzkyGolayDeriv2Quad11(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 11"""
    matrices = [[15, 6, -1, -6, -9, -10, -9, -6, -1, 6, 15]] * 2
    divisors = [429] * 2


class SavitzkyGolayDeriv2Quad13(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 13"""
    matrices = [[22, 11, 2, -5, -10, -13, -14, -13, -10, -5, 2, 11, 22]] * 2
    divisors = [1001] * 2


class SavitzkyGolayDeriv2Quad15(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 15"""
    matrices = [[91, 52, 19, -8, -29, -44, -53, -56, -53, -44, -29, -8, 19, 52, 91]] * 2
    divisors = [6188] * 2


class SavitzkyGolayDeriv2Quad17(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 17"""
    matrices = [[40, 25, 12, 1, -8, -15, -20, -23, -24, -23, -20, -15, -8, 1, 12, 25, 40]] * 2
    divisors = [3876] * 2


class SavitzkyGolayDeriv2Quad19(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 19"""
    matrices = [[51, 34, 19, 6, -5, -14, -21, -26, -29, -30, -29, -26, -21, -14, -5, 6, 19, 34, 51]] * 2
    divisors = [6783] * 2


class SavitzkyGolayDeriv2Quad21(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 21"""
    matrices = [[190, 133, 82, 37, -2, -35, -62, -83, -98, -107, -110, -107, -98, -83, -62, -35, -2, 37, 82, 133, 190]] * 2
    divisors = [33649] * 2


class SavitzkyGolayDeriv2Quad23(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 23"""
    matrices = [[77, 56, 37, 20, 5, -8, -19, -28, -35, -40, -43, -44, -43, -40, -35, -28, -19, -8, 5, 20, 37, 56, 77]] * 2
    divisors = [17710] * 2


class SavitzkyGolayDeriv2Quad25(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 25"""
    matrices = [[92, 69, 48, 29, 12, -3, -16, -27, -36, -43, -48, -51, -52, -51, -48, -43, -36, -27, -16, -3, 12, 29, 48, 69, 92]] * 2
    divisors = [26910] * 2


class SavitzkyGolayDeriv2Quart7(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 7"""
    matrices = [[-13, 67, -19, -70, -19, 67, -13]] * 2
    divisors = [132] * 2


class SavitzkyGolayDeriv2Quart9(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 9"""
    matrices = [[-126, 371, 151, -211, -370, -211, 151, 371, -126]] * 2
    divisors = [1716] * 2


class SavitzkyGolayDeriv2Quart11(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 11"""
    matrices = [[-90, 174, 146, 1, -136, -190, -136, 1, 146, 174, -90]] * 2
    divisors = [1716] * 2


class SavitzkyGolayDeriv2Quart13(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 13"""
    matrices = [[-2211, 2970, 3504, 1614, -971, -3016, -3780, -3016, -971, 1614, 3504, 2970, -2211]] * 2
    divisors = [58344] * 2


class SavitzkyGolayDeriv2Quart15(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 15"""
    matrices = [[-31031, 29601, 44495, 31856, 6579, -19751, -38859, -45780, -38859, -19751, 6579, 31856, 44495, 29601, -31031]] * 2
    divisors = [1108536] * 2


class SavitzkyGolayDeriv2Quart17(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 17"""
    matrices = [[-2132, 1443, 2691, 2405, 1256, -207, -1557, -2489, -2820, -2489, -1557, -207, 1256, 2405, 2691, 1443, -2132]] * 2
    divisors = [100776] * 2


class SavitzkyGolayDeriv2Quart19(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 19"""
    matrices = [[-32028, 15028, 35148, 36357, 25610, 8792, -9282, -24867, -35288, -38940, -35288, -24867, -9282, 8792, 25610, 36357, 35148, 15028, -32028]] * 2
    divisors = [1961256] * 2


class SavitzkyGolayDeriv2Quart21(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 21"""
    matrices = [[-12597, 3876, 11934, 13804, 11451, 6578, 626, -5226, -10061, -13224, -14322, -13224, -10061, -5226, 626, 6578, 11451, 13804, 11934, 3876, -12597]] * 2
    divisors = [980628] * 2


class SavitzkyGolayDeriv2Quart23(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 23"""
    matrices = [[-115577, 20615, 93993, 119510, 110545, 78903, 34815, -13062, -57645, -93425, -116467, -124410, -116467, -93425, -57645, -13062, 34815, 78903, 110545, 119510, 93993, 20615, -115577]] * 2
    divisors = [11248380] * 2


class SavitzkyGolayDeriv2Quart25(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivativeoperator of size 25"""
    matrices = [[-143198, 10373, 99385, 137803, 138262, 112067, 69193, 18285, -33342, -79703, -116143, -139337, -147290, -139337, -116143, -79703, -33342, 18285, 69193, 112067, 138262, 137803, 99385, 10373, -143198]] * 2
    divisors = [17168580] * 2
