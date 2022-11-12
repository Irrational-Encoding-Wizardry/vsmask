"""
Edge and ridge detection submodule
"""

# flake8: noqa
from ._1d import *
from ._2x2 import *
from ._3x3 import *
from ._5x5 import *
from ._abstract import *
from ._misc import *

__all__ = [
    # Abstract
    'EdgeDetect', 'MatrixEdgeDetect', 'SingleMatrix', 'EuclidianDistance', 'Max', 'RidgeDetect',

    # 1 dimension
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
    #
    'SavitzkyGolayDeriv3Cub5',
    'SavitzkyGolayDeriv3Cub7',
    'SavitzkyGolayDeriv3Cub9',
    'SavitzkyGolayDeriv3Cub11',
    'SavitzkyGolayDeriv3Cub13',
    'SavitzkyGolayDeriv3Cub15',
    'SavitzkyGolayDeriv3Cub17',
    'SavitzkyGolayDeriv3Cub19',
    'SavitzkyGolayDeriv3Cub21',
    'SavitzkyGolayDeriv3Cub23',
    'SavitzkyGolayDeriv3Cub25',
    #
    'SavitzkyGolayDeriv3Quint7',
    'SavitzkyGolayDeriv3Quint9',
    'SavitzkyGolayDeriv3Quint11',
    'SavitzkyGolayDeriv3Quint13',
    'SavitzkyGolayDeriv3Quint15',
    'SavitzkyGolayDeriv3Quint17',
    'SavitzkyGolayDeriv3Quint19',
    'SavitzkyGolayDeriv3Quint21',
    'SavitzkyGolayDeriv3Quint23',
    'SavitzkyGolayDeriv3Quint25',
    #
    'SavitzkyGolayDeriv4Quart7',
    'SavitzkyGolayDeriv4Quart9',
    'SavitzkyGolayDeriv4Quart11',
    'SavitzkyGolayDeriv4Quart13',
    'SavitzkyGolayDeriv4Quart15',
    'SavitzkyGolayDeriv4Quart17',
    'SavitzkyGolayDeriv4Quart19',
    'SavitzkyGolayDeriv4Quart21',
    'SavitzkyGolayDeriv4Quart23',
    'SavitzkyGolayDeriv4Quart25',
    #
    'SavitzkyGolayDeriv5Quint7',
    'SavitzkyGolayDeriv5Quint9',
    'SavitzkyGolayDeriv5Quint11',
    'SavitzkyGolayDeriv5Quint13',
    'SavitzkyGolayDeriv5Quint15',
    'SavitzkyGolayDeriv5Quint17',
    'SavitzkyGolayDeriv5Quint19',
    'SavitzkyGolayDeriv5Quint21',
    'SavitzkyGolayDeriv5Quint23',
    'SavitzkyGolayDeriv5Quint25',

    # 2x2
    'Matrix2x2',
    'Roberts',

    # 3x3
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
    'MinMax',

    # 5x5
    'Matrix5x5',
    # Single matrix
    'ExLaplacian1', 'ExLaplacian2', 'ExLaplacian3', 'ExLaplacian4',
    'LoG',
    # Euclidian distance
    'ExPrewitt',
    'ExSobel',
    'FDoG', 'FDoGTCanny',
    'DoG',
    'Farid',
    # Max
    'ExKirsch',

    # Misc
    'get_all_edge_detects', 'get_all_ridge_detect'
]
