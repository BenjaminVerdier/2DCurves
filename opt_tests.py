"""
from numpy import pi, sin, cos
import numpy as np
import math
import integral
from gradient_free_optimizers import HillClimbingOptimizer

show_offset_curves = False

theta = np.arange(0, (2 * pi), 0.01)

def radius_comp(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

def optimizable_function(args):
    return abs(240.667 - integral.approximate_integral_length(args['radius'], args['c1'], args['c2']))

search_space = {
    "radius": np.arange(1., 50., 0.1),
    "c1": np.arange(-2., 2., 0.01),
    "c2": np.arange(-2., 2., 0.01),
}

opt = HillClimbingOptimizer(search_space)
opt.search(optimizable_function, n_iter=30000)
"""
import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def parabola_function(para):
    loss = para["x"] * para["x"]
    return -loss


search_space = {"x": np.arange(-10, 10, 0.1)}

opt = RandomSearchOptimizer(search_space)
opt.search(parabola_function, n_iter=100000)
