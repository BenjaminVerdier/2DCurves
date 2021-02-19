from numpy import pi, sin, cos
import numpy as np
import math
import linetostl
from functools import partial
from scipy.optimize import minimize
import integral
import curves_overlap
import argparse
import matplotlib.pyplot as plt

theta = np.arange(0, (2 * pi), 0.01)

def check_for_neg_rad(radii):
    if np.amin(radii) < 0.01:
        return True
    return False

def check_overlap(theta, radii, W):
    pts = np.array(linetostl.polarToCart(theta, radii))
    return curves_overlap.check_for_intersection(pts, W, False)

def sanity_check(theta, radii, W):
    problem = False
    if check_for_neg_rad(radii):
        problem = True
    if check_overlap(theta, radii, W):
        problem = True
    return not problem

def radius_comp(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

def scan_for_fixed_radius(radius, min_c1, max_c1, min_c2, max_c2, resolution, W):
    c1s = np.arange(min_c1, max_c1 + resolution, resolution)
    c2s = np.arange(min_c2, max_c2 + resolution, resolution)

    valid_c1s = []
    valid_c2s = []
    invalid_c1s = []
    invalid_c2s = []

    for c1 in c1s:
        print("c1=",c1)
        r = []
        for c2 in c2s:
            r = radius_comp(radius, c1, c2)
            if sanity_check(theta, r, W):
                valid_c1s.append(c1)
                valid_c2s.append(c2)
            else:
                invalid_c1s.append(c1)
                invalid_c2s.append(c2)
    plt.scatter(valid_c1s,valid_c2s, c='blue')
    plt.scatter(invalid_c1s,invalid_c2s, c='red')
    plt.savefig('R_' + '{:.2f}'.format(radius) + '_printability.png')

for radius in np.arange(1,6.1,.1):
    print("radius=",radius)
    scan_for_fixed_radius(radius, -.5, .5, -.5, .5, .5, .2)
