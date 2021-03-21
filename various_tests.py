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
import csv
import random

theta = np.arange(0, (2 * pi), 0.01)

def check_for_neg_rad(radii):
    if np.amin(radii) < 0.01:
        return True
    return False

def check_overlap(pts, W):
    return curves_overlap.check_for_intersection(np.array(pts), W, False)

def checkBB(radii, BB):
    if np.amax(radii) > BB:
        return True
    return False

def sanity_check(theta, radii, W, BB):
    problem = False
    #pts = linetostl.polarToCart(theta, radii)
    if checkBB(radii, BB):
        problem = True
    if not problem and check_for_neg_rad(radii):
        problem = True
    #if not problem and check_overlap(pts, W):
        #problem = True
    return not problem

def radius_comp(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

def scan_for_fixed_radius(radius, min_c1, max_c1, min_c2, max_c2, resolution, W, BB):
    c1s = np.arange(min_c1, max_c1 + resolution, resolution)
    c2s = np.arange(min_c2, max_c2 + resolution, resolution)

    valid_c1s = []
    valid_c2s = []
    invalid_c1s = []
    invalid_c2s = []

    for c1 in c1s:
        r = []
        for c2 in c2s:
            r = radius_comp(radius, c1, c2)
            if sanity_check(theta, r, W, BB):
                valid_c1s.append(c1)
                valid_c2s.append(c2)
            else:
                invalid_c1s.append(c1)
                invalid_c2s.append(c2)
    valid = plt.scatter(valid_c1s,valid_c2s, c='blue', label='Printable')
    invalid = plt.scatter(invalid_c1s,invalid_c2s, c='red', label='Not Printable')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("c1")
    plt.ylabel("c2")
    plt.savefig('R_' + '{:.2f}'.format(radius) + '_printability.png')
    plt.clf()


"""
# Sweeping radii for printability
for radius in np.arange(1,6,.5):
    print("radius=",radius)
    scan_for_fixed_radius(radius, -.5, .5, -.5, .5, .01, .6, 19)
"""

def optimizable_function(target, c1, c2, param):
    return abs(target - integral.approximate_integral_length(param[0], c1, c2))

def scan_for_optimal_radius_and_printability(target, min_c1, max_c1, min_c2, max_c2, resolution, W, BB):
    c1s = np.arange(min_c1, max_c1 + resolution, resolution)
    c2s = np.arange(min_c2, max_c2 + resolution, resolution)

    valid_c1s = []
    valid_c2s = []
    corresponding_radii = []
    invalid_c1s = []
    invalid_c2s = []

    r = []
    for c1 in c1s:
        for c2 in c2s:
            func_to_opt = partial(optimizable_function, target, c1, c2)
            x0=[38]
            res = minimize(func_to_opt, np.array(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
            radius = res.x[0]
            r = radius_comp(radius, c1, c2)
            if sanity_check(theta, r, W, BB):
                valid_c1s.append(c1)
                valid_c2s.append(c2)
                corresponding_radii.append(radius)
            else:
                invalid_c1s.append(c1)
                invalid_c2s.append(c2)
    valid = plt.scatter(valid_c1s,valid_c2s, c='blue', label='Printable')
    invalid = plt.scatter(invalid_c1s,invalid_c2s, c='red', label='Not Printable')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("c1")
    plt.ylabel("c2")
    plt.savefig('optimized_printability_round_target_' + '{:.2f}'.format(target)  + '.png')
    plt.clf()
    with open('c1_c2_r_correspondance_table_round_target' + '{:.2f}'.format(target)  + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['c1','c2','radius'])
        for row in zip(valid_c1s,valid_c2s,corresponding_radii):
            writer.writerow(row)


# Scanning c1-c2 space for optimal radii and checking printability
#scan_for_optimal_radius_and_printability(240.667, -1, 1, -1, 1, .02, .6, 19)

def interpolation(r_1, c1_1, c2_1, r_2, c1_2, c2_2, steps):
    tuples = np.linspace((r_1, c1_1, c2_1), (r_2, c1_2, c2_2), steps)
    lengths = [0] * steps
    for i in range(steps):
        lengths[i] = integral.approximate_integral_length(*tuples[i])
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((3, 3), (2, 0))
    ax3 = plt.subplot2grid((3, 3), (2, 1))
    ax4 = plt.subplot2grid((3, 3), (2, 2))
    ax1.plot(lengths)
    ax1.set_title('Length')
    ax2.plot([r_1,r_2])
    ax2.set_ylim(0,11)
    ax2.set_xlim(-0.1,1.1)
    ax2.set_xlabel("R")
    ax3.plot([c1_1,c1_2])
    ax3.set_ylim(-0.6,0.6)
    ax3.set_xlim(-0.1,1.1)
    ax3.set_xlabel('c1')
    ax4.plot([c2_1,c2_2])
    ax4.set_ylim(-0.6,0.6)
    ax4.set_xlim(-0.1,1.1)
    ax4.set_xlabel('c2')
    plt.savefig('length_for_interpolation_r_{:.2f}'.format(r_1) + '_{:.2f}'.format(r_2) + '_c1_{:.2f}'.format(c1_1) + '_{:.2f}'.format(c1_2) + '_c2_{:.2f}'.format(c2_1) + '_{:.2f}'.format(c2_2) + '.png')
    plt.clf()

"""
# Length variation for a given interpolation
for _ in range(10):
    r_1 = random.uniform(1,10)
    r_2 = random.uniform(1,10)
    c1_1 = random.uniform(-0.5,0.5)
    c1_2 = random.uniform(-0.5,0.5)
    c2_1 = random.uniform(-0.5,0.5)
    c2_2 = random.uniform(-0.5,0.5)
    interpolation(r_1, c1_1, c2_1, r_2, c1_2, c2_2, 100)
"""

interpolation(7.509071223, 0.02, -0.96, 11.8772307, -0.16, -0.58, 100)
