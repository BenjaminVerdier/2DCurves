from numpy import pi, sin, cos
import numpy as np
import math
import linetostl
from functools import partial
from scipy.optimize import minimize
import integral
import curves_overlap
import argparse

show_offset_curves = False

theta = np.arange(0, (2 * pi), 0.01)

def check_for_neg_rad(radii):
    if np.amin(radii) < 0.01:
        return True
    return False

def checkBB(theta, radii, BB):
    pts = linetostl.polarToCart(theta, radii)
    if np.amax(pts) > BB or np.amin(pts) < -BB:
        return True
    return False

def check_overlap(theta, radii, W):
    pts = np.array(linetostl.polarToCart(theta, radii))
    return curves_overlap.check_for_intersection(pts, W, show_offset_curves)

def sanity_check(theta, radii, BB, W):
    problem = False
    if checkBB(theta, radii, BB):
        print("Curve exceeds bounding box!")
        problem = True
    if check_for_neg_rad(radii):
        print("Curve has negative radius!")
        problem = True
    if check_overlap(theta, radii, W):
        print("Curve self-overlaps!")
        problem = True
    if not problem:
        print("The curve is valid.")
        return True
    else:
        return False

def radius_comp(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

def optimizable_function(target, c1, c2, param):
    return abs(target - integral.approximate_integral_length(param[0], c1, c2))

def getStlName(r, c1, c2):
    return 'r_' + "{:.2f}".format(r) + '_c1_' + "{:.2f}".format(c1) + '_c2_' + "{:.2f}".format(c2) + '.stl'


def main(target, c1, c2, N, H, name, do_sanity_check, BB, W):
    global theta
    theta = np.arange(0, (2 * pi), N)
    func_to_opt = partial(optimizable_function, target, c1, c2)
    x0=[1]
    res = minimize(func_to_opt, np.array(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    radius = res.x[0]
    print(radius)
    radii = radius_comp(radius, c1, c2)
    compute_stl = True
    if do_sanity_check:
        compute_stl = sanity_check(theta, radii, BB, W)
    if compute_stl:
        if len(name) == 0:
            name = getStlName(radius, c1, c2)
        linetostl.lineToSTL(curves_overlap.offset_curve(np.array(linetostl.polarToCart(theta, radii)), -W), name, H)

if __name__ == '__main__':
    #default values
    target = 6.28
    c1 = 0
    c2 = 0
    N = 0.01
    H = 1
    name = ''
    do_sanity_check = True
    BB = 5
    W = 0.1
    parser=argparse.ArgumentParser()
    parser.add_argument('--target', type=float, help='Target length, default=6.28')
    parser.add_argument('--c1', type=float, help='Value of c1, default=0')
    parser.add_argument('--c2', type=float, help='Value of c2, default=0')
    parser.add_argument('--N', type=float, help='Definition of the curve, default=0.01')
    parser.add_argument('--H', type=float, help='Height of the stl, default=1')
    parser.add_argument('--name', help='Name of the stl file, default="r_xxx_c1_yyy_c2_zzz.stl"')
    parser.add_argument('-s', '--skip_check', action="store_true", help='Wether the program should skip the curve validity check.')
    parser.add_argument('--BB', type=float, help='Bounding box size, default = 5')
    parser.add_argument('--W', type=float, help='Width for sanity check, default = 0.1')
    parser.add_argument('--show_offset', action="store_true", help='Wether the program should skip the curve validity check.')
    args=parser.parse_args()
    if args.target:
        target = args.target
    if args.c1:
        c1 = args.c1
    if args.c2:
        c2 = args.c2
    if args.N:
        N = args.N
    if args.H:
        H = args.H
    if args.name:
        name = args.name
    if args.skip_check:
        do_sanity_check = False
    if args.BB:
        BB = args.BB
    if args.W:
        W = args.W
    if args.show_offset:
        show_offset_curves = True
    main(target, c1, c2, N, H, name, do_sanity_check, BB, W)
