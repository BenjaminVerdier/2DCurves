from numpy import pi, sin, cos
import numpy as np
import mapbox_earcut as earcut
from stl import mesh
from linetostl import polarToCart
import argparse
import integral
from functools import partial
from scipy.optimize import minimize

theta = np.arange(0, (2 * pi), 0.01)

def radius_comp(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

def optimizable_function(target, c1, c2, param):
    return abs(target - integral.approximate_integral_length(param[0], c1, c2))

def interpolate_stl(target, c1_1, c2_1, c1_2, c2_2, z, steps, name):
    c1s = np.linspace(c1_1, c1_2, steps)
    c2s = np.linspace(c2_1, c2_2, steps)

    pts_3d = [[]]*steps
    height_step = z / (steps-1)
    tuples = [()]*steps

    for i in range(steps):
        func_to_opt = partial(optimizable_function, target, c1s[i], c2s[i])
        x0=[38]
        res = minimize(func_to_opt, np.array(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
        tuples[i] = (res.x[0], c1s[i], c2s[i])
        temp_polar = radius_comp(*tuples[i])
        temp_cart = polarToCart(theta, temp_polar)
        pts_3d[i] =  [ [ax,ay,i*height_step] for (ax,ay) in temp_cart]


    pts_base = polarToCart(theta, radius_comp(*tuples[0]))
    pts_top = polarToCart(theta, radius_comp(*tuples[steps-1]))

    triangles_indices_base = earcut.triangulate_float32(np.array(pts_base).reshape(-1,2), np.array([len(pts_base)])).reshape(-1,3)
    triangles_indices_top = earcut.triangulate_float32(np.array(pts_top).reshape(-1,2), np.array([len(pts_top)])).reshape(-1,3)

    base = [ [pts_3d[0][a], pts_3d[0][b], pts_3d[0][c]] for (a,b,c) in triangles_indices_base]
    top = [ [pts_3d[steps-1][a], pts_3d[steps-1][b], pts_3d[steps-1][c]] for (a,b,c) in triangles_indices_top]

    n_facets = 2 * len(theta) * (steps-1) + len(base) + len(top)

    data = np.zeros(n_facets, dtype=mesh.Mesh.dtype)

    for i in range(len(base)):
        data['vectors'][i] = np.array([base[i][0], base[i][1], base[i][2]])

    offset = len(base)

    for i in range(len(top)):
        data['vectors'][offset + i] = np.array([top[i][0], top[i][1], top[i][2]])

    offset += len(top)

    for i in range(steps-1):
        for j in range(len(theta)):
            data['vectors'][offset + 2*j] = np.array([pts_3d[i][j], pts_3d[i+1][j-1], pts_3d[i][j-1]])
            data['vectors'][offset + 2*j + 1] = np.array([pts_3d[i][j], pts_3d[i+1][j], pts_3d[i+1][j-1]])
        offset += 2 * len(theta)

    new_mesh = mesh.Mesh(data)

    new_mesh.save(name)

if __name__ == '__main__':
    #default values
    c1_1 = 0
    c2_1 = 0
    c1_2 = 0
    c2_2 = 0
    target = 240.667
    N = 100
    H = 1
    name = ''
    parser=argparse.ArgumentParser()
    parser.add_argument('--c1_1', type=float, help='Value of c1 of the base, default=0')
    parser.add_argument('--c2_1', type=float, help='Value of c2 of the base, default=0')
    parser.add_argument('--c1_2', type=float, help='Value of c1 of the base, default=0')
    parser.add_argument('--c2_2', type=float, help='Value of c2 of the base, default=0')
    parser.add_argument('--target', type=float, help='Length target, default=240.667')
    parser.add_argument('--N', type=float, help='Number of interpolation steps, default=100')
    parser.add_argument('--H', type=float, help='Height of the stl, default=1')
    parser.add_argument('--name', help='Name of the stl file, default="r_xxx_XXX_c1_yyy_YYY_c2_zzz_ZZZ.stl"')
    args=parser.parse_args()
    if args.c1_1:
        c1_1 = args.c1_1
    if args.c2_1:
        c2_1 = args.c2_1
    if args.c1_2:
        c1_2 = args.c1_2
    if args.c2_2:
        c2_2 = args.c2_2
    if args.target:
        target = args.target
    if args.N:
        N = args.N
    if args.H:
        H = args.H
    if args.name:
        name = args.name
    else:
        name =  'optimizeds_c1_{:.2f}'.format(c1_1) + '_{:.2f}'.format(c1_2) + '_c2_{:.2f}'.format(c2_1) + '_{:.2f}'.format(c2_2) + '.stl'
    interpolate_stl(target, c1_1, c2_1, c1_2, c2_2, H, N, name)
