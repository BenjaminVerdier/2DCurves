from numpy import pi, sin, cos
import numpy as np
import mapbox_earcut as earcut
from stl import mesh
from linetostl import polarToCart
import argparse

theta = np.arange(0, (2 * pi), 0.01)

def radius_comp(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

def twist_stl(R, c1, c2, T, z, steps, name):
    twists = np.linspace(0, T, steps)

    pts_3d = [[]]*steps
    height_step = z / (steps-1)

    for i in range(steps):
        temp_polar = radius_comp(R, c1, c2)
        temp_cart = polarToCart(theta + twists[i], temp_polar)
        pts_3d[i] =  [ [ax,ay,i*height_step] for (ax,ay) in temp_cart]


    pts_base = polarToCart(theta, radius_comp(R, c1, c2))
    pts_top = polarToCart(theta + twists[steps-1], radius_comp(R, c1, c2))

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
    R = 1
    c1 = 0
    c2 = 0
    T = 0
    N = 100
    H = 1
    name = ''
    parser=argparse.ArgumentParser()
    parser.add_argument('--R', type=float, help='Radius of the base, default=1')
    parser.add_argument('--c1', type=float, help='Value of c1, default=0')
    parser.add_argument('--c2', type=float, help='Value of c2, default=0')
    parser.add_argument('--T', type=float, help='Total twist, default=0')
    parser.add_argument('--N', type=float, help='Number of interpolation steps, default=100')
    parser.add_argument('--H', type=float, help='Height of the stl, default=1')
    parser.add_argument('--name', help='Name of the stl file, default="r_xxx_c1_yyy_c2_zzz_T_ttt.stl"')
    args=parser.parse_args()
    if args.R:
        R = args.R
    if args.c1:
        c1 = args.c1
    if args.c2:
        c2 = args.c2
    if args.T:
        T = args.T
    if args.N:
        N = args.N
    if args.H:
        H = args.H
    if args.name:
        name = args.name
    else:
        name =  'r_{:.2f}'.format(R) + '_c1_{:.2f}'.format(c1) + '_c2_{:.2f}'.format(c2) + '_T_{:.2f}'.format(T) + '.stl'
    twist_stl(R, c1, c2, T, H, N, name)
