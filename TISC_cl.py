from numpy import pi, sin, cos
import numpy as np
import mapbox_earcut as earcut
from stl import mesh
import argparse
from functools import partial
from scipy.optimize import minimize
import utils

# Globals
theta = np.arange(0, (2 * pi), 0.01)

# function to be partialized (for target and fixed parameters) and minimized
def optimizable_function(target, c1, c2, param):
    return abs(target - utils.approximate_integral_SC_length(param[0], c1, c2))

def interpolate_stl(target, target_var, c1_1, c2_1, c1_2, c2_2, twist, z, steps, name):
    # linear interpolation of targets, c1s, c2s and twist angles which are the parameters of the curve at each slice
    targets = np.linspace(target - target_var/2, target + target_var/2, steps)
    c1s = np.linspace(c1_1, c1_2, steps)
    c2s = np.linspace(c2_1, c2_2, steps)
    twists = np.linspace(0, T, steps)

    pts_3d = [[]]*steps
    height_step = z / (steps-1)
    tuples = [()]*steps

    # For each 'slice'
    for i in range(steps):
        # Partialize the function with the right values for c1 and c2
        func_to_opt = partial(optimizable_function, targets[i], c1s[i], c2s[i])
        # Inital value for the radius, works for c1 = c2 = 0
        x0=[38]
        # minimize the length difference
        res = minimize(func_to_opt, np.array(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
        # We get out (R, c1, c2) tuple, with optimized R
        tuples[i] = (res.x[0], c1s[i], c2s[i])
        # compute radii for current slice's parameters
        temp_polar = utils.radius_SC(theta, *tuples[i])
        # Convert polar coordinates to cartesian, adding the twist of the slice
        temp_cart = utils.polarToCart(theta + twists[i], temp_polar)
        # Convert to 3d and store in our array
        pts_3d[i] =  [ [ax,ay,i*height_step] for (ax,ay) in temp_cart]

    # We need to triangulize the base and top, so first we recompute the cartesian coordinates of those
    pts_base = utils.polarToCart(theta, utils.radius_SC(theta, *tuples[0]))
    pts_top = utils.polarToCart(theta + twist, utils.radius_SC(theta, *tuples[steps-1]))

    #triangulate those faces
    triangles_indices_base = earcut.triangulate_float32(np.array(pts_base).reshape(-1,2), np.array([len(pts_base)])).reshape(-1,3)
    triangles_indices_top = earcut.triangulate_float32(np.array(pts_top).reshape(-1,2), np.array([len(pts_top)])).reshape(-1,3)

    #sort the results to fit our data formatting
    base = [ [pts_3d[0][a], pts_3d[0][b], pts_3d[0][c]] for (a,b,c) in triangles_indices_base]
    top = [ [pts_3d[steps-1][a], pts_3d[steps-1][b], pts_3d[steps-1][c]] for (a,b,c) in triangles_indices_top]

    # number of facets, necessary to create our mesh object.
    # between 2 slices, we have 2 faces per segment, ie 2 faces per point because our curve is closed.
    # Plus we have the number of triangles of the top and base faces.
    n_facets = 2 * len(theta) * (steps-1) + len(base) + len(top)

    data = np.zeros(n_facets, dtype=mesh.Mesh.dtype)

    # Add faces of the base
    for i in range(len(base)):
        data['vectors'][i] = np.array([base[i][0], base[i][1], base[i][2]])

    offset = len(base)
    # Add faces of the top, minding the offset
    for i in range(len(top)):
        data['vectors'][offset + i] = np.array([top[i][0], top[i][1], top[i][2]])

    offset += len(top)
    # Add the faces of the sides. This should work out so that their normals are pointing outward.
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
    T = 0
    target = 240.667
    target_var = 0
    N = 100
    H = 1
    name = ''
    parser=argparse.ArgumentParser()
    parser.add_argument('--c1_1', type=float, help='Value of c1 of the base, default=0')
    parser.add_argument('--c2_1', type=float, help='Value of c2 of the base, default=0')
    parser.add_argument('--c1_2', type=float, help='Value of c1 of the base, default=c1_1')
    parser.add_argument('--c2_2', type=float, help='Value of c2 of the base, default=c2_1')
    parser.add_argument('--T', type=float, help='Total twist, default=0')
    parser.add_argument('--target', type=float, help='Length target, in mm, default=240.667')
    parser.add_argument('--target_var', type=float, help='Variation of target length between top and bottom, in mm, default=0')
    parser.add_argument('--N', type=float, help='Number of interpolation steps, default=100')
    parser.add_argument('--H', type=float, help='Height of the stl, default=1')
    parser.add_argument('--name', help='Name of the stl file, default="optimized__c1_yyy_YYY_c2_zzz_ZZZ_T_ttt.stl"')
    args=parser.parse_args()
    if args.c1_1:
        c1_1 = args.c1_1
    if args.c2_1:
        c2_1 = args.c2_1
    if args.c1_2:
        c1_2 = args.c1_2
    else:
        c1_2 = c1_1
    if args.c2_2:
        c2_2 = args.c2_2
    else:
        c2_2 = c2_1
    if args.T:
        T = args.T
    if args.target:
        target = args.target
    if args.target_var:
        target_var = args.target_var
    if args.N:
        N = args.N
    if args.H:
        H = args.H
    if args.name:
        name = args.name
    else:
        name =  'optimized_c1_{:.2f}'.format(c1_1) + '_{:.2f}'.format(c1_2) + '_c2_{:.2f}'.format(c2_1) + '_{:.2f}'.format(c2_2) + '_T_{:.2f}'.format(T) + '.stl'
    interpolate_stl(target, target_var, c1_1, c2_1, c1_2, c2_2, T, H, N, name)
