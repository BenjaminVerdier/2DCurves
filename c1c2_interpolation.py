from numpy import pi, sin, cos
import numpy as np
import mapbox_earcut as earcut
from stl import mesh
from linetostl import polarToCart

TOP_Z = 1

theta = np.arange(0, (2 * pi), 0.01)

def radius_comp(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

def interpolate_stl(r_1, c1_1, c2_1, r_2, c1_2, c2_2, z = TOP_Z, steps = 1000):
    tuples = np.linspace((r_1, c1_1, c2_1), (r_2, c1_2, c2_2), steps)

    pts_3d = [[]]*steps
    height_step = z / (steps-1)

    for i in range(steps):
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

    new_mesh.save('test.stl')

interpolate_stl(5,-0.1,0.2,4,0.2,-0.3,10,100)
