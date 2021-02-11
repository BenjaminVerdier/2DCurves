from numpy import pi, sin, cos
import numpy as np
import mapbox_earcut as earcut
from stl import mesh

TOP_Z = 1

def polarToCart(t,r):
    return [[cos(theta) * r, sin(theta) * r] for (theta, r) in zip(t, r)]

def lineToSTL(pts, name, z = TOP_Z):
    pts_base = [ [ax,ay,0] for (ax,ay) in pts]
    pts_top = [ [ax,ay,z] for (ax,ay) in pts]

    #points to triangles

    triangles_indices = earcut.triangulate_float32(np.array(pts).reshape(-1,2), np.array([len(pts)])).reshape(-1,3)

    #2d triangles to 3d triangles

    base = [ [pts_base[a], pts_base[b], pts_base[c]] for (a,b,c) in triangles_indices]
    top = [ [pts_top[a], pts_top[b], pts_top[c]] for (a,b,c) in triangles_indices]

    n_facets = 2 * len(pts) + 2 * triangles_indices.shape[0]

    data = np.zeros(n_facets, dtype=mesh.Mesh.dtype)

    for i in range(len(base)):
        data['vectors'][2*i] = np.array([base[i][0], base[i][1], base[i][2]])
        data['vectors'][2*i + 1] = np.array([top[i][0], top[i][1], top[i][2]])

    for i in range(len(pts)):
        data['vectors'][2*len(base) + 2*i] = np.array([pts_base[i], pts_top[i-1], pts_base[i-1]])
        data['vectors'][2*len(base) + 2*i + 1] = np.array([pts_base[i], pts_top[i], pts_top[i-1]])

    new_mesh = mesh.Mesh(data)

    new_mesh.save(name)
