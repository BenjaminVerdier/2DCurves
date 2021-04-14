from numpy import pi, sin, cos
import numpy as np
from math import sqrt
from functools import partial
import mpmath as mp
import mapbox_earcut as earcut
from stl import mesh

from intersect import intersection
from ground.base import get_context
from bentley_ottmann.planar import edges_intersect

import matplotlib.pyplot as plt

#================ Geometry Related ================

# returns polar corrdinates from theta, r arrays of polar coordinates.
def polarToCart(t,r):
    return [[cos(theta) * r, sin(theta) * r] for (theta, r) in zip(t, r)]

# returns radii of summed cosine formula
def radius_SC(theta, radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))


# creates an stl file from a set of points defining a closed curve
def lineToSTL(pts, name, z = 1):
    # 3D vertices of the base and top faces.
    pts_base = [ [ax,ay,0] for (ax,ay) in pts]
    pts_top = [ [ax,ay,z] for (ax,ay) in pts]

    # Triangulate the 2D curve
    triangles_indices = earcut.triangulate_float32(np.array(pts).reshape(-1,2), np.array([len(pts)])).reshape(-1,3)

    # 2d triangles to 3d triangles
    base = [ [pts_base[a], pts_base[b], pts_base[c]] for (a,b,c) in triangles_indices]
    top = [ [pts_top[a], pts_top[b], pts_top[c]] for (a,b,c) in triangles_indices]

    # Need to get the number of triangles: 2 per segment plus twice the number given by our triangulization
    n_facets = 2 * len(pts) + 2 * triangles_indices.shape[0]

    data = np.zeros(n_facets, dtype=mesh.Mesh.dtype)

    #Add base and top triangles
    for i in range(len(base)):
        data['vectors'][2*i] = np.array([base[i][0], base[i][1], base[i][2]])
        data['vectors'][2*i + 1] = np.array([top[i][0], top[i][1], top[i][2]])
    #Add side triangles.
    for i in range(len(pts)):
        data['vectors'][2*len(base) + 2*i] = np.array([pts_base[i], pts_top[i-1], pts_base[i-1]])
        data['vectors'][2*len(base) + 2*i + 1] = np.array([pts_base[i], pts_top[i], pts_top[i-1]])

    new_mesh = mesh.Mesh(data)
    new_mesh.save(name)


#================ Length Calculations ================

# returns length of fermat spiral with coefficient a for theta between 0 and x.
def fermat_length(a,x):
    F = mp.ellipf(mp.acos((0.5-x) / (0.5+x)), 0.5)
    return a*(F + sqrt(2*x*(4*x**2+1))) / sqrt(18)

# returns length of archimedian spiral with coefficient a for theta between 0 and x.
def arch_length(a,x):
    return a*(x*sqrt(x*x + 1) + np.arcsinh(x)) / 2

# returns sum of length of closed curve defined by points in pts.
# pts is a n x 2 numpy array.
def length_from_points(pts):
    total_length = 0
    for i in range(pts.shape[0]):
        total_length += np.linalg.norm(pts[i,:] - pts[i-1,:])
    return total_length

# returns the approximate length of summed cosine curve with parameters R, c1, c2 using Simpson's rule.
def approximate_integral_SC_length(R, c1, c2, N=50):
    def func(R, c1, c2, theta):
        return np.sqrt( (R * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta)))**2 + (4 * R * (c1 * sin(4 * theta) + 2 * c2 * sin(8 * theta)))**2 )
    a,b = (0, 2*pi)
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = func(R, c1, c2, x)
    S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    return S


# original code for the Simpson's rule computation.
def simps(f, a, b, N=50):
    '''Approximate the integral of f(x) from a to b by Simpson's rule.

    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.

    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0
    '''
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = f(x)
    S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    return S


#================ Curve Sanity Checks ================

# offsets the curve along its normal at every point. width can be negative.
def offset_curve(pts, width):
    new_curve = pts.copy()
    for i in range(len(pts)-1):
        tangent = pts[i+1] - pts[i-1]
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal/np.linalg.norm(normal)
        new_curve[i] = pts[i] + width * normal
    tangent = pts[0] - pts[len(pts)-2]
    normal = np.array([-tangent[1], tangent[0]])
    normal = normal/np.linalg.norm(normal)
    new_curve[-1] = pts[-1] + width * normal
    return new_curve

# checks for self-intersection in a curve.
# basically a wrapper for bentley_ottmann.planar.edges_intersect
def check_for_self_intersection(pts):
    context = get_context()
    Point, Contour = context.point_cls, context.contour_cls
    ctr = [ Point(a,b) for [a,b] in pts ]
    curve = Contour(ctr)
    return edges_intersect(curve)

# checks for intersections between the curve and its offsets, plus self intersections of offsets.
# Probably too much.
def check_for_intersection(pts, width, display=True):
    curve_pos = offset_curve(pts, width/2)
    curve_neg = offset_curve(pts, -width/2)
    if display:
        plt.figure(2)
        plt.plot(curve_pos[:,0], curve_pos[:,1], color='r')
        plt.plot(curve_neg[:,0], curve_neg[:,1], color='b')
        plt.plot(pts[:,0], pts[:,1], color='g')
        plt.show()
    x, y = intersection(curve_pos[:,0], curve_pos[:,1],curve_neg[:,0], curve_neg[:,1])
    return len(x) > 0 or check_for_self_intersection(curve_pos) or check_for_self_intersection(curve_neg)

# returns false iff radii contains a value close enough to 0 or negative.
def check_for_neg_rad(radii):
    return np.amin(radii) < 0.01

# wrapper for overlap function, converts pts to numpy array, doesn't display
def check_overlap(theta, radii, W):
    pts = polarToCart(theta, radii)
    return check_for_intersection(np.array(pts), W, False)

def checkBB(radii, BB):
    return np.amax(radii) > BB

def sanity_check(theta, radii, W, BB, c_bb = True, c_negrad = True, c_overlap = True):
    if c_bb and checkBB(radii, BB):
        return False
    if c_negrad and check_for_neg_rad(radii):
        return False
    if c_overlap and check_overlap(theta, radii, W):
        return False
    return True

def sanity_check_verbose(theta, radii, W, BB, c_bb = True, c_negrad = True, c_overlap = True):
    problem = False
    if c_bb and checkBB(radii, BB):
        print("Curve exceeds bounding box!")
        problem = True
    if c_negrad and check_for_neg_rad(radii):
        print("Curve has negative radius!")
        problem = True
    if c_overlap and check_overlap(theta, radii, W):
        print("Curve self-overlaps!")
        problem = True
    if not problem:
        print("The curve is valid.")
        return True
    return False
