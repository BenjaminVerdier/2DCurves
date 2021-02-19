import numpy as np
from intersect import intersection
from ground.base import get_context
from bentley_ottmann.planar import edges_intersect

import matplotlib.pyplot as plt

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

def check_for_self_intersection(pts):
    context = get_context()
    Point, Contour = context.point_cls, context.contour_cls
    ctr = [ Point(a,b) for [a,b] in pts ]
    curve = Contour(ctr)
    return edges_intersect(curve)


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
