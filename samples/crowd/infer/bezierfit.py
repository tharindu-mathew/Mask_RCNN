from numpy import array, linalg, matrix
from scipy.special import comb as nOk
import numpy as np

Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
bezierM = lambda ts: matrix([[Mtk(3,t,k) for k in range(4)] for t in ts])

def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points


def fit_points(points):
    V=array
    E,  W,  N,  S =  V((1,0)), V((-1,0)), V((0,1)), V((0,-1))
    cw = 100
    ch = 300
    cpb = V((0, 0))
    cpe = V((cw, 0))
    xys=[cpb,cpb+ch*N+E*cw/8,cpe+ch*N+E*cw/8, cpe]

    # ts = V(range(11), dtype='float')/10
    ts = V(range(len(points)), dtype='float')/10
    M = bezierM(ts)
    # points = M*points.tolist() #produces the points on the bezier curve at t in ts

    control_points = lsqfit(points, M)
    # assert(linalg.norm(control_points-points)<10e-5)
    # control_points.tolist()[1]
    return np.squeeze(np.asarray(control_points))

def convert_bezier_to_catmull_rom(bezier_control_points):
    catmull_pts = np.zeros_like(bezier_control_points)
    # from https://pomax.github.io/bezierinfo/#curvefitting
    catmull_pts[0] = bezier_control_points[3] + 6 * (bezier_control_points[0] - bezier_control_points[1])
    catmull_pts[1] = bezier_control_points[0]
    catmull_pts[2] = bezier_control_points[3]
    catmull_pts[3] = bezier_control_points[0] + 6 * (bezier_control_points[3] - bezier_control_points[2])
    return catmull_pts


