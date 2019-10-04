import numpy


# In[2]:

def CatmullRomSpline(P0, P1, P2, P3, a, nPoints=100):
    """
    P0, P1, P2, and P3 should be (x,y) point pairs that define the Catmull-Rom spline.
    nPoints is the number of points to include in this curve segment.
    """
    # Convert the points to numpy so that we can do array multiplication
    P0, P1, P2, P3 = map(numpy.array, [P0, P1, P2, P3])

    # Calculate t0 to t4
    alpha = a
    def tj(ti, Pi, Pj):
        xi, yi = Pi
        xj, yj = Pj
        return ( ( (xj-xi)**2 + (yj-yi)**2 )**0.5 )**alpha + ti

    t0 = 0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    # Only calculate points between P1 and P2
    t = numpy.linspace(t1,t2,nPoints)

    # Reshape so that we can multiply by the points P0 to P3
    # and get a point for each value of t.
    t = t.reshape(len(t),1)

    A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
    A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
    A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3

    B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
    B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3

    C  = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
    return C

def CatmullRomChain(P,alpha=0.5):
    """
    Calculate Catmull Rom for a chain of points and return the combined curve.
    """
    sz = len(P)

    # The curve C will contain an array of (x,y) points.
    C = []
    for i in range(sz-3):
        c = CatmullRomSpline(P[i], P[i+1], P[i+2], P[i+3],alpha)
        C.extend(c)

    return C

import numpy as np

def create_catmull_spline(points, area=False):
    if (area):
        points = np.vstack((points, points[0]))
        #points = np.vstack((points, points[0]))

    x1 = points[0][0]
    x2 = points[1][0]
    y1 = points[0][1]
    y2 = points[1][1]
    x3 = points[-2][0]
    x4 = points[-1][0]
    y3 = points[-2][1]
    y4 = points[-1][1]
    dom = max(points[:, 0]) - min(points[:, 0])
    rng = max(points[:, 1]) - min(points[:, 1])
    pctdom = 1
    pctdom = float(pctdom) / 100
    prex = x1 + np.sign(x1 - x2) * dom * pctdom
    prey = (y1 - y2) / (x1 - x2) * (prex - x1) + y1
    endx = x4 + np.sign(x4 - x3) * dom * pctdom
    endy = (y4 - y3) / (x4 - x3) * (endx - x4) + y4
    print
    len(points)
    points = list(points)
    points.insert(0, np.array([prex, prey]))
    points.append(np.array([endx, endy]))

    c = CatmullRomChain(points)
    return c

# In[139]:

# Define a set of points for curve to go through
Points = numpy.random.rand(10,2)
#Points=array([array([153.01,722.67]),array([152.73,699.92]),array([152.91,683.04]),array([154.6,643.45]),
#        array([158.07,603.97])])
#Points = array([array([0,92.05330318]),
#               array([2.39580622,29.76345192]),
#               array([10.01564963,16.91470591]),
#               array([15.26219886,71.56301997]),
#               array([15.51234733,73.76834447]),
#               array([24.88468545,50.89432899]),
#               array([27.83934153,81.1341789]),
#               array([36.80443404,56.55810783]),
#               array([43.1404725,16.96946811]),
#               array([45.27824599,15.75903418]),
#               array([51.58871027,90.63583215])])

# x1=Points[0][0]
# x2=Points[1][0]
# y1=Points[0][1]
# y2=Points[1][1]
# x3=Points[-2][0]
# x4=Points[-1][0]
# y3=Points[-2][1]
# y4=Points[-1][1]
# dom=max(Points[:,0])-min(Points[:,0])
# rng=max(Points[:,1])-min(Points[:,1])
# pctdom=1
# pctdom=float(pctdom)/100
# prex=x1+sign(x1-x2)*dom*pctdom
# prey=(y1-y2)/(x1-x2)*(prex-x1)+y1
# endx=x4+sign(x4-x3)*dom*pctdom
# endy=(y4-y3)/(x4-x3)*(endx-x4)+y4
# print len(Points)
# Points=list(Points)
# Points.insert(0,array([prex,prey]))
# Points.append(array([endx,endy]))
# print len(Points)


# In[140]:

#Define alpha
# a=0.

# Calculate the Catmull-Rom splines through the points
