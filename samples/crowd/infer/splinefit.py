import numpy as np

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from lmfit import Minimizer, Parameters, report_fit
from samples.crowd.infer import catmull

from samples.crowd.infer import bezierfit


# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """Model a decaying sine wave and subtract data."""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']
    model = amp * np.sin(x*omega + shift) * np.exp(-x*x*decay)
    return model - data


# def construct_catmull_curve(points):
#     x1 = points[0][0]
#     x2 = points[1][0]
#     y1 = points[0][1]
#     y2 = points[1][1]
#     x3 = points[-2][0]
#     x4 = points[-1][0]
#     y3 = points[-2][1]
#     y4 = points[-1][1]
#     dom = max(points[:, 0]) - min(points[:, 0])
#     rng = max(points[:, 1]) - min(points[:, 1])
#     pctdom = 1
#     pctdom = float(pctdom) / 100
#     prex = x1 + np.sign(x1 - x2) * dom * pctdom
#     prey = (y1 - y2) / (x1 - x2) * (prex - x1) + y1
#     endx = x4 + np.sign(x4 - x3) * dom * pctdom
#     endy = (y4 - y3) / (x4 - x3) * (endx - x4) + y4
#     points = list(points)
#     points.insert(0, np.array([prex, prey]))
#     points.append(np.array([endx, endy]))
#
#     c = CatmullRomChain(points)
#     return c

def curve_min_func(params, actual_curve_pts, orig_pts, img_size):
    num_pts = len(params) // 2
    estimated_pts = []
    for i in range(num_pts):
        pt = np.array([params['x' + str(i)], params['y' + str(i)]])
        estimated_pts.append(pt)

    finetuned_pts = np.stack(estimated_pts)
    finetuned_curve_pts = catmull.create_catmull_spline(finetuned_pts)

    # total_error = 0
    curve_fit_error = compute_error_curve_fit_to_img(actual_curve_pts, finetuned_curve_pts)
    # direction_error = compute_vector_direction_error(finetuned_curve_pts, orig_pts)
    direction_error = compute_vector_direction_error(finetuned_curve_pts, orig_pts)
    total_error = curve_fit_error + direction_error
    total_error /= img_size

    # return total_error * np.ones(len(params))
    # total_error = total_error * np.ones(len(params))
    print(estimated_pts, total_error)
    return total_error




def compute_vector_direction_error(finetuned, orig):
    orig_dir_v = orig[-1] - orig[0]
    finetuned_dir_v  = finetuned[-1] - finetuned[0]
    return 1 if np.sign(np.dot(orig_dir_v, finetuned_dir_v)) != 1 else 0
    # v = finetuned_dir_v - orig_dir_v
    # return np.sqrt(np.dot(v, v))

def compute_error_curve_fit_to_img(actual_curve_pts, estimated_curve_pts):
    # some black pixels represent the curve
    curve_color = 0
    error = 0

    for pt in estimated_curve_pts:
        # img_pt = np.round(pt * w).astype('uint32')
        # val = img[img_pt[1], img_pt[0]]
        # # check col
        # curr_row = img[img_pt[1]]
        # curr_col = transpose_img[img_pt[0]]
        # actual_curve_pts_row = np.where(curr_row == curve_color)
        # actual_curve_pts_col = np.where(curr_col == curve_color)
        # np.abs(actual_curve_pts_row - img_pt[0])
        # error += 0 if (val == curve_color) else 1
        diff = np.min(np.sqrt((actual_curve_pts - pt)**2))
        error += diff
    return error

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(self,
              x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
              linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def fit_data(img, orig_pts):
    # orig_pts = orig_pts * 1.0 / ratio
    num_pts = orig_pts.shape[0]

    h, w = img.shape
    img_size = h * w
    actual_curve_pts = np.asarray(np.where(img == 0)).reshape((-1, 2)) / w
    x, y = np.asarray(np.where(img == 0)) / w
    import scipy.interpolate as interpolate
    tck, u = interpolate.splprep([x, y], s=6)

    N = 100
    xmin, xmax = 0, 1.01
    unew = np.linspace(xmin, xmax, N)
    spline = interpolate.splev(unew, tck)

    plt.plot(x, y, 'bo', label='Original points')
    plt.plot(spline[0], spline[1], 'r', label='BSpline')
    plt.show()

def fit_data3(img, orig_pts):
    ratio = 224.0 / 256.0
    orig_pts = orig_pts * 1.0 / ratio
    params = Parameters()
    num_pts = orig_pts.shape[0]
    for i in range(num_pts):
        params.add('x' + str(i), orig_pts[i][0], min=0.0, max=1.0)
        params.add('y' + str(i), orig_pts[i][1], min=0.0, max=1.0)

    h, w = img.shape
    img_size = h * w
    actual_curve_pts = np.asarray(np.where(img == 0)).reshape((-1, 2)) / w
    act_x, act_y = np.asarray(np.where(img == 0)) / w
    bezier_pts = bezierfit.fit_points(actual_curve_pts)
    estimate_pts = bezierfit.convert_bezier_to_catmull_rom(bezier_pts)
    estimate = catmull.create_catmull_spline(estimate_pts)
    estimate_x, estimate_y = zip(*estimate)


    orig = catmull.create_catmull_spline(orig_pts)
    orig_x, orig_y = zip(*orig)
    plt.plot(act_x, act_y, '.', estimate_x, estimate_y, '-', orig_x, orig_y, '+')
    plt.show()

def fit_data2(img, estimated_pts):
    ratio = 224.0 / 256.0
    estimated_pts = estimated_pts * 1.0/ratio
    params = Parameters()
    num_pts = estimated_pts.shape[0]
    for i in range(num_pts):
        params.add('x' + str(i), estimated_pts[i][0], min=0.0, max=1.0)
        params.add('y' + str(i), estimated_pts[i][1], min=0.0, max=1.0)

    h, w = img.shape
    img_size = h * w
    actual_curve_pts = np.asarray(np.where(img == 0)).reshape((-1, 2)) / w
    act_x, act_y = np.asarray(np.where(img == 0)) / w
    p = np.poly1d(np.polyfit(act_x, act_y, 3))

    estimate = catmull.create_catmull_spline(estimated_pts)
    x1, y1 = zip(*estimate)
    xp = np.linspace(0, 1, 100)
    plt.plot(act_x, act_y, '.', xp, p(xp), '-', x1, y1, '+')

    # actual_curve_pts = np.asarray(np.where(img == 255)).reshape((-1, 2)) / img_size
    # do fit, here with the default leastsq algorithm
    num_for_optimization = 50

    # if (len(actual_curve_pts) > num_for_optimization):
    #     actual_curve_pts = actual_curve_pts[np.random.choice(actual_curve_pts.shape[0], 2, replace=False), :]

    # minner = Minimizer(curve_min_func, params, fcn_args=(actual_curve_pts, estimated_pts, img_size))
    # # result = minner.minimize()
    # result = minner.minimize(method='powell')
    #
    # # calculate final result
    # final_pts = []
    # for i in range(num_pts):
    #     final_pts.append(result.params['x' + str(i)])
    #     final_pts.append(result.params['y' + str(i)])
    #
    # # final = estimated_pts.reshape((num_pts * 2,)) + result.x
    # final = np.asarray(final_pts)
    #
    # # write error report
    # report_fit(result)
    #
    # estimate = catmull.create_catmull_spline(estimated_pts)
    # finetuned = catmull.create_catmull_spline(final.reshape((num_pts, 2)))
    #
    # # plt.xlim(-2, 2)
    # # plt.ylim(-2, 2)
    # # plt.autoscale(False)
    # x1, y1 = zip(*estimate)
    # z1 = np.linspace(0, 1, len(x1))
    # plt.plot(x1, y1, 'k+')
    # # colorline(x1, y1, z1, cmap=plt.get_cmap('bwr'), linewidth=2)
    #
    # x2, y2 = zip(*finetuned)
    # z2 = np.linspace(0, 1, len(x2))
    # plt.plot(x2, y2, 'r+')
    # # colorline(x2, y2, z2, cmap=plt.get_cmap('PRGn'), linewidth=2)
    #
    # x3, y3 = zip(*actual_curve_pts)
    # plt.plot(x3, y3, 'b.')
    #
    plt.show()
    final = None
    return final


def fit_data_sample():
    # create a set of Parameters
    params = Parameters()
    params.add('amp', value=10, min=0)
    params.add('decay', value=0.1)
    params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2.)
    params.add('omega', value=3.0)

    # create data to be fitted
    x = np.linspace(0, 15, 301)
    data = (5.0 * np.sin(2.0 * x - 0.1) * np.exp(-x * x * 0.025) +
            np.random.normal(size=x.size, scale=0.2))

    # do fit, here with the default leastsq algorithm
    minner = Minimizer(fcn2min, params, fcn_args=(x, data))
    result = minner.minimize()

    # calculate final result
    final = data + result.residual

    # write error report
    report_fit(result)

    # try to plot results
    try:
        import matplotlib.pyplot as plt
        plt.plot(x, data, 'k+')
        plt.plot(x, final, 'r')
        plt.show()
    except ImportError:
        pass