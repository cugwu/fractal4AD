# MIT License
#
# Copyright (c) 2023 Connor Anderson
# Modified by Cynthia I. Ugwu, 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from cv2 import cvtColor, COLOR_HSV2RGB
import numba
import pickle
import numpy as np


def sample_svs(n, a, rng=None):
    '''Sample singular values. 2*`n` singular values are sampled such that the following conditions
    are satisfied, for singular values sv_{i} and i = 0, ..., 2n-1:

    1. 0 <= sv_{i} <= 1
    2. sv_{2i} >= sv_{2i+1}
    3. w.T @ S = `a`, for S = [sv_{0}, ..., sv_{2n-1}] and w = [1, 2, ..., 1, 2]

    Args:
        n (int): number of pairs of singular values to sample.
        a (float): constraint on the weighted sum of all singular values. Note that a must be in the
            range (0, 3*n).
        rng (Optional[numpy.random._generator.Generator]): random number generator. If None (default), it defaults
            to np.random.default_rng().

    Returns:
        Numpy array of shape (n, 2) containing the singular values.
    '''
    if rng is None: rng = np.random.default_rng()
    if a < 0:
        a == 0
    elif a > 3 * n:
        a == 3 * n
    s = np.empty((n, 2))
    p = a
    q = a - 3 * n + 3
    # sample the first 2*(n-1) singular values (first n-1 pairs)
    for i in range(n - 1):
        s1 = rng.uniform(max(0, q / 3), min(1, p))
        q -= s1
        p -= s1
        s2 = rng.uniform(max(0, q / 2), min(s1, p / 2))
        q = q - 2 * s2 + 3
        p -= 2 * s2
        s[i, :] = s1, s2
    # sample the last pair of singular values
    s2 = rng.uniform(max(0, (p - 1) / 2), p / 3)
    s1 = p - 2 * s2
    s[-1, :] = s1, s2

    return s


def sample_system(n=None, constrain=True, bval=1, rng=None, beta=None):
    '''Return n random affine transforms. If constrain=True, enforce the transforms
    to be strictly contractive (by forcing singular values to be less than 1).

    Args:
        n (Union[range,Tuple[int,int],List[int,int],None]): range of values to sample from for the number of
            transforms to sample. If None (default), then sample from range(2, 8).
        constrain (bool): if True, enforce contractivity of transformations. Technically, an IFS must be
            contractive; however, FractalDB does not enforce it during search, so it is left as an option here.
            Default: True.
        bval (Union[int,float]): maximum magnitude of the translation parameters sampled for each transform.
            The translation parameters don't effect contractivity, and so can be chosen arbitrarily. Ignored and set
            to 1 when constrain is False. Default: 1.
        rng (Optional[numpy.random._generator.Generator]): random number generator. If None (default), it defaults
            to np.random.default_rng().
        beta (float or Tuple[float, float]): range for weighted sum of singular values when constrain==True. Let
            q ~ U(beta[0], beta[1]), then we enforce $\sum_{i=0}^{n-1} (s^i_1 + 2*s^i_2) = q$.

    Returns:
        Numpy array of shape (n, 2, 3), containing n sets of 2x3 affine transformation matrices.
        '''
    if rng is None:
        rng = np.random.default_rng()
    if n is None:
        n = rng.integers(2, 8)
    elif isinstance(n, (tuple, list)):
        n = rng.integers(*n)

    if beta is None:
        beta = ((5 + n) / 2, (6 + n) / 2)

    if constrain:
        # sample a matrix with singular values < 1 (a contraction)
        # 1. sample the singular vectors--random orthonormal matrices--by randomly rotating the standard basis
        base = np.sign(rng.random((2 * n, 2, 1)) - 0.5) * np.eye(2)
        angle = rng.uniform(-np.pi, np.pi, 2 * n)
        ss = np.sin(angle)
        cc = np.cos(angle)
        rmat = np.empty((2 * n, 2, 2))
        rmat[:, 0, 0] = cc
        rmat[:, 0, 1] = -ss
        rmat[:, 1, 0] = ss
        rmat[:, 1, 1] = cc
        uv = rmat @ base
        u, v = uv[:n], uv[n:]
        # 2. sample the singular values
        a = rng.uniform(*beta)
        s = sample_svs(n, a, rng)
        # 3. sample the translation parameters from Uniform(-bval, bval) and create the transformation matrix
        m = np.empty((n, 2, 3))
        m[:, :, :2] = u * s[:, None, :] @ v
        m[:, :, 2] = rng.uniform(-bval, bval, (n, 2))
    else:
        m = rng.uniform(-1, 1, (n, 2, 3))

    return m


@numba.njit(cache=True)
def iterate(sys, n_iter=100000):
    '''Compute points in the fractal defined by the system `sys` by random iteration. `n_iter` iterations
    are performed, and a transform is sampled at each iteration according to the probabilites defined by
    `ps`.

    Args:
        sys (np.ndarray): array of shape (n, 2, 3), containing the affine transform parameters.
        n_iter (int): number of iterations/points to calculate.
    Returns:
        ndarray of shape (n_iter, 2) containing the (x, y) coordinates of the generated points.
    '''
    det = sys[:, 0, 0] * sys[:, 1, 1] - sys[:, 0, 1] * sys[:, 1, 0]
    ps = np.abs(det)
    ps = ps / ps.sum()
    ps = np.cumsum(ps)
    coords = np.empty((n_iter, 2))

    # starting point is $v = (I-A_1)^(-1) b$ since this point is gaurenteed to be in the set
    # (assuming that A_1 is contractive) (A_1 = sys[0])
    s = 1 / (1 + det[0] - sys[0, 0, 0] - sys[0, 1, 1])
    x = s * ((1 - sys[0, 1, 1]) * sys[0, 0, 2] + sys[0, 0, 1] * sys[0, 1, 2])
    y = s * ((1 - sys[0, 0, 0]) * sys[0, 1, 2] + sys[0, 1, 0] * sys[0, 0, 2])
    xt = x
    yt = y
    for i in range(n_iter):
        r = np.random.rand()
        for k in range(len(ps)):
            if r < ps[k]: break
        a, b, e, c, d, f = sys[k].ravel()
        x = a * xt + b * yt + e
        y = c * xt + d * yt + f
        if np.isfinite(x) and np.isfinite(y):
            coords[i] = x, y
            xt = x
            yt = y
        else:
            break
    return coords


@numba.njit(cache=True)
def minmax(coords):
    '''Returns both the minimum and maximum values along the 0 axis of an array with shape (n, 2). This only
    requires a single pass through the array, and is faster than calling np.min and np.max seperately.

    Args:
        coords (np.ndarray): an array of shape (n, 2)

    Returns:
        Two ndarrays of shape (2,), the first containing the minimum values and the second containing the maximums
    '''
    mins = np.full(2, np.inf)
    maxs = np.full(2, -np.inf)
    for i in range(len(coords)):
        x, y = coords[i]
        if x < mins[0]: mins[0] = x
        if y < mins[1]: mins[1] = y
        if x > maxs[0]: maxs[0] = x
        if y > maxs[1]: maxs[1] = y
    return mins, maxs


@numba.njit(cache=True)
def _extent(region):
    x1, y1, x2, y2 = region
    xspan = x2 - x1
    xspan = xspan if xspan > 0 else 1
    yspan = y2 - y1
    yspan = yspan if yspan > 0 else 1
    return xspan, yspan


@numba.njit(cache=True)
def _render_binary(coords, s, region):
    '''Renders a square, binary image from coordinate points and a given region.

    Args:
        coords (np.ndarray): coordinate array of shape (n, 2).
        s (int): side length of the rendered image. The image will have width = height = s.
        region (np.ndarray): array of shape (4,), containing [minx, miny, maxx, maxy]. These four values
            define the region in coordinate space that will be rendered to the image. Coordinate points
            that fall outside the bounds of the region will be ignored.

    Returns:
        A binary image as an ndarray of shape (s, s).
    '''
    imgb = np.zeros((s, s), dtype=np.uint8)
    xspan, yspan = _extent(region)
    xscale = (s - 1) / xspan
    yscale = (s - 1) / yspan
    xmin, ymin = region[0], region[1]
    for i in range(len(coords)):
        r = int((coords[i, 0] - xmin) * xscale)
        c = int((coords[i, 1] - ymin) * yscale)
        if r >= 0 and r < s and c >= 0 and c < s:
            imgb[r, c] = 1
    return imgb


@numba.njit(cache=True)
def _render_binary_patch(coords, s, region, patch):
    '''Renders a square, binary image from coordinate points and a given region. Instead of rendering a
    single point for each coordinate, a 3x3 patch is rendered, centered on the coordinate.

    Args:
        coords (np.ndarray): coordinate array of shape (n, 2).
        s (int): side length of the rendered image. The image will have width = height = s.
        region (np.ndarray): array of shape (4,), containing [minx, miny, maxx, maxy]. These four values
            define the region in coordinate space that will be rendered to the image. Coordinate points
            that fall outside the bounds of the region will be ignored.
        patch (np.ndarray): array of shape (3, 3), where each value is either 0 or 1 (binary).

    Returns:
        A grayscale image as an ndarray of shape (s, s).
    '''
    imgb = np.zeros((s, s), dtype=np.uint8)
    xspan, yspan = _extent(region)
    xscale = (s - 1) / xspan
    yscale = (s - 1) / yspan
    xmin, ymin = region[0], region[1]
    for i in range(len(coords)):
        rr = int((coords[i, 0] - xmin) * xscale)
        cc = int((coords[i, 1] - ymin) * yscale)
        for j in range(len(patch)):
            r = rr + patch[j, 0] - 1
            c = cc + patch[j, 1] - 1
            if r >= 0 and r < s and c >= 0 and c < s:
                imgb[r, c] = 1
    return imgb


@numba.njit(cache=True)
def _render_graded(coords, s, region):
    '''Renders a square, grayscale image from coordinate points and a given region. The grayscale values for
    a given pixel is proportional to the number of coordinate points that land on that pixel.

    See _render_binary for an explanation of the arguments.
    '''
    imgf = np.zeros((s, s), dtype=np.float64)
    xspan, yspan = _extent(region)
    xscale = (s - 1) / xspan
    yscale = (s - 1) / yspan
    xmin, ymin = region[0], region[1]
    for i in range(len(coords)):
        r = int((coords[i, 0] - xmin) * xscale)
        c = int((coords[i, 1] - ymin) * yscale)
        if r >= 0 and r < s and c >= 0 and c < s:
            imgf[r, c] += 1
    mval = imgf.max()
    if mval > 0:
        imgf /= mval
    return imgf


@numba.njit(cache=True)
def _render_graded_patch(coords, s, region, patch):
    '''Renders a square, grayscale image from coordinate points and a given region. The grayscale values for
    a given pixel is proportional to the number of coordinate points that land on that pixel. Instead of rendering
    a single point for each coordinate, a 3x3 patch is rendered, centered on the coordinate.

    See _render_binary_patch for an explanation of the arguments.
    '''
    imgf = np.zeros((s, s), dtype=np.float64)
    xspan, yspan = _extent(region)
    xscale = (s - 1) / xspan
    yscale = (s - 1) / yspan
    xmin, ymin = region[0], region[1]
    for i in range(len(coords)):
        rr = int((coords[i, 0] - xmin) * xscale)
        cc = int((coords[i, 1] - ymin) * yscale)
        for j in range(len(patch)):
            r = rr + patch[j, 0] - 1
            c = cc + patch[j, 1] - 1
            if r >= 0 and r < s and c >= 0 and c < s:
                imgf[r, c] += 1
    mval = imgf.max()
    if mval > 0:
        imgf /= mval
    return imgf


def render(coords, s=256, binary=True, region=None, patch=False):
    '''Render an image from a set of coordinates and an optionally specified region.

    Args:
        coords (np.ndarray): coordinate array of shape (n, 2).
        s (int): side length of the rendered image. The image will have width = height = s.
        binary (bool): if True, render a binary image; otherwise, render a grayscale image, where the grayscale
            value is proportional to the number of coordinates that map to the pixel.
        region (Optional[np.ndarray]): array of shape (4,), containing [minx, miny, maxx, maxy]. These four
            values define the region in coordinate space that will be rendered to the image. Coordinate points
            that fall outside the bounds of the region will be ignored. If None (default), the minimum and
            maximum coordinate values are used.
        patch (bool): if False, render each coordinate as a single point. If True, renders a 3x3 patch centered
            at each coordinate. The patch is randomly sampled (each value is chosen uniformly from [0, 1]).

    Returns:
        An image (either grayscale or binary, depending) as an ndarray of shape (s, s).
    '''
    if region is None:
        region = np.concatenate(minmax(coords))
    else:
        region = np.asarray(region)
    if patch:
        p = np.stack(np.divmod(np.arange(9)[np.random.randint(0, 2, (9,), dtype=bool)], 3), 1)
    if binary:
        if patch:
            return _render_binary_patch(coords, s, region, p)
        else:
            return _render_binary(coords, s, region)
    else:
        if patch:
            return _render_graded_patch(coords, s, region, p)
        else:
            return _render_graded(coords, s, region)


@numba.njit(cache=True)
def _hsv_colorize(rendered, min_sat=0.3, min_val=0.5):
    '''Creates a 3-channel HSV image from a 1-channel gray image.
    '''
    h, w = rendered.shape[:2]
    img = np.empty((h, w, 3), dtype=np.uint8)

    hue_shift = np.random.rand() * 255
    sat = np.uint8(np.random.uniform(min_sat, 1) * 255)
    val = np.uint8(np.random.uniform(min_val, 1) * 255)

    for i in range(h):
        for j in range(w):
            x = rendered[i, j]
            if x > 0:
                img[i, j, 0] = np.uint8(x * 255 + hue_shift)  # implicit MOD(256)
                img[i, j, 1] = sat
                img[i, j, 2] = val
            else:
                img[i, j, 0] = 0
                img[i, j, 1] = 0
                img[i, j, 2] = 0
    return img


@numba.njit(cache=True)
def composite(fg, bg):
    '''Copy nonzero pixels from fg into bg. Modifies bg in-place.'''
    for i in range(fg.shape[0]):
        for j in range(fg.shape[1]):
            if fg[i, j, 0] != 0 or fg[i, j, 1] != 0 or fg[i, j, 2] != 0:
                bg[i, j] = fg[i, j]
    return bg


def colorize(rendered, min_sat=0.3, min_val=0.5):
    '''Turns a grayscale image into a color image, where the colors are randomly chosen as explained below.

    First, the grayscale values are converted to the range [0, 255]. A reference hue value h is chosen
    uniformly from [0, 255], and the hue for each pixel p becomes (p + h) mod 256. Then global saturation
    and value scales are chosen uniformly from the ranges [min_sat, 1] and [min_val, 1]. Finally, the image
    is converted to RGB.

    Args:
        rendered (np.ndarray): grayscale image of shape (w, h), with values in the range [0, 1].
        min_sat (float): minimum "saturation" value, defining the range of possible saturation values to draw from.
        min_val (float): minimum "value" value, defining the range of possible "value" (as in light/dark) values to
            draw from.

    Returns:
        A color image as an ndarray of shape (w, h, 3).
    '''
    img = _hsv_colorize(rendered, min_sat, min_val)
    cvtColor(img, COLOR_HSV2RGB, dst=img)
    return img


def random_systems(num_systems, n=(2, 5), bval=None, beta=None):
    '''Sample random systems.
    Args:
        num_systems (int): the number of systems to sample.
        n (int or Tuple[int, int]): the size or range of sizes allowable for the systems.
        bval (float): allowable magnitude of translation parameters.
        beta (float or Tuple[float,float]): singular values constraint. See sample_systems.

    '''
    import tqdm
    systems = []
    for i in tqdm.trange(num_systems):
        s = sample_system(n, bval=bval, beta=beta)
        systems.append({'system': s})
    sys = Sample_Systems(systems, n, bval, beta)
    return sys


class Sample_Systems():
    def __init__(self, systems, n=(2,9), bval=1.0, beta=None):
        self.systems = systems
        if n is None:
            self.n = (2,9)
            self.n_max = 8
        else:
            self.n = n
            self.n_max = n[1] - 1
        self.bval = bval
        self.beta =  beta

    def save_systems_pkl(self, save_path):
        kwargs = dict(
            num_systems=len(self.systems),
            n=self.n,
            bval=self.bval,
            beta=self.beta,
        )
        pickle.dump({'params': self.systems, 'hparams': kwargs}, open(save_path, 'wb'))
        print(f'Saved to {save_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./debug.pkl')
    parser.add_argument('--num_systems', type=int, default=100)
    parser.add_argument('--min_n', type=int, default=2)
    parser.add_argument('--max_n', type=int, default=4)
    parser.add_argument('--bval', type=float, default=1)
    parser.add_argument('--beta_min', type=float, default=None)
    parser.add_argument('--beta_max', type=float, default=None)
    args = parser.parse_args()

    beta = None
    if args.beta_min:
        if args.beta_max:
            beta = (args.beta_min, args.beta_max)
        beta = (args.beta_min, args.beta_min)

    n = None
    if args.min_n:
        if args.max_n:
            n = (args.min_n, args.max_n+1)

    sys = random_systems(args.num_systems, n, args.bval, beta)
    sys.save_systems_pkl(args.save_path)