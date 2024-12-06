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

from functools import partial
from typing import Optional, Tuple, Union

import os
import tqdm
import pickle
import numpy as np
from PIL import Image
from cv2 import GaussianBlur, resize, INTER_LINEAR

import ifs, diamondsquare

class _GeneratorBase(object):
    def __init__(
            self,
            size: int = 224,
            jitter_params: Union[bool, str] = True,
            flips: bool = True,
            sigma: Optional[Tuple[float, float]] = (0.5, 1.0),
            blur_p: Optional[float] = 0.5,
            niter=1000000,
            patch=True,
    ):
        self.size = size
        self.jitter_params = jitter_params
        self.flips = flips
        self.sigma = sigma
        self.blur_p = blur_p
        self.niter = niter
        self.patch = patch

        self.rng = np.random.default_rng()
        self.cache = {'fg': [], 'bg': []}
        self._set_jitter()

    def _set_jitter(self):
        if isinstance(self.jitter_params, str):
            if self.jitter_params.startswith('fractaldb'):
                k = int(self.jitter_params.split('-')[1]) / 10
                choices = np.linspace(1 - 2 * k, 1 + 2 * k, 5, endpoint=True)
                self.jitter_fnc = partial(self._fractaldb_jitter, choices=choices)
            elif self.jitter_params.startswith('svd'):
                self.jitter_fnc = self._svd_jitter
            elif self.jitter_params.startswith('sval'):
                self.jitter_fnc = self._sval_jitter
        elif self.jitter_params:
            self.jitter_fnc = self._basic_jitter
        else:
            self.jitter_fnc = lambda x: x

    def _fractaldb_jitter(self, sys, choices=(.8, .9, 1, 1.1, 1.2)):
        n = len(sys)
        y, x = np.divmod(self.rng.integers(0, 6, (n,)), 3)
        sys[range(n), y, x] *= self.rng.choice(choices)
        return sys

    def _basic_jitter(self, sys, prange=(0.8, 1.1)):
        # tweak system parameters--randomly choose one transform and scale it
        # this actually amounts to scaling the singular values by a random factor
        n = len(sys)
        sys[self.rng.integers(0, n)] *= self.rng.uniform(*prange)
        return sys

    def _svd_jitter(self, sys):
        '''Jitter the parameters of one of the systems functions, in SVD space.'''
        k = self.rng.integers(0, len(sys) * 3)
        sidx, pidx = divmod(k, 3)
        if pidx < 2:
            q = self.rng.uniform(-0.5, 0.5)
            u, s, v = np.linalg.svd(sys[sidx, :, :2])
            cq, sq = np.cos(q), np.sin(q)
            r = np.array([[cq, -sq], [sq, cq]])
            if pidx == 0:
                u = r @ u
            else:
                v = r @ v
            sys[sidx, :, :2] = (u * s[None, :]) @ v
        else:
            x, y = self.rng.uniform(-0.5, 0.5, (2,))
            sys[sidx, :, 2] += [x, y]
        return sys

    def _sval_jitter(self, sys):
        k = self.rng.integers(0, sys.shape[0])
        svs = np.linalg.svd(sys[..., :2], compute_uv=False)
        fac = (svs * [1, 2]).sum()
        minf = 0.5 * (5 + sys.shape[0])
        maxf = minf + 0.5
        ss = svs[k, 0] + 2 * svs[k, 1]
        smin = (minf - fac + ss) / ss
        smax = (maxf - fac + ss) / ss
        m = self.rng.uniform(smin, smax)
        u, s, v = np.linalg.svd(sys[k, :, :2])
        s = s * m
        sys[k, :, :2] = (u * s[None]) @ v
        return sys

    def jitter(self, sys):
        attempts = 4 if self.jitter_params else 0
        for i in range(attempts):
            # jitter system parameters
            sysc = sys.copy()
            sysc = self.jitter_fnc(sysc)
            # occasionally the modified parameters cause the system to explode
            svd = np.linalg.svd(sysc[:, :, :2], compute_uv=False)
            if svd.max() > 1: continue
            break
        else:
            # fall back on not jittering the parameters
            sysc = sys
        return sysc

    def _iterate(self, sys):
        rng = self.rng

        coords = ifs.iterate(sys, self.niter)
        region = np.concatenate(ifs.minmax(coords))

        return coords, region

    def render(self, sys):
        raise NotImplementedError()

    def random_flips(self, img):
        # random flips/rotations
        if self.rng.random() > 0.5:
            img = img.transpose(1, 0)
        if self.rng.random() > 0.5:
            img = img[::-1]
        if self.rng.random() > 0.5:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        return img

    def to_color(self, img):
        return ifs.colorize(img)

    def to_gray(self, img):
        return (img * 127).astype(np.uint8)[..., None].repeat(3, axis=2)

    def render_background(self):
        bg = diamondsquare.colorized_ds(self.size)
        return bg

    def composite(self, foreground, base, idx=None):
        return ifs.composite(foreground, base)

    def random_blur(self, img):
        sigma = self.rng.uniform(*self.sigma)
        img = GaussianBlur(img, (3, 3), sigma, dst=img)
        return img

    def generate(self, sys):
        raise NotImplementedError()

    def __call__(self, sys, *args, **kwargs):
        return self.generate(sys, *args, **kwargs)


class IFSGenerator(_GeneratorBase):
    def __init__(self, size: int = 224, jitter_params: Union[bool, str] = True, flips: bool = True,
                 scale: Optional[Tuple[float, float]] = (0.5, 2.0), translate: Optional[float] = 0.2,
                 sigma: Optional[Tuple[float, float]] = (0.5, 1.0), blur_p: Optional[float] = 0.5, color=True,
                 background=False, niter=100000, patch=False):
        super().__init__(size, jitter_params, flips, sigma, blur_p, niter, patch)
        self.size = size
        self.jitter_params = jitter_params
        # # # # data augmentation # # # #
        self.flips = flips
        self.scale = scale
        self.translate = translate
        self.sigma = sigma
        self.blur_p = blur_p
        self.color = color
        self.background = background
        # # # # # # # # # # # # # # # # #
        self.niter = niter
        self.patch = patch

        self.rng = np.random.default_rng()
        self.cache = {'fg': [], 'bg': []}
        self._set_jitter()

    def render(self, sys):
        rng = self.rng
        coords, region = self._iterate(sys)

        # transform rendering window (scale and translate)
        if self.translate or self.scale:
            extent = (region[2:] - region[:2])
            center = (region[2:] + region[:2]) / 2
            if self.translate:
                center += extent * rng.uniform(-self.translate, self.translate, (2,))
            if self.scale:
                extent *= rng.uniform(*self.scale, (2,)) / 2
            region[:2] = center - extent
            region[2:] = center + extent

        img = ifs.render(coords, self.size, binary=not self.color, region=region, patch=self.patch)
        return img

    def generate(self, sys):
        rng = self.rng

        sysc = self.jitter(sys)
        img = self.render(sysc)
        self.cache['fg'] = img

        # random flips
        if self.flips:
            img = self.random_flips(img)

        # colorize
        if self.color:
            img = self.to_color(img)
        else:
            img = self.to_gray(img)

        # add random background
        if self.background:
            bg = self.render_background()
            self.cache['bg'] = bg.copy()
            img = self.composite(img, bg)

        # randomly apply gaussian blur
        if self.blur_p and rng.random() > 0.5:
            img = self.random_blur(img)

        return img, sysc


class MultiGenerator(_GeneratorBase):
    def __init__(
        self,
        size: int = 224,
        n_objects: Tuple[int, int] = (10, 30),
        size_range: Tuple[float, float] = (0.15, 0.5),
        jitter_params: Union[bool, str] = False,
        flips: bool = True,
        sigma: Optional[Tuple[float, float]] = (0.5, 1.0),
        blur_p: Optional[float] = 0.5,
        color=True,
        niter = 100000,
        patch = True,
        nobj_p = None,
        background=False,
    ):
        self.size = size
        self.n_objects = n_objects
        self.size_range = size_range
        self.jitter_params = jitter_params
        self.flips = flips
        self.sigma = sigma
        self.blur_p = blur_p
        self.color = color
        self.niter = niter
        self.patch = patch
        self.background = background
        self.rng = np.random.default_rng()

        self.cache = []

        if nobj_p is None:
            self.nobj_p = np.ones(n_objects[1] - n_objects[0] + 1)
        else:
            self.nobj_p = np.array(nobj_p, dtype=np.float64)
        self.nobj_p /= self.nobj_p.sum()

        self._set_jitter()

    def __len__(self):
        return len(self.cache)

    def render(self, sys):
        rng = self.rng
        coords, region = self._iterate(sys)
        # render the fractal at half resolution--it will be resized during generation phase
        img = ifs.render(coords, self.size // 2, binary=False, region=region, patch=self.patch)
        return img

    def generate(self, sys):
        self.cache = sys # list of systems part of  class

        rng = self.rng
        img = self.render_background() if self.background else np.zeros((self.size, self.size, 3), dtype=np.uint8)
        n = rng.choice(range(self.n_objects[0], self.n_objects[1]+1), p=self.nobj_p)
        n = min(n, len(self))

        # I'm creating images with random fractals from the cache
        for i in range(n):
            idx = rng.integers(0, len(self))
            # render image
            sysc = self.cache[idx]
            sysc = self.jitter(sysc)
            fg = self.render(sysc)
            # random flips
            if self.flips:
                fg = self.random_flips(fg)
            # color
            if self.color:
                fg = self.to_color(fg)
            else:
                fg = self.to_gray(fg)

            # random size
            f = rng.uniform(*self.size_range)
            s = int(f * self.size)
            fg = resize(fg, (s, s), interpolation=INTER_LINEAR)
            # random location
            x, y = rng.integers(-(s//2), self.size-(s//2), 2)
            x1 = 0 if x >= 0 else -x
            x2 = s if x < self.size - s else self.size - x
            y1 = 0 if y >= 0 else -y
            y2 = s if y < self.size - s else self.size - y
            fg = fg[y1:y2, x1:x2]
            # add object to image
            y = max(y, 0)
            x = max(x, 0)
            self.composite(fg, img[y:y+fg.shape[0], x:x+fg.shape[1]])

        # randomly apply gaussian blur
        if self.blur_p and rng.random() > self.blur_p:
            img = self.random_blur(img)

        return img

class FractalDatabase(object):
    def __init__(
        self,
        param_file: str = './debug_systems.pkl',
        num_systems: int = 5,
        num_class: int = 2,
        per_class: int = 3,
        jitter_params: bool = False,
        background: bool = False,
        size: int = 512,
        dirs: str='./images',
    ):
        if param_file:
            info_systems = pickle.load(open(param_file, 'rb'))
            kwargs = info_systems['hparams']

            self.params = info_systems['params']
            self.n_max = kwargs['n'][1] - 1

            if len(self.params) > num_systems:
                self.params = self.params[:num_systems]
            elif len(self.params) < num_systems:
                kwargs['num_systems'] = int(num_systems - len(self.params))
                sys = ifs.random_systems(**kwargs)
                self.params.extend(sys.systems)
                sys.systems = self.params
                sys.save_systems_pkl(f'./FractalDatabase-{num_systems}.pkl')
            else:
                pass
        else:
            sys = ifs.random_systems(num_systems, (2, 5), 1, None)
            sys.save_systems_pkl(f'./FractalDatabase-{num_systems}.pkl')
            self.params = sys.systems
            self.n_max = sys.n_max

        self.num_samples = num_class * per_class
        self.per_system = num_class * per_class / num_systems
        self.per_class = per_class

        self.generator = IFSGenerator(size=size, jitter_params=jitter_params, background=background)

        # save information:
        self.print_format = f'{{:0{len(str(self.per_class))}d}}'
        self.images_dir = dirs

    def get_system(self, idx):
        return int(idx // self.per_system)

    def get_label(self, idx):
        return int(idx // self.per_class)

    def integer_to_one_hot(self, integer):
        index = integer - 1
        one_hot = np.zeros(self.n_max)
        one_hot[index] = 1
        return one_hot

    def generate_db(self):
        for idx in tqdm.trange(self.num_samples):
            clsidx = self.get_label(idx)

            filename = self.print_format.format(idx)
            image_path = os.path.join(self.images_dir, str(clsidx), f'image_{filename}.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            sysidx = self.get_system(idx)
            param = self.params[sysidx]['system']
            image, _= self.generator(param)

            # Save image using PIL
            Image.fromarray(image).save(image_path)

        return


class MultiFractalDatabase(object):
    def __init__(
        self,
        param_file: str = './ifs-50k.pkl',
        num_systems: int = 5,
        num_class: int = 10,
        per_class: int = 10,
        jitter_params: bool = False,
        background: bool = False,
        size: int = 512,
        size_range: Tuple[float, float] = (0.15, 0.6),
        n_objects: Tuple[int, int] = (4, 10),
        outdir: str='./images',
    ):
        if param_file:
            info_systems = pickle.load(open(param_file, 'rb'))
            kwargs = info_systems['hparams']

            self.params = info_systems['params']
            self.n_max = kwargs['n'][1] - 1

            if len(self.params) > num_systems:
                self.params = self.params[:num_systems]
            elif len(self.params) < num_systems:
                kwargs['num_systems'] = int(num_systems - len(self.params))
                sys = ifs.random_systems(**kwargs)
                self.params.extend(sys.systems)
                sys.systems = self.params
                sys.save_systems_pkl(f'./FractalDatabase-{num_systems}.pkl')
            else:
                pass
        else:
            sys = ifs.random_systems(num_systems, (2, 5), 1, None)
            sys.save_systems_pkl(f'./FractalDatabase-{num_systems}.pkl')
            self.params = sys.systems
            self.n_max = sys.n_max

        self.num_samples = num_class * per_class
        self.per_class = per_class
        self.per_system = int(num_systems / num_class)
        self.generator = MultiGenerator(size=size, size_range=size_range, n_objects=n_objects, jitter_params=jitter_params, background=background)

        # save information:
        self.print_format = f'{{:0{len(str(self.per_class))}d}}'
        self.images_dir = outdir

    def get_label(self, idx):
        return int(idx // self.per_class)

    def generate_db(self):
        for idx in tqdm.trange(self.num_samples):
            clsidx = self.get_label(idx)
            param = [sys['system'] for sys in self.params[int(clsidx * self.per_system): int((clsidx + 1) * self.per_system)]]
            image = self.generator(param)

            # Save image using PIL so it works for ImageFolder
            imgidx = idx - clsidx * self.per_class
            filename = self.print_format.format(imgidx)
            image_path = os.path.join(self.images_dir, str(clsidx), f'image_{filename}.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray(image).save(image_path)

        return


class MultiMandelbulbDataset(object):
    def __init__(self,
                 rootdir: str = '/data/users/cugwu/MandelbulbVAR-1k',
                 num_systems: int = 2000,
                 num_class: int = 200,
                 per_class: int = 2000,
                 background: bool = False,
                 size: int = 512,
                 size_range: Tuple[float, float] = (0.15, 0.6),
                 n_objects: Tuple[int, int] = (4, 10),
                 outdir: str = './images',
                 ):
        num_rules = os.listdir(rootdir)
        systems = num_rules[:num_systems] if len(num_rules) <= num_systems else num_rules
        self.params = []
        for sys in systems:
            dir_sys = os.path.join(rootdir, sys)
            samples = [os.path.join(dir_sys, sample) for sample in os.listdir(dir_sys)]
            self.params.append({'system': samples})

        self.num_class = num_class
        self.per_class = per_class
        self.num_samples = num_class * per_class
        self.per_system = int(len(systems) / num_class)

        self.background = background
        self.size = size
        self.size_range = size_range
        self.images_dir = outdir
        self.n_objects = n_objects

        self.print_format = f'{{:0{len(str(self.per_class))}d}}'

        self.rng = np.random.default_rng()

    def render_background(self):
        bg = diamondsquare.colorized_ds(self.size)
        return bg

    def composite(self, foreground, base):
        return ifs.composite(foreground, base)

    def generator(self, sys):
        rng = self.rng
        img = self.render_background() if self.background else np.zeros((self.size, self.size, 3), dtype=np.uint8)
        n = rng.choice(range(self.n_objects[0], self.n_objects[1]+1))
        n = min(n, len(sys))

        # I'm creating images with random fractals from the cache
        for i in range(n):
            idx = rng.integers(0, len(sys))
            path = rng.choice(sys[idx])
            fg = np.array(Image.open(path).convert('RGB'))
            fg[np.any(fg == 255, axis=-1)] = 0
            # random size
            f = rng.uniform(*self.size_range)
            s = int(f * self.size)
            fg = resize(fg, (s, s), interpolation=INTER_LINEAR)
            # random location
            x, y = rng.integers(-(s//2), self.size-(s//2), 2)
            x1 = 0 if x >= 0 else -x
            x2 = s if x < self.size - s else self.size - x
            y1 = 0 if y >= 0 else -y
            y2 = s if y < self.size - s else self.size - y
            fg = fg[y1:y2, x1:x2]
            # add object to image
            y = max(y, 0)
            x = max(x, 0)
            self.composite(fg, img[y:y+fg.shape[0], x:x+fg.shape[1]])
        return img

    def get_label(self, idx):
        return int(idx // self.per_class)

    def generate_db(self):
        for idx in tqdm.trange(self.num_samples):
            clsidx = self.get_label(idx)
            param = [sys['system'] for sys in self.params[int(clsidx * self.per_system): int((clsidx + 1) * self.per_system)]]
            image = self.generator(param)

            # Save image using PIL
            imgidx = idx - clsidx * self.per_class
            filename = self.print_format.format(imgidx)
            image_path = os.path.join(self.images_dir, str(clsidx), f'image_{filename}.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray(image).save(image_path)

        return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='multi-mandelbulbs')
    parser.add_argument('--dataset_dir', type=str, default='./MultiMandelbulb')
    parser.add_argument('--source_dir', type=str, default='./MandelbulbVAR-1k')
    parser.add_argument('--param_file', type=str, default='')
    parser.add_argument('--num_systems', type=int, default=1037)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--per_class', type=int, default=20)
    parser.add_argument('--remove_jitter', action='store_true', help='disable parameter jittering')
    parser.add_argument('--background', action='store_true', help='create synthetic images with the background')
    parser.add_argument('--size', type=int, default=512, help='image size')
    args = parser.parse_args()

    if args.dataset_type == 'fractals':
        fractaldb = FractalDatabase(args.param_file, args.num_systems, args.num_class, args.per_class,
                                    not args.remove_jitter, args.background, args.size, args.dataset_dir)
    elif args.dataset_type == 'multi-fractals':
        fractaldb = MultiFractalDatabase(args.param_file, args.num_systems, args.num_class, args.per_class,
                                                 not args.remove_jitter, args.background, args.size, outdir=args.dataset_dir)
    elif args.dataset_type == 'multi-mandelbulbs':
        fractaldb = MultiMandelbulbDataset(args.source_dir, args.num_systems, args.num_class,
                                           args.per_class, args.background, args.size, outdir=args.dataset_dir)
    else:
        raise ValueError("Unsupported dataset class: " + str(args.loader))

    fractaldb.generate_db()
    print(f'Dataset saved in {args.dataset_dir}')