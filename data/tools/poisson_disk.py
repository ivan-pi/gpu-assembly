import random
from math import sqrt
from random import random as rand

import numpy as np


def inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class pds:
    """A Poisson Disk Sampling object.

    The original code was developed by Connor Johnson.
    For more information see:
    http://connor-johnson.com/2015/04/08/poisson-disk-sampling/

    The version here adds some poorly designed support for
    periodic Poisson Disk Sampling.

    """

    def __init__(self, width, height, radius, max_shots, periodic=False, seed=None):
        # w and h are the width and height of the field
        self.w = width
        self.h = height
        # n is the number of test points
        self.n = max_shots
        self.periodic = periodic
        self.seed = seed
        self.r = 0.9 * radius
        self.r2 = self.r**2.0
        self.A = 3.0 * self.r2
        # cs is the cell size
        self.cs = self.r / sqrt(2)
        # gw and gh are the number of grid cells
        self.gw = int((self.w / self.cs)) + 1
        self.gh = int((self.h / self.cs)) + 1
        # print(self.gw,self.gh)
        # if self.periodic:
        #     self.gw += 1
        #     self.gh += 1
        # create a grid and a queue
        self.grid = [None] * self.gw * self.gh
        self.queue = list()
        # set the queue size and sample size to zero
        self.qs, self.ss = 0, 0

    def add_seed_points(self, points):

        for point in points:
            x, y = point
            if self.periodic:
                if x < 0.0:
                    x += self.w
                if x >= self.w:
                    x -= self.w
                if y < 0.0:
                    y += self.h
                if y >= self.w:
                    y -= self.h
            self.set_point(x, y)

    def distance(self, x, y):

        # find where (x,y) sits in the grid
        x_idx = int(x / self.cs)
        y_idx = int(y / self.cs)

        # determine a neighborhood of cells around (x,y)
        if self.periodic:
            # x0 = x_idx % self.gw
            # y0 = y_idx % self.gh
            xr = [(x_idx - 2 + i) % self.gw for i in range(5)]
            yr = [(y_idx - 2 + i) % self.gh for i in range(5)]
        else:
            x0 = max(x_idx - 2, 0)
            y0 = max(y_idx - 2, 0)
            x1 = min(x_idx + 3, self.gw)
            y1 = min(y_idx + 3, self.gh)
            xr = list(range(x0, x1))
            yr = list(range(y0, y1))

        # search around (x,y)
        for y_idx in yr:
            for x_idx in xr:
                step = y_idx * self.gw + x_idx
                # if the sample point exists on the grid
                if self.grid[step]:
                    s = self.grid[step]
                    dx = s[0] - x
                    dy = s[1] - y
                    if dx > 0.5 * self.w:
                        dx -= self.w
                    if dx <= -0.5 * self.w:
                        dx += self.w
                    if dy > 0.5 * self.h:
                        dy -= self.h
                    if dy <= -0.5 * self.h:
                        dy += self.h
                    dx = dx**2
                    dy = dy**2
                    # print(np.sqrt(dx + dy),self.r)
                    # and it is too close
                    if dx + dy < self.r2:
                        # then barf
                        return False
        return True

    def set_point(self, x, y):
        s = [x, y]
        self.queue.append(s)
        # find where (x,y) sits in the grid
        x_idx = int(x / self.cs)
        y_idx = int(y / self.cs)
        step = self.gw * y_idx + x_idx
        self.grid[step] = s
        self.qs += 1
        self.ss += 1
        return s

    def rvs(self):
        random.seed(self.seed)
        if self.ss == 0:
            x = rand() * self.w
            y = rand() * self.h
            self.set_point(x, y)
        while self.qs:
            x_idx = int(rand() * self.qs)
            s = self.queue[x_idx]
            for y_idx in range(self.n):
                a = 2.0 * np.pi * rand()  # random angle
                b = self.r * (1.0 + rand())
                # b = np.sqrt( self.A * random() + self.r2 )
                x = s[0] + b * np.cos(a)
                y = s[1] + b * np.sin(a)
                if self.periodic:
                    if x < 0.0:
                        x += self.w
                    if x >= self.w:
                        x -= self.w
                    # if (x <= 0) or (x >= self.w):
                    #     if (x <= 0):
                    #         x += self.w
                    #     if (x >= self.w):
                    #         x -= self.w
                    if (y <= 0) or (y >= self.h):
                        if y <= 0:
                            y += self.h
                        if y >= self.h:
                            y -= self.h
                    if self.distance(x, y):
                        self.set_point(x, y)
                else:
                    if (x >= 0) and (x < self.w):
                        if (y >= 0) and (y < self.h):
                            if self.distance(x, y):
                                self.set_point(x, y)
            del self.queue[x_idx]
            self.qs -= 1
        sample = list(filter(None, self.grid))
        sample = np.asarray(sample, dtype=float)
        return sample


def example():
    obj = pds(1, 1, 0.02, 10)
    sample1 = obj.rvs()
    z = np.sin(6 * sample1[:, 0]) + np.cos(2 * sample1[:, 1])
    print(sample1)
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    triang = Triangulation(sample1[:, 0], sample1[:, 1])

    plt.plot(sample1[:, 0], sample1[:, 1], "o")
    plt.triplot(triang, color="0.8")
    plt.axis("equal")
    plt.show()

    plt.tricontourf(triang, z, 40, cmap=plt.cm.viridis)
    plt.show()

    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(sample1[:, 0], sample1[:, 1], z, cmap=cm.viridis, linewidth=0.2)
    # ax.tricontourf(sample1[:,0],sample1[:,1],z, cmap=cm.viridis)
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":

    example()
