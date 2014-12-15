#! /usr/bin/env python3.4

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

f = open(sys.argv[1], 'r')
lines = [list(map(float, l.split())) for l in f]
if len(lines[0]) == 2:
    x, y = zip(*lines)

    line, = plt.plot(x, y, 'o')
    plt.show()
else:
    if len(lines) > 1200:
        lines = random.sample(lines, 1200)
    x, y, z = zip(*lines)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')

    plt.show()

