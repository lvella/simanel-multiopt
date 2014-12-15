#! /usr/bin/env python3.4

import numpy as np
import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1], 'r')
lines = [map(float, l.split()) for l in f]
x, y = zip(*lines)

line, = plt.plot(x, y, 'o')
plt.show()
