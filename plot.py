import numpy as np
import matplotlib.pyplot as plt

f = open('output.dat', 'r')
lines = [map(float, l.split()) for l in f]
x, y = zip(*lines)

line, = plt.plot(x, y, 'o')
plt.show()
