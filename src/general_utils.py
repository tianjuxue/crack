import numpy as np
import matplotlib.pyplot as plt


def show_map():
    x1 = np.linspace(0, 0.5, 101)
    y1 = 0.5*x1
    x2 = np.linspace(0.5, 1, 101)
    y2 = -6*x2**3 + 14*x2**2 -9*x2 + 2
    x3 = np.linspace(1, 2, 101)
    y3 = x3
    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x3, y3)
    plt.show()