import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.io import loadmat
import h5py


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)


def load_data():
    x1 = loadmat('data/mat/calib.mat')
    x2 = loadmat('data/mat/tippoint.mat')
    for key, value in x2.items():
        print(key)

    path_tracer = 'data/mat/tracer_track.mat'
    arrays = {}
    f = h5py.File(path_tracer)
    for k, v in f.items():
        arrays[k] = np.array(v)
        print(k)

    print(arrays['tks'].shape)


if __name__ == '__main__':
    load_data()
