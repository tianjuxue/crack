import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.io import loadmat
import h5py
import pandas as pd
from pandas import DataFrame, Series  
import pims
import trackpy as tp



np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)

# mpl.rc('figure',  figsize=(10, 5))
# mpl.rc('image', cmap='gray')


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


@pims.pipeline
def gray(image):
    return image

def test():
    frames = gray(pims.open('data/png/crack_tip/*.png'))
    print(frames)
    print(frames[0][100, :])
    plt.imshow(frames[0])
    f = tp.locate(frames[0], diameter=15, invert=False, minmass=1)
    f.head()
    tp.annotate(f, frames[0])


if __name__ == '__main__':
    # load_data()
    test()
    plt.show()
 