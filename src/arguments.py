import argparse
import sys
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=3)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument(
        '--verbose',
        help='Verbose for debug',
        action='store_true',
        default=True)


args = parser.parse_args()

if args.verbose:
    fe.set_log_level(20)
else:
    fe.set_log_level(30)


plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Computer Modern Roman"]}) 