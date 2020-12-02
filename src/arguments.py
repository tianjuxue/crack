import argparse
import sys
import numpy as np
import fenics as fe

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
