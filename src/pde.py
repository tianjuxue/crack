import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
from . import arguments


class PDE(object):
    def __init__(self, args):
        self.args = args
        self._build_mesh()
        self._build_function_space()
        self._create_boundary_measure()


if __name__ == '__main__':
	args = arguments.args