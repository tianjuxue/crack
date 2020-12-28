import fenics as fe
import dolfin_adjoint as da
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
import shutil
from functools import partial
import scipy.optimize as opt
from pyadjoint.overloaded_type import create_overloaded_object
from ..pde import PDE
from .. import arguments
from ..constitutive import *
from ..mfem import distance_function_segments_ufl, map_function_normal, inverse_map_function_normal, map_function_ufl
