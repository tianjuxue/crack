import fenics as fe
import dolfin_adjoint as da
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
from functools import partial
from pyadjoint.overloaded_type import create_overloaded_object
from ..pde import MappedPDE
from .. import arguments
from ..constitutive import *
from ..mfem import map_function_ufl


class PureShear(MappedPDE):
    def __init__(self, args):
        self.case_name = "pure_shear"
        self.solution_scheme = 'explicit'
        self.local_refinement_iteration = 1
        super(PureShear, self).__init__(args)

        self.displacements = 1e-1*np.concatenate((np.linspace(0, 0.08, 11), np.linspace(0.08, 0.16, 301)))
        # self.displacements = 1e-1*np.linspace(0.0, 0.15, 51)
 
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))
 
        self.mu = 80.77        
        self.lamda = 121.15
        self.G_c = 2.7*1e-3
        self.psi_cr = 0.

        self.l0 = 0.02
        print(self.mesh.hmax())
        print(self.mesh.hmin())        
        print("self.l0 is {}".format(self.l0))

        self.map_type = 'identity'

        if self.map_type == 'linear' or self.map_type == 'smooth':
            self.map_flag = True
        elif self.map_type == 'identity':
            self.finish_flag = True
            self.map_flag = False

        self.rho_default = 0.14
        self.d_integral_interval = 0.1 * self.rho_default**2
        self.initialize_control_points_and_impact_radii()


    def initialize_control_points_and_impact_radii(self):
        self.control_points = []
        self.impact_radii = []
        control_points = np.asarray([[self.length/2, self.height/2]])
        for new_tip_point in control_points:
            self.compute_impact_radii(new_tip_point)


    def build_mesh(self):
        self.length = 1.
        self.height = 1.
 
        plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(self.length, self.height))
        notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1e-10), fe.Point(0, self.height / 2 - 1e-10), fe.Point(self.length / 2, self.height / 2)])

        resolution = 50 if self.local_refinement_iteration == 0 else 100

        self.mesh = mshr.generate_mesh(plate - notch, resolution)

        length = self.length
        height = self.height

        class Lower(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0)

        class Upper(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], height)

        class Left(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0)

        class Right(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and fe.near(x[0], length)

        class Corner(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return fe.near(x[0], 0) and fe.near(x[1], 0)

        self.lower = Lower()
        self.upper = Upper()
        self.corner = Corner()
        self.left = Left()
        self.right = Right()


    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)
 
        # self.presLoad = da.Expression(("t", 0.), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper)
        # self.BC_u = [BC_u_lower, BC_u_upper] 
        # self.BC_d = []

        self.presLoad = da.Expression(("t", 0), t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        BC_u_left = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.left)
        BC_u_right = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.right)
        self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        self.BC_d = []


def test(args):
    pde = PureShear(args)
    # pde.staggered_solve()
    pde.post_processing()
 

if __name__ == '__main__':
    args = arguments.args
    test(args)
