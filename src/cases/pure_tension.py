import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
from ..pde import MappedPDE
from .. import arguments
from ..constitutive import *
from ..mfem import map_function_ufl


class PureTension(MappedPDE):
    def __init__(self, args):
        self.case_name = "pure_tension"
        self.solution_scheme = 'explicit'
        # self.map_type = 'identity'
        # self.local_refinement_iteration = 1
        self.map_type = args.map_type
        self.local_refinement_iteration = args.local_refinement_iteration

        super(PureTension, self).__init__(args)

        self.displacements = 1e-2*np.concatenate((np.linspace(0, 0.6, 21), np.linspace(0.6, 0.7, 301)))
        # self.displacements = 1e-2*np.linspace(0.0, 0.7, 11)
 
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))
 
        self.mu = 80.77        
        self.lamda = 121.15
        self.G_c = 2.7*1e-3
        self.psi_cr = 0.

        # self.l0 = 5 * (self.mesh.hmin() + self.mesh.hmax())
        self.l0 = 0.02
        print(self.mesh.hmax())
        print(self.mesh.hmin())        
        print("self.l0 is {}".format(self.l0))

        if self.map_type == 'linear' or self.map_type == 'smooth':
            self.map_flag = True
        elif self.map_type == 'identity':
            self.map_flag = False
        self.finish_flag = True

        self.rho_default = 15.
        self.initialize_control_points_and_impact_radii()


    def initialize_control_points_and_impact_radii(self):
        self.control_points = np.array([[self.length/2., self.height/2.], [self.length, self.height/2.]])
        self.impact_radii = np.array([self.height/4, self.height/4])


    def build_mesh(self):
        self.length = 1.
        self.height = 1.
 
        # plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(self.length, self.height))
        # notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1e-10), fe.Point(0, self.height / 2 - 1e-10), fe.Point(self.length / 2, self.height / 2)])
        # domain = plate - notch

        domain = mshr.Polygon([fe.Point(self.length / 2, self.height / 2), 
                               fe.Point(0, self.height / 2 - 1e-10), 
                               fe.Point(0, 0),
                               fe.Point(self.length, 0),
                               fe.Point(self.length, self.height/2),
                               fe.Point(self.length, self.height),
                               fe.Point(0, self.height),
                               fe.Point(0, self.height/2 + 1e-10)])

        resolution = 50 * np.power(2, self.local_refinement_iteration)
        self.mesh = mshr.generate_mesh(domain, resolution)

        length = self.length
        height = self.height

        class Lower(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0)

        class Upper(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], height)

        class Corner(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return fe.near(x[0], 0) and fe.near(x[1], 0)

        self.lower = Lower()
        self.upper = Upper()
        self.corner = Corner()


    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)

        # self.presLoad = fe.Expression("t", t=0.0, degree=1)
        # BC_u_lower = fe.DirichletBC(self.U.sub(1), fe.Constant(0),  self.lower)
        # BC_u_upper = fe.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        # BC_u_corner = fe.DirichletBC(self.U.sub(0), fe.Constant(0.0), self.corner, method='pointwise')
        # self.BC_u = [BC_u_lower, BC_u_upper, BC_u_corner]
        # self.BC_d = []

        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_lower = fe.DirichletBC(self.U, fe.Constant((0, 0)),  self.lower)
        BC_u_upper = fe.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        self.BC_u = [BC_u_lower, BC_u_upper]
        self.BC_d = []


if __name__ == '__main__':
    args = arguments.args
    test(args)
