import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
from ..pde import MappedPDE
from .. import arguments
from ..constitutive import *
from ..mfem import map_function_ufl, map_function_normal
 

class ThreePointBending(MappedPDE):
    def __init__(self, args):
        self.case_name = "three_point_bending"
        self.solution_scheme = 'explicit'
        # self.map_type = 'smooth'
        # self.local_refinement_iteration = 0
        self.map_type = args.map_type
        self.local_refinement_iteration = args.local_refinement_iteration
        super(ThreePointBending, self).__init__(args)

        self.displacements = -1e-1*np.concatenate((np.linspace(0, 0.5, 21),
                                                   np.linspace(0.5, 0.8, 201)))

        # self.displacements = -1e-1*np.linspace(0.0, 1., 21)
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))
 
        self.mu = 8.       
        self.lamda = 12.
        self.G_c = 5.*1e-4
        self.psi_cr = 0.

        # self.l0 = (self.mesh.hmin() + self.mesh.hmax())
        self.l0 = 0.08

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
        radius = np.sqrt( (self.notch_length/2.)**2 + self.notch_height**2 )

        # self.control_points = np.array([[self.length/2., self.notch_height], [self.length/2., 2 * self.height]])
        self.control_points = np.array([[self.length/2., self.notch_height], [self.length/2., self.height - 1.2*radius]])

        self.impact_radii = np.array([radius, radius])


    def build_mesh(self):
        self.length = 8.
        self.height = 2.
        self.notch_length = 0.2
        self.notch_height = 0.4

        domain = mshr.Polygon([fe.Point(0., 0.), 
                  fe.Point(self.length/2. - self.notch_length/2., 0.),
                  fe.Point(self.length/2., self.notch_height),
                  fe.Point(self.length/2. + self.notch_length/2., 0.),
                  fe.Point(self.length, 0.),
                  fe.Point(self.length, self.height),
                  fe.Point(self.length/2. + self.notch_length/2., self.height),
                  fe.Point(self.length/2., self.height),
                  fe.Point(self.length/2. - self.notch_length/2., self.height),
                  fe.Point(0., self.height)])

        # resolution = 100 if self.local_refinement_iteration == 0 else 200
        # self.mesh = mshr.generate_mesh(domain, resolution)

        self.mesh = mshr.generate_mesh(domain, 100)
        for i in range(self.local_refinement_iteration):
            cell_markers = fe.MeshFunction('bool', self.mesh, self.mesh.topology().dim())
            cell_markers.set_all(False)
            for cell in fe.cells(self.mesh):
                p = cell.midpoint()
                if  p[0] > 14./32.*self.length and p[0] < 18./32.*self.length and p[1] < self.height - self.notch_height:
                    cell_markers[cell] = True
            self.mesh = fe.refine(self.mesh, cell_markers)

        length = self.length
        height = self.height
        notch_length = self.notch_length

        class Upper(fe.SubDomain):
            def inside(self, x, on_boundary):
                # return on_boundary
                return on_boundary and fe.near(x[1], height) 


        class LeftCorner(fe.SubDomain):
            def inside(self, x, on_boundary):
                return fe.near(x[0], 0.) and fe.near(x[1], 0.)

        class RightCorner(fe.SubDomain):
            def inside(self, x, on_boundary):
                return fe.near(x[0], length) and fe.near(x[1], 0.)

        class MiddlePoint(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                # return fe.near(x[0], length/2.) and fe.near(x[1], height)
                return x[0] > length/2. - notch_length/2.  and  x[0] < length/2. + notch_length/2. and fe.near(x[1], height)

        self.upper = Upper()
        self.left = LeftCorner()
        self.right = RightCorner()
        self.middle = MiddlePoint()
 

    def set_bcs_staggered(self):
        self.middle.mark(self.boundaries, 1)
        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_left = fe.DirichletBC(self.U, fe.Constant((0., 0.)), self.left, method='pointwise')
        BC_u_right = fe.DirichletBC(self.U.sub(1), fe.Constant(0.), self.right, method='pointwise')
        BC_u_middle = fe.DirichletBC(self.U.sub(1), self.presLoad, self.middle, method='pointwise')
        self.BC_u = [BC_u_left, BC_u_right, BC_u_middle]
        self.BC_d = []


if __name__ == '__main__':
    args = arguments.args
    test(args)
