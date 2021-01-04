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


class LShape(MappedPDE):
    def __init__(self, args):
        self.case_name = "L_shape"
        super(LShape, self).__init__(args)

        # self.displacements = np.concatenate((np.linspace(0, 0.08, 11), np.linspace(0.08, 0.15, 101)))
        # self.displacements = 1e-2*np.concatenate((np.linspace(0, 0.10, 11), np.linspace(0.10, 0.15, 51)))

        self.displacements = 1e-2*np.concatenate((np.linspace(0, 0.10, 11), 
                                                  np.linspace(0.10, 0.15, 31),
                                                  np.linspace(0.15, -0.05, 31),
                                                  np.linspace(-0.05, 0.3, 41)))
 

        # self.displacements = 1e-2*np.linspace(0.0, 0.15, 51)
 
        self.relaxation_parameters = np.linspace(1., 1., len(self.displacements))
 
        self.psi_cr = 1e-7

        self.E = 1e5
        self.nu = 0.3
        self.mu = self.E / (2 * (1 + self.nu))
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)

        self.l0 = 2 * self.mesh.hmin()

        self.map_type = 'identity'
        if self.map_type == 'linear':
            self.l0 /= 2
            # self.finish_flag = True
        elif self.map_type == 'identity':
            self.finish_flag = True

        self.rho_default = 80.
        self.d_integral_interval = 1.5*self.rho_default
        self.initialize_control_points_and_impact_radii()


    def initialize_control_points_and_impact_radii(self):
        self.control_points = []
        self.impact_radii = []
        control_points = np.asarray([[self.length/2, self.height/2]])
        for new_tip_point in control_points:
            self.compute_impact_radii(new_tip_point)


    def build_mesh(self):
        self.length = 500
        self.height = 500
        self.segment = 30
 
        rectangle_large = mshr.Rectangle(fe.Point(0, 0), fe.Point(self.length, self.height))
        rectangle_small = mshr.Rectangle(fe.Point(self.length/2., 0), fe.Point(self.length, self.height/2.))

        self.mesh = mshr.generate_mesh(rectangle_large - rectangle_small, 50)
        # self.mesh = da.RectangleMesh(fe.Point(0, 0), fe.Point(self.length, self.height), 40, 40, diagonal="crossed")
 
        # Add dolfin-adjoint dependency
        self.mesh  = create_overloaded_object(self.mesh)

        length = self.length
        height = self.height
        segment = self.segment

        class Lower(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0)

        class Upper(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], height)

        class Segment(fe.SubDomain):
            def inside(self, x, on_boundary):
                return  fe.near(x[1], height/2.) and x[0] >= length - segment
              
        self.lower = Lower()
        self.upper = Upper()
        self.segment = Segment()
  
 
    def set_bcs_staggered(self):
        self.lower.mark(self.boundaries, 1)
        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)),  self.lower)
        BC_u_segment = da.DirichletBC(self.U.sub(1), self.presLoad,  self.segment, method="pointwise")
        self.BC_u = [BC_u_lower, BC_u_segment]
        self.BC_d = []

  
    def build_weak_form_staggered(self): 
        self.x_hat = fe.variable(fe.SpatialCoordinate(self.mesh))
        self.x = map_function_ufl(self.x_hat, self.control_points, self.impact_radii, self.map_type, self.boundary_info)  
        self.grad_gamma = fe.diff(self.x, self.x_hat)

        def mfem_grad_wrapper(grad):
            def mfem_grad(u):
                return fe.dot(grad(u), fe.inv(self.grad_gamma))
            return mfem_grad

        self.mfem_grad = mfem_grad_wrapper(fe.grad)

        # A special note (Tianju): We hope to use Model C, but Newton solver fails without the initial guess by Model A 
        if self.i < 2:
            self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
            self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        else:
            self.psi_plus = partial(psi_plus_linear_elasticity_model_C, lamda=self.lamda, mu=self.mu)
            self.psi_minus = partial(psi_minus_linear_elasticity_model_C, lamda=self.lamda, mu=self.mu)
            print("use model C")

        sigma_plus = cauchy_stress_plus(strain(self.mfem_grad(self.x_new)), self.psi_plus)
        sigma_minus = cauchy_stress_minus(strain(self.mfem_grad(self.x_new)), self.psi_minus)

        self.G_u = (g_d(self.d_new) * fe.inner(sigma_plus, strain(self.mfem_grad(self.eta))) \
            + fe.inner(sigma_minus, strain(self.mfem_grad(self.eta)))) * fe.det(self.grad_gamma) * fe.dx

        self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
                + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx


    def update_history(self):
        psi_new = self.psi_plus(strain(self.mfem_grad(self.x_new)))  
        return psi_new


    def call_back(self):
        if self.i < 3:
            self.update_weak_form = True


def test(args):
    pde = LShape(args)
    pde.staggered_solve()
    plt.show()
 

if __name__ == '__main__':
    args = arguments.args
    test(args)
