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


class PureTension(MappedPDE):
    def __init__(self, args):
        self.case_name = "pure_tension"
        self.solution_scheme = 'explicit'
        super(PureTension, self).__init__(args)

        self.displacements = 1e-2*np.concatenate((np.linspace(0, 0.5, 11), np.linspace(0.5, 0.7, 101)))
        # self.displacements = 1e-2*np.linspace(0.0, 0.7, 21)
 
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))
 
        self.mu = 80.77        
        self.lamda = 121.15
        self.G_c = 2.7*1e-3
        self.psi_cr = 0.

        # self.l0 = 5 * (self.mesh.hmin() + self.mesh.hmax())
        # self.l0 = 2 * (self.mesh.hmin() + self.mesh.hmax())

        self.l0 = 0.04780085687755729/2

        print(self.mesh.hmax())
        print(self.mesh.hmin())
        self.l0 = 2*(self.mesh.hmax() + self.mesh.hmin())
        print("self.l0 is {}".format(self.l0))

        self.map_type = 'identity'

        if self.map_type == 'linear' or self.map_type == 'smooth':
            self.l0 /= 2

        self.finish_flag = True

        self.rho_default = 15.
        self.d_integral_interval = 1.5*self.rho_default
        self.initialize_control_points_and_impact_radii()


    def initialize_control_points_and_impact_radii(self):
        self.control_points = np.array([[self.length/2., self.height/2.], [self.length, self.height/2.]])
        self.impact_radii = np.array([self.height/4, self.height/4])


    def build_mesh(self):
        self.length = 1.
        self.height = 1.
 
        plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(self.length, self.height))
        notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1e-10), fe.Point(0, self.height / 2 - 1e-10), fe.Point(self.length / 2, self.height / 2)])

        self.mesh = mshr.generate_mesh(plate - notch, 100)

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

        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_lower = fe.DirichletBC(self.U.sub(1), fe.Constant(0),  self.lower)
        BC_u_upper = fe.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        BC_u_corner = fe.DirichletBC(self.U.sub(0), fe.Constant(0.0), self.corner, method='pointwise')
        self.BC_u = [BC_u_lower, BC_u_upper, BC_u_corner]
        self.BC_d = []

        # self.presLoad = fe.Expression((0, "t"), t=0.0, degree=1)
        # BC_u_lower = fe.DirichletBC(self.U, fe.Constant((0., 0.)), self.lower)
        # BC_u_upper = fe.DirichletBC(self.U, self.presLoad, self.upper) 
        # BC_u_left = fe.DirichletBC(self.U.sub(0), fe.Constant(0),  self.left)
        # BC_u_right = fe.DirichletBC(self.U.sub(0), fe.Constant(0),  self.right)
        # self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        # self.BC_d = []

        # self.presLoad = fe.Expression("t", t=0.0, degree=1)
        # BC_u_lower = fe.DirichletBC(self.U, fe.Constant((0., 0.)), self.lower)
        # BC_u_upper = fe.DirichletBC(self.U.sub(1), self.presLoad, self.upper) 
        # # BC_u_left = fe.DirichletBC(self.U.sub(1), fe.Constant(0),  self.left)
        # # BC_u_right = fe.DirichletBC(self.U.sub(1), fe.Constant(0),  self.right)
        # self.BC_u = [BC_u_lower, BC_u_upper] 
        # self.BC_d = []


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

        if self.solution_scheme == 'explicit':
            self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx
            # self.G_d = (self.psi_plus(strain(self.mfem_grad(self.x_new))) * self.zeta * g_d_prime(self.d_new, g_d) \
            #         + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx
        else:
            self.G_d = (history(self.H_old, self.psi_plus(strain(self.mfem_grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx

    def update_history(self):
        psi_new = self.psi_plus(strain(self.mfem_grad(self.x_new)))  
        return psi_new


def test(args):
    pde = PureTension(args)
    pde.staggered_solve()
    plt.show()
 

if __name__ == '__main__':
    args = arguments.args
    test(args)
