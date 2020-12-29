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


class HalfCrackSqaure(MappedPDE):
    def __init__(self, args):
        self.case_name = "half_crack_square"
        super(HalfCrackSqaure, self).__init__(args)

        # self.displacements = np.concatenate((np.linspace(0, 0.08, 11), np.linspace(0.08, 0.15, 101)))
        self.displacements = np.concatenate((np.linspace(0, 0.16, 11), np.linspace(0.16, 0.18, 101)))
        # self.displacements = np.linspace(0.0, 0.3, 101)
 
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))

        self.psi_cr = 0.01

        self.mu = 1e3
        self.nu = 0.4
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)
        self.l0 = 2 * self.mesh.hmin()

        self.map_type = 'linear'
        if self.map_type == 'linear':
            self.l0 /= 2
            # self.finish_flag = True
        elif self.map_type == 'identity':
            self.finish_flag = True


    def build_mesh(self):
        self.length = 100
        self.height = 100
        self.notch_length = 6

        plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(self.length, self.height))
        # notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1), fe.Point(0, self.height / 2 - 1), fe.Point(self.notch_length, self.height / 2)])
        notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1e-10), fe.Point(0, self.height / 2 - 1e-10), fe.Point(self.length / 2, self.height / 2)])
        # notch = mshr.Polygon([fe.Point(self.length / 4, self.height / 2), fe.Point(self.length / 2, self.height / 2 - 1e-10), \
        #                       fe.Point(self.length * 3 / 4, self.height / 2), fe.Point(self.length / 2, self.height / 2 + 1e-10)])
        self.mesh = mshr.generate_mesh(plate - notch, 50)

        # self.mesh = da.RectangleMesh(fe.Point(0, 0), fe.Point(self.length, self.height), 40, 40, diagonal="crossed")
 

        # Add dolfin-adjoint dependency
        self.mesh  = create_overloaded_object(self.mesh)

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

        class Notch(fe.SubDomain):
            def inside(self, x, on_boundary):
                return  fe.near(x[1], height / 2) and x[0] < length / 2


        # class Notch(fe.SubDomain):
        #     def inside(self, x, on_boundary):
        #         return  x[1] < height / 2 +  height / 40 and x[1] > height / 2 -  height / 40  and x[0] < length / 2


        self.lower = Lower()
        self.upper = Upper()
        self.corner = Corner()
        self.notch = Notch()
        self.left = Left()
        self.right = Right()


    def set_bcs_monolithic(self):
        self.upper.mark(self.boundaries, 1)

        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.M.sub(0).sub(1), da.Constant(0),  self.lower)
        BC_u_upper = da.DirichletBC(self.M.sub(0).sub(1), self.presLoad,  self.upper)
        BC_u_corner = da.DirichletBC(self.M.sub(0).sub(0), da.Constant(0.0), self.corner, method='pointwise')
        BC_d_notch = fe.DirichletBC(self.M.sub(1), da.Constant(1.), self.notch, method='pointwise')

        self.BC = [BC_u_lower, BC_u_upper, BC_u_corner]
        
        # self.presLoad = da.Expression((0, "t"), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.M.sub(0), da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.M.sub(0), self.presLoad, self.upper) 
        # self.BC = [BC_u_lower, BC_u_upper] 


    def build_weak_form_monolithic(self):
        self.x_hat = fe.variable(fe.SpatialCoordinate(self.mesh))
        self.x = map_function_ufl(self.x_hat, self.control_points, self.impact_radii, self.map_type, self.boundary_info)  
        self.grad_gamma = fe.diff(self.x, self.x_hat)

        def mfem_grad_wrapper(grad):
            def mfem_grad(u):
                return fe.dot(grad(u), fe.inv(self.grad_gamma))
            return mfem_grad

        self.mfem_grad = mfem_grad_wrapper(fe.grad)

        self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)

        sigma_plus = cauchy_stress_plus(strain(self.mfem_grad(self.x_new)), self.psi_plus)
        sigma_minus = cauchy_stress_minus(strain(self.mfem_grad(self.x_new)), self.psi_minus)

        G_u = (g_d(self.d_new) * fe.inner(sigma_plus, strain(self.mfem_grad(self.eta))) \
            + fe.inner(sigma_minus, strain(self.mfem_grad(self.eta)))) * fe.det(self.grad_gamma) * fe.dx

        G_d = (self.H_new * self.zeta * g_d_prime(self.d_new, g_d) \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx

        # G_d = (history(self.H_old, self.psi_plus(strain(fe.grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        G_d += 0.1 * (self.d_new - self.d_pre) * self.zeta * fe.det(self.grad_gamma) * fe.dx

        self.G = G_u + G_d



    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)

        # self.presLoad = da.Expression("t", t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.lower)
        # BC_u_upper = da.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        # BC_u_corner = da.DirichletBC(self.U.sub(0), da.Constant(0.0), self.corner, method='pointwise')
        # BC_d_notch = fe.DirichletBC(self.W, fe.Constant(1.), self.notch, method='pointwise')
        # self.BC_u = [BC_u_lower, BC_u_upper, BC_u_corner]
        # self.BC_d = []

        # self.presLoad = da.Expression((0, "t"), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        # BC_u_left = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.left)
        # BC_u_right = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.right)
        # self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        # self.BC_d = []


        self.presLoad = da.Expression(("t", 0), t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        BC_u_left = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.left)
        BC_u_right = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.right)
        self.BC_u = [BC_u_lower, BC_u_upper] 
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

        self.psi_plus = partial(psi_plus_linear_elasticity_model_C, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_C, lamda=self.lamda, mu=self.mu)

        sigma_plus = cauchy_stress_plus(strain(self.mfem_grad(self.x_new)), self.psi_plus)
        sigma_minus = cauchy_stress_minus(strain(self.mfem_grad(self.x_new)), self.psi_minus)

        self.G_u = (g_d(self.d_new) * fe.inner(sigma_plus, strain(self.mfem_grad(self.eta))) \
            + fe.inner(sigma_minus, strain(self.mfem_grad(self.eta)))) * fe.det(self.grad_gamma) * fe.dx

        self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx
 
        # self.G_d = (history(self.H_old, self.psi_plus(strain(self.mfem_grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx

        # g_c = 0.01
        # self.G_d = (self.psi_plus(strain(self.mfem_grad(self.x_new))) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + g_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx

        # g_c = 0.01
        # self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + g_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx

        # self.G_d += 0.5 * (self.d_new - self.d_pre) * self.zeta * fe.det(self.grad_gamma) * fe.dx


    def update_history(self):
        psi_new = self.psi_plus(strain(self.mfem_grad(self.x_new)))  
        return psi_new




def test(args):
    pde_hc = HalfCrackSqaure(args)
    # pde_hc.monolithic_solve()
    pde_hc.staggered_solve()
 

if __name__ == '__main__':
    args = arguments.args
    test(args)
