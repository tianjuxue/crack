import fenics as fe
import numpy as np
import mshr
import matplotlib.pyplot as plt
from functools import partial
from ..pde import PDE
from .. import arguments
from ..constitutive import *


class DoubleCircles(PDE):
    def __init__(self, args):
        self.case_name = "circular_holes"
        super(DoubleCircles, self).__init__(args)
        # self.displacements = np.concatenate((np.linspace(1, 11, 6), 
        #     np.linspace(12, 26.5, 30), np.linspace(27, 40, 53)))
        # self.relaxation_parameters = np.concatenate((np.linspace(0.2, 0.2, 11), 
        #     np.linspace(0.1, 0.1, 24), np.linspace(0.02, 0.02, 54)))

        self.displacements = np.linspace(1, 11, 6)
        self.relaxation_parameters = np.linspace(0.2, 0.2, 6)

        self.l0 = 1.
        self.psi_cr = 0.03
        self.mu = 0.19
        self.nu = 0.45
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)
        self.kappa = self.lamda + 2. / 3. * self.mu
        self.E = 3 * self.kappa * (1 - 2 * self.nu)
        self.beta = 2 * self.nu / (1 - 2 * self.nu)
        

    def build_mesh(self):
        length = 60
        height = 30
        radius = 5
        plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(length, height))
        circle1 = mshr.Circle(fe.Point(length/3, height/3), radius)
        circle2 = mshr.Circle(fe.Point(length*2/3, height*2/3), radius)
        material_domain = plate - circle1 - circle2
        self.mesh = mshr.generate_mesh(material_domain, 50)

        # Add dolfin-adjoint dependency
        self.mesh  = create_overloaded_object(self.mesh)

        class Left(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0)

        class Right(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], length)

        class Corner(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return fe.near(x[0], 0) and fe.near(x[1], 0)

        self.left = Left()
        self.right = Right()
        self.corner = Corner()


    def set_bcs_monolithic(self):
        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_left = da.DirichletBC(self.M.sub(0).sub(0), da.Constant(0),  self.left)
        BC_u_right = da.DirichletBC(self.M.sub(0).sub(0), self.presLoad,  self.right )
        BC_u_corner = da.DirichletBC(self.M.sub(0).sub(1), da.Constant(0), self.corner, method='pointwise')
        self.BC = [BC_u_left, BC_u_right, BC_u_corner] 
        # self.right.mark(self.boundaries, 1)


    def build_weak_form_monolithic(self):
        self.psi_plus = partial(psi_plus_Miehe, mu=self.mu, beta=self.beta)
        self.psi_minus = partial(psi_minus_Miehe, mu=self.mu, beta=self.beta)

        PK_plus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_plus)
        PK_minus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_minus)

        G_u = (g_d(self.d_new) * fe.inner(PK_plus, fe.grad(self.eta)) + fe.inner(PK_minus, fe.grad(self.eta)) )* fe.dx
        G_d = self.H_old * self.zeta * g_d_prime(self.d_new, g_d) * fe.dx \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new))) * fe.dx  
        self.G = G_u + G_d


    def set_bcs_staggered(self):
        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_left = da.DirichletBC(self.U.sub(0), da.Constant(0), self.left)
        BC_u_right = da.DirichletBC(self.U.sub(0), self.presLoad, self.right )
        BC_u_corner = da.DirichletBC(self.U.sub(1), da.Constant(0), self.corner, method='pointwise')
        self.BC_u = [BC_u_left, BC_u_right, BC_u_corner]     
        self.BC_d = []


    def build_weak_form_staggered(self):
        self.psi_plus = partial(psi_plus_Miehe, mu=self.mu, beta=self.beta)
        self.psi_minus = partial(psi_minus_Miehe, mu=self.mu, beta=self.beta)

        PK_plus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_plus)
        PK_minus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_minus)
 
        self.G_u = (g_d(self.d_new) * fe.inner(PK_plus, fe.grad(self.eta)) +  fe.inner(PK_minus, fe.grad(self.eta)))* fe.dx
        self.G_d = self.H_old * self.zeta * g_d_prime(self.d_new, g_d) * fe.dx \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new))) * fe.dx  


    def update_history(self):
        psi_new = self.psi_plus(self.I + fe.grad(self.x_new))  
        return psi_new


def test(args):
    pde_dc = DoubleCircles(args)
    pde_dc.monolithic_solve()
    # pde_dc.staggered_solve()


if __name__ == '__main__':
    args = arguments.args
    test(args)