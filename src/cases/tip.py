import fenics as fe
import ufl
import numpy as np
import mshr
import matplotlib.pyplot as plt
from functools import partial
from ..pde import MappedPDE
from .. import arguments
from ..constitutive import *
from ..mfem import distance_function_segments_ufl, map_function_ufl, map_function_normal


class Tip(MappedPDE):
    def __init__(self, args):
        self.case_name = "tip"
        super(Tip, self).__init__(args)

        self.displacements = [1] # Not really useful
        self.relaxation_parameters = np.linspace(1., 1., len(self.displacements))
 
        self.psi_cr = 1e-7

        self.E = 1e5
        self.nu = 0.3
        self.mu = self.E / (2 * (1 + self.nu))
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)
        self.K_I = 1.
        self.kappa = 3 - 4 * self.nu

        self.l0 = 2 * self.mesh.hmin()
 
        self.l0 = 5

        self.map_type = 'linear'
        self.finish_flag = True
        self.initialize_control_points_and_impact_radii()


    def initialize_control_points_and_impact_radii(self):
        self.control_points = np.array([[self.length/2., self.height/2.], [self.length, self.height/2.]])
        self.impact_radii = np.array([self.height/4, self.height/4])


    def build_mesh(self):
        self.length = 100
        self.height = 100
        plate = mshr.Rectangle(fe.Point(-self.length/2, -self.height/2), fe.Point(self.length/2, self.height/2))
        notch = mshr.Polygon([fe.Point(-self.length/2, 1e-10), fe.Point(-self.length/2, -1e-10), fe.Point(0, 0)])
        self.mesh = mshr.generate_mesh(plate - notch, 100)


    def set_bcs_staggered(self):

        compute_analytical_solutions = self.compute_analytical_solutions_fully_broken
        control_points = self.control_points
        impact_radii  = self.impact_radii
        boundary_info = self.boundary_info
        map_type = self.map_type


        class DisplacementExpression(fe.UserExpression):
            def eval(self, values, x_hat):
                x = map_function_normal(x_hat, control_points, impact_radii, map_type, boundary_info) 
                u_exact, _ = compute_analytical_solutions(x)
                values[0] = float(u_exact[0])
                values[1] = float(u_exact[1])

            def value_shape(self):
                return (2,)

        class DamageExpression(fe.UserExpression):
            def eval(self, values, x_hat):
                x = map_function_normal(x_hat, control_points, impact_radii, map_type, boundary_info) 
                _, d_exact = compute_analytical_solutions(x)
                values[0] = float(d_exact)

            def value_shape(self):
                return ()

        def boundary(x, on_boundary):
            return on_boundary


        height = self.height

        class Lower(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], -height/2)

        class Upper(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], height/2)

        self.lower = Lower()
        self.upper = Upper()


        # Remarks(Tianju): Yet another bug of FEniCS
        # If we do the following:
        # BC_u = fe.DirichletBC(self.U, DisplacementExpression(), boundary)
        # BC_d = fe.DirichletBC(self.W, DamageExpression(), boundary)
        # FEniCS throws an error (which it shouldn't throw)
        # The bug is reported here: https://bitbucket.org/fenics-project/dolfin/issues/1070/userexpression-instantiated-inside-python
        # self.disp_exp =  DisplacementExpression()
        # self.damage_exp = DamageExpression()
        # BC_u = fe.DirichletBC(self.U, self.disp_exp, boundary)
        # BC_d = fe.DirichletBC(self.W, self.damage_exp, boundary)
        # self.BC_u = [BC_u]
        # self.BC_d = [BC_d]

        self.disp_exp =  DisplacementExpression()
        BC_u_lower = fe.DirichletBC(self.U, self.disp_exp, self.lower)
        BC_u_upper = fe.DirichletBC(self.U, self.disp_exp, self.upper)

        self.BC_u = [BC_u_lower, BC_u_upper]
        self.BC_d = []


    def build_weak_form_staggered(self):
        self.x_hat = fe.variable(fe.SpatialCoordinate(self.mesh))
        self.x = fe.variable(map_function_ufl(self.x_hat, self.control_points, self.impact_radii, self.map_type, self.boundary_info))
        self.grad_gamma = fe.diff(self.x, self.x_hat)

        def mfem_div(A):
            # Remarks(Tianju): There is a bug in the function ufl.operators.contraction 
            # Do the following changes to the source code to fix the bug.
            # https://bitbucket.org/fenics-project/ufl/issues/111/tensor-contraction-function-is-broken
            # Otherwise it throws an error (which it shouldn't throw)
            grad_A = fe.grad(A)
            grad_gamma_invT = fe.inv(self.grad_gamma).T
            indices1 = [ufl.rank(grad_A) - 2, ufl.rank(grad_A) - 1]
            indices2 = [0, 1]
            return ufl.operators.contraction(grad_A, indices1, grad_gamma_invT, indices2)
     
        def mfem_grad(A):
            return fe.dot(fe.grad(A), fe.inv(self.grad_gamma))

        self.mfem_grad = mfem_grad
        self.mfem_div = mfem_div

        self.psi_plus_func = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus_func = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)

        sigma_plus = cauchy_stress_plus(strain(self.mfem_grad(self.x_new)), self.psi_plus_func)
        sigma_minus = cauchy_stress_minus(strain(self.mfem_grad(self.x_new)), self.psi_minus_func)
        psi_plus =  self.psi_plus_func(strain(self.mfem_grad(self.x_new)))

        self.u_exact, self.d_exact = self.compute_analytical_solutions_fully_broken(self.x)



        sigma_plus_exact = cauchy_stress_plus(strain(self.mfem_grad(self.u_exact)), self.psi_plus_func)
        sigma_minus_exact = cauchy_stress_minus(strain(self.mfem_grad(self.u_exact)), self.psi_minus_func)
        psi_plus_exact = self.psi_plus_func(strain(self.mfem_grad(self.u_exact)))

        body_force_u = -mfem_div(g_d(self.d_exact)*sigma_plus_exact + sigma_minus_exact)

        body_force_d = history(self.H_old, psi_plus_exact, self.psi_cr) * g_d_prime(self.d_exact, g_d) \
                     + 2 * self.psi_cr * (self.d_exact - self.l0**2 * mfem_div(mfem_grad(self.d_exact))) 

        # self.G_u = (fe.inner(sigma_plus, strain(self.mfem_grad(self.eta))) \
        #     + fe.inner(sigma_minus, strain(self.mfem_grad(self.eta)))) * fe.det(self.grad_gamma) * fe.dx

        self.G_u = (g_d(self.d_exact) * fe.inner(sigma_plus, strain(self.mfem_grad(self.eta))) \
            + fe.inner(sigma_minus, strain(self.mfem_grad(self.eta)))) * fe.det(self.grad_gamma) * fe.dx

        # self.G_u = (g_d(self.d_new) * fe.inner(sigma_plus, strain(self.mfem_grad(self.eta))) \
        #     + fe.inner(sigma_minus, strain(self.mfem_grad(self.eta)))) * fe.det(self.grad_gamma) * fe.dx

        # self.G_d = (history(self.H_old, psi_plus, self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
        #         + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx

        self.G_d = (self.d_new - self.d_exact) * self.zeta * fe.det(self.grad_gamma) * fe.dx

        # self.G_u += -fe.dot(body_force_u, self.eta)* fe.det(self.grad_gamma) * fe.dx

        # self.G_d += -body_force_d * self.zeta * fe.det(self.grad_gamma) * fe.dx


    # Deprecated. We will not use this function.
    def compute_analytical_solutions_model_I(self, x):
        x1 = x[0]
        x2 = x[1]
        theta = ufl.atan_2(x2, x1)
        r = fe.sqrt(x1**2 + x2**2)
        u1 = fe.sqrt(r / (2 * np.pi) ) * self.K_I / (2 * self.mu) * fe.cos(theta / 2) * (self.kappa - fe.cos(theta))
        u2 = fe.sqrt(r / (2 * np.pi) ) * self.K_I / (2 * self.mu) * fe.sin(theta / 2) * (self.kappa - fe.cos(theta))
        u_exact = fe.as_vector([u1, u2])
        d_exact = fe.exp(-r/(self.l0))
        # d_exact = fe.conditional(fe.gt(r, self.radius), fe.exp(-(r - self.radius) / (self.l0)), 1.)
        return u_exact, d_exact


    def compute_analytical_solutions_fully_broken(self, x):
        x1 = x[0]
        x2 = x[1]
        u1 = fe.Constant(0.)
        u2 = fe.conditional(fe.gt(x2, 0.), fe.Constant(1.), fe.Constant(0.))
        u_exact = fe.as_vector([u1, u2])
        distance_field, _ = distance_function_segments_ufl(x, self.control_points, self.impact_radii)
        d_exact = fe.exp(-distance_field/(self.l0))
        return u_exact, d_exact


    def evaluate_errors(self): 
        print("Evaluate L2 errors...")
        u_error_l2 = np.sqrt(float(fe.assemble(fe.inner(self.x_new - self.u_exact, self.x_new - self.u_exact) * fe.det(self.grad_gamma) * fe.dx)))

        u_error_semi_h1 = np.sqrt(float(fe.assemble(fe.inner(self.mfem_grad(self.x_new - self.u_exact), \
                                                             self.mfem_grad(self.x_new - self.u_exact)) * fe.det(self.grad_gamma) * fe.dx)))
       
        d_error = np.sqrt(float(fe.assemble((self.d_new - self.d_exact)**2 * fe.det(self.grad_gamma) * fe.dx)))
        print("Displacement error l2 is {}".format(u_error_l2))
        print("Displacement error semi h1 is {}".format(u_error_semi_h1))
        print("Damage error is {}".format(d_error))


def test(args):
    pde_tip = Tip(args)
    pde_tip.staggered_solve()
    pde_tip.evaluate_errors()
    # plt.show()

if __name__ == '__main__':
    args = arguments.args
    test(args)
