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
        self.solution_scheme = 'explicit'
        self.map_type = 'smooth'
        self.local_refinement_iteration = 2
        # self.map_type = args.map_type
        # self.local_refinement_iteration = args.local_refinement_iteration
        super(Tip, self).__init__(args)

        self.display_intermediate_results = False

        # Not really useful, just to override base class member variables
        self.displacements = [1] 
        self.relaxation_parameters = np.linspace(1., 1., len(self.displacements))
        self.presLoad = fe.Expression("t", t=0, degree=1)
 
        self.mu = 80.77        
        self.lamda = 121.15
        self.G_c = 2.7*1e-3
        self.psi_cr = 0.

        self.l0 = 0.1

        if self.map_type == 'linear' or self.map_type == 'smooth':
            self.map_flag = True
        elif self.map_type == 'identity':
            self.map_flag = False
        self.finish_flag = True

        self.initialize_control_points_and_impact_radii()


    def initialize_control_points_and_impact_radii(self):
        self.control_points = np.array([[self.length/2., self.height/2.], [self.length, self.height/2.]])
        self.impact_radii = np.array([self.height/4, self.height/4])


    def build_mesh(self):
        self.length = 1.
        self.height = 1.
        plate = mshr.Rectangle(fe.Point(0., 0.), fe.Point(self.length, self.height))
        notch = mshr.Polygon([fe.Point(0., self.height/2. + 1e-10), fe.Point(0., self.height/2. - 1e-10), fe.Point(self.length/2., self.height/2.)])

        resolution = 50 * np.power(2, self.local_refinement_iteration)
        self.mesh = mshr.generate_mesh(plate - notch, resolution)


    def set_bcs_staggered(self):
        self.fix_load = fe.Constant(1e-2)
        compute_analytical_solutions = self.compute_analytical_solutions_fully_broken
        control_points = self.control_points
        impact_radii  = self.impact_radii
        boundary_info = self.boundary_info
        map_type = self.map_type
        height = self.height

        class DisplacementExpression(fe.UserExpression):
            def eval(self, values, x_hat):
                x = map_function_normal(x_hat, control_points, impact_radii, map_type, boundary_info) 
                u_exact, _ = compute_analytical_solutions(x)
                values[0] = float(u_exact[0])
                values[1] = float(u_exact[1])

            def value_shape(self):
                return (2,)

        class Lower(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0)

        class Upper(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], height)

        self.lower = Lower()
        self.upper = Upper()

        self.disp_exp =  DisplacementExpression()
        BC_u_lower = fe.DirichletBC(self.U, self.disp_exp, self.lower)
        BC_u_upper = fe.DirichletBC(self.U, self.disp_exp, self.upper)
        self.BC_u = [BC_u_lower, BC_u_upper]
        self.BC_d = []


    def build_weak_form_staggered(self):
        self.x_hat = fe.variable(fe.SpatialCoordinate(self.mesh))
        self.x = fe.variable(map_function_ufl(self.x_hat, self.control_points, self.impact_radii, self.map_type, self.boundary_info))
        self.grad_gamma = fe.diff(self.x, self.x_hat)

        def mfem_grad_wrapper(grad):
            def mfem_grad(u):
                return fe.dot(grad(u), fe.inv(self.grad_gamma))
            return mfem_grad

        self.mfem_grad = mfem_grad_wrapper(fe.grad)

        self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)

        sigma_plus = cauchy_stress_plus(strain(self.mfem_grad(self.x_new)), self.psi_plus)
        sigma_minus = cauchy_stress_minus(strain(self.mfem_grad(self.x_new)), self.psi_minus)

        self.u_exact, self.d_exact = self.compute_analytical_solutions_fully_broken(self.x)
 
        self.G_u = (g_d(self.d_exact) * fe.inner(sigma_plus, strain(self.mfem_grad(self.eta))) \
            + fe.inner(sigma_minus, strain(self.mfem_grad(self.eta)))) * fe.det(self.grad_gamma) * fe.dx

        self.G_d = (self.d_new - self.d_exact) * self.zeta * fe.det(self.grad_gamma) * fe.dx

        self.u_initial = self.nonzero_initial_guess(self.x)
        self.x_new.assign(fe.project(self.u_initial, self.U))


    def nonzero_initial_guess(self, x):
        x1 = x[0]
        x2 = x[1]
        u1 = fe.Constant(0.)
        u2 = self.fix_load * x2 / self.height 
        u_initial = fe.as_vector([u1, u2])
        return u_initial


    def compute_analytical_solutions_fully_broken(self, x):
        x1 = x[0]
        x2 = x[1]
        u1 = fe.Constant(0.)
        u2 = fe.conditional(fe.gt(x2, self.height/2.), self.fix_load, fe.Constant(0.))
        u_exact = fe.as_vector([u1, u2])
        distance_field, _ = distance_function_segments_ufl(x, self.control_points, self.impact_radii)
        d_exact = fe.exp(-distance_field/(self.l0))
        return u_exact, d_exact


    def energy_norm(self, u):
        psi_plus = self.psi_plus(strain(self.mfem_grad(u)))
        psi_minus = self.psi_minus(strain(self.mfem_grad(u)))
        return np.sqrt(float(fe.assemble((g_d(self.d_exact) * psi_plus + psi_minus) * fe.det(self.grad_gamma) * fe.dx)))


    def evaluate_errors(self): 
        print("Evaluate L2 errors...")
        u_error_l2 = np.sqrt(float(fe.assemble(fe.inner(self.x_new - self.u_exact, self.x_new - self.u_exact) * fe.det(self.grad_gamma) * fe.dx)))

        u_error_semi_h1 = np.sqrt(float(fe.assemble(fe.inner(self.mfem_grad(self.x_new - self.u_exact), \
                                                             self.mfem_grad(self.x_new - self.u_exact)) * fe.det(self.grad_gamma) * fe.dx)))
       
        d_error = np.sqrt(float(fe.assemble((self.d_new - self.d_exact)**2 * fe.det(self.grad_gamma) * fe.dx)))
        print("Displacement error l2 is {}".format(u_error_l2))
        print("Displacement error semi h1 is {}".format(u_error_semi_h1))
        print("Damage error is {}".format(d_error))

        u_energy_error = self.energy_norm(self.x_new - self.u_exact)
        print("Displacement error energy_norm is {}".format(u_energy_error))        


    # Override base class method
    def save_data_in_loop(self):
        pass

    def create_custom_xdmf_files(self):
        pass

    def show_force_displacement(self):
        pass


def test(args):
    pde_tip = Tip(args)
    pde_tip.staggered_solve()
    pde_tip.evaluate_errors()


if __name__ == '__main__':
    args = arguments.args
    test(args)
