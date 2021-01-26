import fenics as fe
import ufl
import numpy as np
import mshr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from functools import partial
from ..pde import MappedPDE
from .. import arguments
from ..constitutive import *
from ..mfem import distance_function_segments_ufl, map_function_ufl, map_function_normal


class Analysis(MappedPDE):
    def __init__(self, args):
        self.case_name = "analysis"
        self.solution_scheme = 'explicit'
        # self.map_type = 'smooth'
        # self.local_refinement_iteration = 2
        self.model_flag = args.model_flag
        self.map_type = args.map_type
        self.local_refinement_iteration = args.local_refinement_iteration
        self.psi_plus_linear_elasticity = args.psi_plus_linear_elasticity
        self.psi_minus_linear_elasticity = args.psi_minus_linear_elasticity

        super(Analysis, self).__init__(args)

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

        # print(self.mesh.hmax())
        # print(self.mesh.hmin())
        # print(1./2.*np.sqrt(2)*(1./2.)**self.total_refinement)

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

        self.mesh = fe.Mesh()
        editor = fe.MeshEditor()
        editor.open(self.mesh, 'triangle', 2, 2)
        editor.init_vertices(10)
        editor.init_cells(8)
        editor.add_vertex(0, fe.Point(0.5, 0.5))
        editor.add_vertex(1, fe.Point(1., 0.5))
        editor.add_vertex(2, fe.Point(1., 1.))
        editor.add_vertex(3, fe.Point(0.5, 1.))
        editor.add_vertex(4, fe.Point(0., 1.))
        editor.add_vertex(5, fe.Point(0., 0.5))
        editor.add_vertex(6, fe.Point(0., 0.))
        editor.add_vertex(7, fe.Point(0.5, 0.))
        editor.add_vertex(8, fe.Point(1., 0.))
        editor.add_vertex(9, fe.Point(0., 0.5))
        editor.add_cell(0, [0, 1, 3])
        editor.add_cell(1, [1, 2, 3])
        editor.add_cell(2, [0, 3, 4])
        editor.add_cell(3, [0, 4, 5])
        editor.add_cell(4, [0, 9, 7])
        editor.add_cell(5, [6, 7, 9])
        editor.add_cell(6, [0, 7, 8])
        editor.add_cell(7, [0, 8, 1])
        editor.close()

        base_refinement = 4
        self.total_refinement = base_refinement + self.local_refinement_iteration

        for i in range(self.total_refinement):
            self.mesh = fe.refine(self.mesh)


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

        self.psi_plus = partial(self.psi_plus_linear_elasticity, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(self.psi_minus_linear_elasticity, lamda=self.lamda, mu=self.mu)
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
       
        # d_error_l2 = np.sqrt(float(fe.assemble((self.d_new - self.d_exact)**2 * fe.det(self.grad_gamma) * fe.dx)))
        print("Displacement error l2 is {}".format(u_error_l2))
        print("Displacement error semi h1 is {}".format(u_error_semi_h1))
        # print("Damage error is {}".format(d_error_l2))

        self.u_energy_error = self.energy_norm(self.x_new - self.u_exact)
        print("Displacement error energy_norm is {}".format(self.u_energy_error))  

 
    def create_custom_xdmf_files(self):
        self.file_results_custom = fe.XDMFFile('data/xdmf/{}/u_refine_{}_mfem_{}_model_{}.xdmf'.format(self.case_name, 
            self.local_refinement_iteration, self.map_flag, self.model_flag))
        self.file_results_custom.parameters["functions_share_mesh"] = True


    def save_data_in_loop(self):
        self.file_results_custom.write(self.x_new, self.i)
        self.file_results_custom.write(self.d_new, self.i)
        self.file_results_custom.write(self.map_plot, self.i)

        if self.map_flag:
            np.save('data/numpy/{}/control_points.npy'.format(self.case_name), self.control_points)
            np.save('data/numpy/{}/impact_radii.npy'.format(self.case_name), self.impact_radii)
            np.save('data/numpy/{}/boundary_info.npy'.format(self.case_name), self.boundary_info)


    # For analysis purposes
    def interpolate_solution(self):
        fix_load = float(self.fix_load)
        np_x = np.asarray(self.x_new.vector())
        for i in range(len(np_x)):
            if np_x[i] < 3./8.*fix_load:
                np_x[i] = 0
            else:
                np_x[i] = fix_load

        self.x_new.vector()[:] = np_x
        self.file_results.write(self.x_new, 0)


    # Override base class method            
    def show_force_displacement(self):
        pass


# model_flag = 0, 1, 2 means Model A, Model B and Model C
def run_one_model(args, model_flag, map_type):
    u_energy_errors = []
    mesh_sizes = []

    if model_flag == 0:
        psi_plus_linear_elasticity = psi_plus_linear_elasticity_model_A
        psi_minus_linear_elasticity = psi_minus_linear_elasticity_model_A
    elif model_flag == 1:
        psi_plus_linear_elasticity = psi_plus_linear_elasticity_model_B
        psi_minus_linear_elasticity = psi_minus_linear_elasticity_model_B
    else:
        psi_plus_linear_elasticity = psi_plus_linear_elasticity_model_C
        psi_minus_linear_elasticity = psi_minus_linear_elasticity_model_C

    args.model_flag = model_flag
    args.map_type = map_type

    local_refinement_iterations = [0, 1, 2, 3]

    for local_refinement_iteration in local_refinement_iterations:
        args.local_refinement_iteration = local_refinement_iteration
        args.psi_plus_linear_elasticity = psi_plus_linear_elasticity
        args.psi_minus_linear_elasticity = psi_minus_linear_elasticity
        pde = Analysis(args)
        pde.staggered_solve()
        pde.evaluate_errors() 
        u_energy_errors.append(pde.u_energy_error)
        mesh_sizes.append(pde.mesh.hmin())

    np.save('data/numpy/{}/model_{}_map_{}.npy'.format(pde.case_name, model_flag, map_type), np.array(u_energy_errors))
    np.save('data/numpy/{}/mesh_sizes.npy'.format(pde.case_name), np.array(mesh_sizes))


def main(args):
    post_processing_flag = True

    # Default settings in case needed
    args.psi_plus_linear_elasticity = psi_plus_linear_elasticity_model_A
    args.psi_minus_linear_elasticity = psi_minus_linear_elasticity_model_A
    args.model_flag = 0
    args.map_type = 'identity'
    args.local_refinement_iteration = 0
    pde = Analysis(args)

    model_flags = [0, 1, 2]
    map_types = ['identity', 'smooth']

    if post_processing_flag:
       
        mesh_sizes = np.load('data/numpy/{}/mesh_sizes.npy'.format(pde.case_name))
        
        for model_flag in model_flags:
            fig, ax = plt.subplots(num=model_flag, figsize=(8, 6))
            for map_type in map_types:
                label = 'MPFM' if map_type == 'smooth' else 'PFM'

                if model_flag == 0:
                    label = label + ' - Isotropic'
                elif model_flag == 1:
                    label = label + ' - Anisotropic (Amor)'
                else:
                    label = label + ' - Anisotropic (Miehe)'

                color = 'red' if map_type == 'smooth' else 'blue'
                u_energy_errors = np.load('data/numpy/{}/model_{}_map_{}.npy'.format(pde.case_name, model_flag, map_type))
                plt.plot(mesh_sizes, u_energy_errors, linestyle='--', linewidth=2, marker='s', markersize=10, label=label, color=color)

                print(np.log2(u_energy_errors[1]/u_energy_errors[-1]))

            plt.yscale("log")
            plt.xscale("log")

            # ax.set_xticks([1e-2])
            # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

            ax.get_xaxis().set_tick_params(which='minor', labelsize=15)
            ax.get_xaxis().set_tick_params(which='major', labelsize=17)
            ax.get_yaxis().set_tick_params(which='minor', labelsize=15, rotation=90)
            ax.get_yaxis().set_tick_params(which='major', labelsize=17, rotation=90)

            plt.legend(fontsize=18, frameon=False)
            plt.xlabel(r"$h$", fontsize=20)
            plt.ylabel(r"$\Vert \boldsymbol{u}^h - \boldsymbol{u}_e \Vert_E$", fontsize=20)

            # plt.axis('equal')

            p1 = [3*1e-2, 2*1e-2]
            p2 = [4*1e-2, 2*1e-2]
            p3 = [p2[0], np.exp(0.5*np.log(p2[0]/p1[0]))*p1[1]]
            ax.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], color='black')  

            plt.text(p1[0] + 15*1e-4, p1[1] + 20*1e-4, '0.5', fontsize=18)

            fig.savefig('data/pdf/{}/convergence_model_{}.pdf'.format(pde.case_name, model_flag), bbox_inches='tight')
    else:
        for model_flag in model_flags:
            for map_type in map_types:
                run_one_model(args, model_flag, map_type)


def generate_interpolate_xdmf(args):
    args.psi_plus_linear_elasticity = psi_plus_linear_elasticity_model_A
    args.psi_minus_linear_elasticity = psi_minus_linear_elasticity_model_A
    args.model_flag = 0
    args.map_type = 'identity'
    args.local_refinement_iteration = 0
    pde = Analysis(args)
    pde.staggered_solve()
    pde.interpolate_solution()



if __name__ == '__main__':
    args = arguments.args
    main(args)
    # generate_interpolate_xdmf(args)
    plt.show()