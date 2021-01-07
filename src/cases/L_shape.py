import fenics as fe
import dolfin_adjoint as da
import sys
import meshio
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
from ..mesh_converter import save_with_meshio, load_with_meshio


class LShape(MappedPDE):
    def __init__(self, args):
        self.case_name = "L_shape"
        self.mesh_refinement_level = "refine_0"
        self.solution_scheme = 'explicit'
        self.local_refinement_iteration = 0
        super(LShape, self).__init__(args)

        # self.displacements = 1e-2*np.linspace(0.0, 0.15, 51)
        # self.displacements = 1e-2*np.concatenate((np.linspace(0, 0.10, 11), np.linspace(0.10, 0.15, 51)))

        # self.displacements = np.concatenate((np.linspace(0, 0.25, 26), 
        #                                      np.linspace(0.25, 0.27, 101), 
        #                                      np.linspace(0.27, 0.3, 201),
        #                                      np.linspace(0.3, -0.2, 51),
        #                                      np.linspace(-0.2, 0.2, 41),
        #                                      np.linspace(0.2, 0.5, 101)))

        self.displacements = np.concatenate((np.linspace(0, 0.25, 26), 
                                             np.linspace(0.25, 0.3, 201),
                                             np.linspace(0.3, -0.2, 51),
                                             np.linspace(-0.2, 0.2, 41),
                                             np.linspace(0.2, 0.5, 201)))

        # self.displacements =  np.linspace(0, 0.2, 10)    

        # self.displacements =  np.concatenate((np.linspace(0, 0.25, 26), 
        #                                       np.linspace(0.25, 0.27, 101), 
        #                                       np.linspace(0.27, 0.3, 201)))

        self.relaxation_parameters = np.linspace(1., 1., len(self.displacements))
 
        # Standard L-shape test parameters for non-cyclic loading
        # self.E = 25.85
        # self.nu = 0.18
        # self.mu = self.E / (2 * (1 + self.nu))
        # self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)
        # self.G_c = 95*1e-6

        # Standard L-shape test parameters for cyclic loading
        self.mu = 10.95
        self.lamda = 6.16
        self.G_c = 8.9*1e-5

        # self.psi_cr = 1e-7
        # self.psi_cr = self.G_c / (2 * self.l0)
        self.psi_cr = 0.

        # self.l0 = self.mesh.hmax() + self.mesh.hmin()
        self.l0 = 6

        print(self.mesh.hmax())
        print(self.mesh.hmin())
        print("self.l0 is {}".format(self.l0))

        self.map_type = 'identity'

        if self.map_type == 'linear' or self.map_type == 'smooth':
            self.map_flag = True
            self.local_refinement_iteration = 0
        elif self.map_type == 'identity':
            self.finish_flag = True
            self.map_flag = False

        self.rho_default = 150.
        self.d_integral_interval = 3*self.rho_default
        self.initialize_control_points_and_impact_radii()


    def initialize_control_points_and_impact_radii(self):
        self.control_points = []
        self.impact_radii = []
        control_points = np.asarray([[self.length/2, self.height/2]])
        for new_tip_point in control_points:
            self.compute_impact_radii(new_tip_point)


    def build_mesh(self):
        path_to_msh = 'data/gmsh/{}/{}/mesh'.format(self.case_name, self.mesh_refinement_level)
        save_with_meshio(path_to_msh, 2)

        path_to_xdmf = 'data/gmsh/{}/{}/mesh.xdmf'.format(self.case_name, self.mesh_refinement_level)
        xdmf_mesh = fe.XDMFFile(path_to_xdmf)
        self.mesh = fe.Mesh()
        xdmf_mesh.read(self.mesh)

        self.length = 500
        self.height = 500
        self.segment = 30
 
        # Reference
        # https://scicomp.stackexchange.com/questions/32647/how-to-use-meshfunction-in-fenics-dolfin
        # https://fenicsproject.org/qa/596/setting-condition-for-mesh-refinement/
        for i in range(self.local_refinement_iteration):
            cell_markers = fe.MeshFunction('bool', self.mesh, self.mesh.topology().dim())
            cell_markers.set_all(False)
            for cell in fe.cells(self.mesh):
                p = cell.midpoint()
                if  p[0] > 3./20.*self.length and p[0] < 10.5/20.*self.length and p[1] > 9.5/20.*self.height and p[1] < 13/20*self.height:
                # if np.sqrt((p[0] - self.length/2.)**2 + (p[1] - self.height/2.)**2) < self.length/5.:
                    cell_markers[cell] = True
            self.mesh = fe.refine(self.mesh, cell_markers)

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
                return  fe.near(x[1], height/2.) and fe.near(x[0], length - segment)
                # return  fe.near(x[1], height/2.) and x[0] >= length - segment

        self.lower = Lower()
        self.upper = Upper()
        self.segment = Segment()
  
 
    def set_bcs_staggered(self):
        self.lower.mark(self.boundaries, 1)
        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_lower = fe.DirichletBC(self.U, fe.Constant((0., 0.)),  self.lower)
        BC_u_segment = fe.DirichletBC(self.U.sub(1), self.presLoad,  self.segment, method="pointwise")
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

        if self.solution_scheme == 'explicit':
            self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx
        else:
            self.G_d = (history(self.H_old, self.psi_plus(strain(self.mfem_grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(self.mfem_grad(self.zeta), self.mfem_grad(self.d_new)))) * fe.det(self.grad_gamma) * fe.dx


    def update_history(self):
        psi_new = self.psi_plus(strain(self.mfem_grad(self.x_new)))  
        return psi_new


    def update_weak_form_due_to_Model_C_bug(self):
        if self.i < 3:
            self.update_weak_form = True


    def create_custom_xdmf_files(self):
        self.file_results_L_shape = fe.XDMFFile('data/xdmf/{}/u_refine_{}_mfem_{}.xdmf'.format(self.case_name, 
            self.local_refinement_iteration, self.map_flag))
        self.file_results_L_shape.parameters["functions_share_mesh"] = True


    def save_data_in_loop(self):
        np.save('data/numpy/{}/force_refine_{}_mfem_{}.npy'.format(self.case_name, 
            self.local_refinement_iteration, self.map_flag), self.force_recorded)
        np.save('data/numpy/{}/displacement_refine_{}_mfem_{}.npy'.format(self.case_name, 
            self.local_refinement_iteration, self.map_flag), self.delta_u_recorded)

        self.file_results_L_shape.write(self.x_new, self.i)
        self.file_results_L_shape.write(self.d_new, self.i)
        self.file_results_L_shape.write(self.map_plot, self.i)


    def post_processing(self):
        delta_u_recorded_coarse = np.load('data/numpy/{}/displacement_refine_{}_mfem_{}.npy'.format(self.case_name, 0, False))
        force_recorded_coarse = np.load('data/numpy/{}/force_refine_{}_mfem_{}.npy'.format(self.case_name, 0, False))

        delta_u_recorded_fine = np.load('data/numpy/{}/displacement_refine_{}_mfem_{}.npy'.format(self.case_name, 1, False))
        force_recorded_fine = np.load('data/numpy/{}/force_refine_{}_mfem_{}.npy'.format(self.case_name, 1, False))

        delta_u_recorded_mfem = np.load('data/numpy/{}/displacement_refine_{}_mfem_{}.npy'.format(self.case_name, 0, True))
        force_recorded_mfem = np.load('data/numpy/{}/force_refine_{}_mfem_{}.npy'.format(self.case_name, 0, True))

        fig = plt.figure(0)
        # plt.plot(delta_u_recorded_coarse, force_recorded_coarse, linestyle='--', marker='o', color='blue', label='coarse')
        # plt.plot(delta_u_recorded_fine, force_recorded_fine, linestyle='--', marker='o', color='yellow', label='fine')
        # plt.plot(delta_u_recorded_mfem, force_recorded_mfem, linestyle='--', marker='o', color='red', label='mfem')

        plt.plot(delta_u_recorded_coarse, force_recorded_coarse, linestyle='-', linewidth=4, color='blue', label='coarse')
        plt.plot(delta_u_recorded_fine, force_recorded_fine, linestyle='-', linewidth=4, color='yellow', label='fine')
        plt.plot(delta_u_recorded_mfem, force_recorded_mfem, linestyle='-', linewidth=4, color='red', label='mfem')

        plt.legend(fontsize=14)
        plt.tick_params(labelsize=14)
        plt.xlabel("Vertical displacement of top side", fontsize=14)
        plt.ylabel("Force on top side", fontsize=14)
        plt.grid(True)
        plt.show()


def test(args):
    pde = LShape(args)

    # pde.staggered_solve()

    pde.post_processing()
 

if __name__ == '__main__':
    args = arguments.args
    test(args)
