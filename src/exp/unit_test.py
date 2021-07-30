'''A minimum working example using both staggered and monolithic solvers
'''
import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
from functools import partial


fe.parameters["form_compiler"]["quadrature_degree"] = 4


# Miehe paper:https://doi.org/10.1016/j.cma.2014.11.016
# Equation (53)
def history(H_old, psi_new, psi_cr):
    history_max_tmp = fe.conditional(fe.gt(psi_new - psi_cr, 0), psi_new - psi_cr, 0)
    history_max = fe.conditional(fe.gt(history_max_tmp, H_old), history_max_tmp, H_old)
    return history_max


# ---------------------------------------------------------------- 
# Degradation functions

def g_d(d):
    degrad = (1 - d)**2
    return degrad 


def g_d_prime(d, degrad_func):
    d = fe.variable(d)
    degrad = degrad_func(d)
    degrad_prime = fe.diff(degrad, d)
    return degrad_prime


# ---------------------------------------------------------------- 
# Linear elasticity

def strain(grad_u):
    return 0.5*(grad_u + grad_u.T)


def psi_linear_elasticity(epsilon, lamda, mu):
    return lamda / 2 * fe.tr(epsilon)**2 + mu * fe.inner(epsilon, epsilon)


def cauchy_stress(epsilon, psi):
    epsilon = fe.variable(epsilon)
    energy = psi(epsilon)
    sigma = fe.diff(energy, epsilon)
    return sigma


class PDE(object):
    def __init__(self):
        self.preparation()
        self.build_mesh()
        self.set_boundaries()
        self.l0 = 2 * self.mesh.hmin()
        self.staggered_tol = 1e-6 
        self.staggered_maxiter = 1000
        self.delta_u_recorded = []
        self.sigma_recorded = []


    def preparation(self):
        files = glob.glob('data/pvd/{}/*'.format(self.case_name))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print('Failed to delete {}, reason: {}' % (f, e))

    def set_boundaries(self):
        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.ds = fe.Measure("ds")(subdomain_data=self.boundaries)   


    def monolithic_solve(self):
        self.U = fe.VectorElement('CG', self.mesh.ufl_cell(), 1)  
        self.W = fe.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.M = fe.FunctionSpace(self.mesh, self.U * self.W)

        self.WW = fe.FunctionSpace(self.mesh, 'DG', 0) 

        m_test = fe.TestFunctions(self.M)
        m_delta = fe.TrialFunctions(self.M)
        m_new = fe.Function(self.M)

        self.eta, self.zeta = m_test
        self.x_new, self.d_new = fe.split(m_new)

        self.H_old = fe.Function(self.WW)

        vtkfile_u = fe.File('data/pvd/{}/u.pvd'.format(self.case_name))
        vtkfile_d = fe.File('data/pvd/{}/d.pvd'.format(self.case_name))

        self.build_weak_form_monolithic()
        dG = fe.derivative(self.G, m_new)

        self.set_bcs_monolithic()
        p = fe.NonlinearVariationalProblem(self.G, m_new, self.BC, dG)
        solver = fe.NonlinearVariationalSolver(p)

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):

            print('\n')
            print('=================================================================================')
            print('>> Step {}, disp boundary condition = {} [mm]'.format(i, disp))
            print('=================================================================================')

            self.H_old.assign(fe.project(history(self.H_old, self.psi(strain(fe.grad(self.x_new))), self.psi_cr), self.WW))

            self.presLoad.t = disp

            newton_prm = solver.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100  
            newton_prm['absolute_tolerance'] = 1e-4
            newton_prm['relaxation_parameter'] = rp

            solver.solve()

            self.x_plot, self.d_plot = m_new.split()
            self.x_plot.rename("u", "u")
            self.d_plot.rename("d", "d")

            vtkfile_u << self.x_plot
            vtkfile_d << self.d_plot

            force_upper = float(fe.assemble(self.sigma[1, 1]*self.ds(1)))
            print("Force upper {}".format(force_upper))
            self.delta_u_recorded.append(disp)
            self.sigma_recorded.append(force_upper)

            print('=================================================================================')


    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 
        self.WW = fe.FunctionSpace(self.mesh, 'DG', 0) 
        
        self.eta = fe.TestFunction(self.U)
        self.zeta = fe.TestFunction(self.W)

        del_x = fe.TrialFunction(self.U)
        del_d = fe.TrialFunction(self.W)

        self.x_new = fe.Function(self.U)
        self.d_new = fe.Function(self.W)

        x_old = fe.Function(self.U)
        d_old = fe.Function(self.W) 

        self.H_old = fe.Function(self.WW)

        self.build_weak_form_staggered()
        J_u = fe.derivative(self.G_u, self.x_new, del_x)
        J_d = fe.derivative(self.G_d, self.d_new, del_d) 

        self.set_bcs_staggered()
        p_u = fe.NonlinearVariationalProblem(self.G_u, self.x_new, self.BC_u, J_u)
        p_d  = fe.NonlinearVariationalProblem(self.G_d,  self.d_new, self.BC_d, J_d)
        solver_u = fe.NonlinearVariationalSolver(p_u)
        solver_d  = fe.NonlinearVariationalSolver(p_d)

        vtkfile_u = fe.File('data/pvd/{}/u.pvd'.format(self.case_name))
        vtkfile_d = fe.File('data/pvd/{}/d.pvd'.format(self.case_name))

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):

            print('\n')
            print('=================================================================================')
            print('>> Step {}, disp boundary condition = {} [mm]'.format(i, disp))
            print('=================================================================================')

            self.H_old.assign(fe.project(history(self.H_old, self.psi(strain(fe.grad(self.x_new))), self.psi_cr), self.WW))

            self.presLoad.t = disp

            newton_prm = solver_u.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100  
            newton_prm['absolute_tolerance'] = 1e-4
            newton_prm['relaxation_parameter'] = rp
 
            iteration = 0
            err = 1.

            while err > self.staggered_tol:
                iteration += 1

                solver_d.solve()

                solver_u.solve()

                err_u = fe.errornorm(self.x_new, x_old, norm_type='l2', mesh=None)
                err_d = fe.errornorm(self.d_new, d_old, norm_type='l2', mesh=None)
                err = max(err_u, err_d)

                x_old.assign(self.x_new)
                d_old.assign(self.d_new)

                print('---------------------------------------------------------------------------------')
                print('>> iteration. {}, error = {:.5}'.format(iteration, err))
                print('---------------------------------------------------------------------------------')

                if err < self.staggered_tol or iteration >= self.staggered_maxiter:
                    print('=================================================================================')
                    print('\n')

                    self.x_new.rename("u", "u")
                    self.d_new.rename("d", "d")
                    vtkfile_u << self.x_new
                    vtkfile_d << self.d_new
                    break

            force_upper = float(fe.assemble(self.sigma[1, 1]*self.ds(1)))
            print("Force upper {}".format(force_upper))
            self.delta_u_recorded.append(disp)
            self.sigma_recorded.append(force_upper)



class TestCase(PDE):
    def __init__(self):
        self.case_name = "brittle"
        super(TestCase, self).__init__()

        self.displacements = np.concatenate((np.linspace(0., 0.15, 11), np.linspace(0.15, 0.4, 51)))
 
        # self.relaxation_parameters = np.concatenate((np.linspace(1, 1, 5), np.linspace(0.1, 0.1, len(self.displacements) - 5)))
        self.relaxation_parameters =  np.linspace(1, 1, len(self.displacements))

        self.psi_cr = 0.01
        self.mu = 1e3
        self.nu = 0.4
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)


    def build_mesh(self):
        self.length = 100
        self.height = 100

        plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(self.length, self.height))
        notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1), fe.Point(0, self.height / 2 - 1), fe.Point(6, self.height / 2)])
        self.mesh = mshr.generate_mesh(plate - notch, 30)

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


    def set_bcs_monolithic(self):
        self.upper.mark(self.boundaries, 1)
        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_lower = fe.DirichletBC(self.M.sub(0).sub(1), fe.Constant(0),  self.lower)
        BC_u_upper = fe.DirichletBC(self.M.sub(0).sub(1), self.presLoad,  self.upper)
        BC_u_corner = fe.DirichletBC(self.M.sub(0).sub(0), fe.Constant(0.0), self.corner, method='pointwise')
        self.BC = [BC_u_lower, BC_u_upper, BC_u_corner]

 
    def build_weak_form_monolithic(self):
        self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)
        self.sigma = cauchy_stress(strain(fe.grad(self.x_new)), self.psi)
     
        G_u = g_d(self.d_new) * fe.inner(self.sigma, strain(fe.grad(self.eta))) * fe.dx

        G_d = (history(self.H_old, self.psi(strain(fe.grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        # g_c = 0.1
        # G_d = (self.psi(strain(fe.grad(self.x_new))) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + g_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        self.G = G_u + G_d


    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)
        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_lower = fe.DirichletBC(self.U.sub(1), fe.Constant(0),  self.lower)
        BC_u_upper = fe.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        BC_u_corner = fe.DirichletBC(self.U.sub(0), fe.Constant(0.0), self.corner, method='pointwise')
        self.BC_u = [BC_u_lower, BC_u_upper, BC_u_corner]
        self.BC_d = []


    def build_weak_form_staggered(self):
        self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)
        self.sigma = cauchy_stress(strain(fe.grad(self.x_new)), self.psi)

        self.G_u = g_d(self.d_new) * fe.inner(self.sigma, strain(fe.grad(self.eta))) * fe.dx
 
        self.G_d = (history(self.H_old, self.psi(strain(fe.grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        # g_c = 0.1
        # self.G_d = (self.psi(strain(fe.grad(self.x_new))) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + g_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx


def test():
    pde = TestCase()
    # pde.monolithic_solve()
    pde.staggered_solve()

    plt.figure()
    plt.plot(pde.delta_u_recorded, pde.sigma_recorded, linestyle='--', marker='o', color='red')
    plt.tick_params(labelsize=14)
    plt.xlabel("Vertical displacement of top side", fontsize=14)
    plt.ylabel("Force on top side", fontsize=14)
    plt.show()


if __name__ == '__main__':
    test()