import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
from functools import partial
from . import arguments
from .constitutive import *


fe.parameters["form_compiler"]["quadrature_degree"] = 4

# sigma_c = 2
# psi_cr = sigma_c**2 / (2 * E)

psi_cr = 0.03

Gc_0 = 0.1
l0 = 1.


mu  = 0.19
nu = 0.45
lamda = mu * ((2. * nu) / (1. - 2. * nu))
kappa = lamda + 2. / 3. * mu
E = 3 * kappa * (1 - 2 * nu)
beta = 2 * nu / (1 - 2 * nu)


staggered_tol = 1e-6 
staggered_maxiter = 20


class PDE(object):
    def __init__(self, args):
        self.args = args
        self.preparation()
        self.build_mesh()
        self.set_boundaries()

    def preparation(self):
        files = glob.glob('data/pvd/{}/*'.format(self.args.case_name))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print('Failed to delete {}, reason: {}' % (f, e))

    def set_boundaries(self):
        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.ds = fe.Measure("ds")(subdomain_data=self.boundaries)   
        self.I = fe.Identity(self.mesh.topology().dim())
        self.normal = fe.FacetNormal(self.mesh)


    def monolithic_solve(self):
        U = fe.VectorElement('CG', self.mesh.ufl_cell(), 2)  
        W = fe.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.M = fe.FunctionSpace(self.mesh, U * W)

        WW = fe.FunctionSpace(self.mesh, 'DG', 0) 
        EE = fe.FunctionSpace(self.mesh, 'CG', 1) 

        m_test = fe.TestFunctions(self.M)
        m_delta = fe.TrialFunctions(self.M)
        m_new = fe.Function(self.M)

        self.eta, self.zeta = m_test
        self.x_new, self.d_new = fe.split(m_new)

        self.H_old = fe.Function(WW)
        E = fe.Function(EE)

        self.build_weak_form_monolithic()
        dG = fe.derivative(self.G, m_new)

        self.set_bcs_monolithic()
        p = fe.NonlinearVariationalProblem(self.G, m_new, self.BC, dG)
        solver = fe.NonlinearVariationalSolver(p)

        vtkfile_u = fe.File('data/pvd/{}/u.pvd'.format(self.args.case_name))
        vtkfile_d = fe.File('data/pvd/{}/d.pvd'.format(self.args.case_name))
        vtkfile_e = fe.File('data/pvd/{}/e.pvd'.format(self.args.case_name))

        for disp, rp in zip(self.args.displacements, self.args.relaxation_parameters):

            print(' ')
            print('=================================================================================')
            print('>> disp boundary condition = {} [mm]'.format(disp))
            print('=================================================================================')

            self.presLoad.t = disp

            newton_prm = solver.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 1000
            newton_prm['linear_solver'] = 'mumps'   
            newton_prm['absolute_tolerance'] = 1e-4
            newton_prm['relaxation_parameter'] = rp

            solver.solve()

            self.H_old.assign(fe.project(history(self.x_new, self.H_old, self.I, psi_cr, self.psi_plus), WW))

            # E.assign(fe.project(psi(self.I + fe.grad(self.x_new)), EE))
            # E.assign(fe.project(first_PK_stress(self.I + fe.grad(x_new))[0, 0], EE))
            
            print('=================================================================================')
            print(' ')

            x_plot, d_plot = m_new.split()
            x_plot.rename("u", "u")
            d_plot.rename("d", "d")
            vtkfile_u << x_plot
            vtkfile_d << d_plot
            vtkfile_e << self.H_old 


    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 2)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 
        WW = fe.FunctionSpace(self.mesh, 'DG', 0) 
        
        self.eta = fe.TestFunction(self.U)
        self.zeta = fe.TestFunction(self.W)

        del_x = fe.TrialFunction(self.U)
        del_d = fe.TrialFunction(self.W)

        self.x_new = fe.Function(self.U)
        self.d_new = fe.Function(self.W)

        x_old = fe.Function(self.U)
        d_old = fe.Function(self.W) 

        self.H_old = fe.Function(WW)

        self.build_weak_form_staggered()
        J_ut = fe.derivative(self.G_ut, self.x_new, del_x)
        J_d = fe.derivative(self.G_d, self.d_new, del_d) 

        self.set_bcs_staggered()
        p_ut = fe.NonlinearVariationalProblem(self.G_ut, self.x_new, self.BC, J_ut)
        p_d  = fe.NonlinearVariationalProblem(self.G_d,  self.d_new, self.BC_d, J_d)
        solver_ut = fe.NonlinearVariationalSolver(p_ut)
        solver_d  = fe.NonlinearVariationalSolver(p_d)

        vtkfile_u = fe.File('data/pvd/circular_holes/u.pvd')
        vtkfile_d = fe.File('data/pvd/circular_holes/d.pvd')

        for disp, rp in zip(self.args.displacements, self.args.relaxation_parameters):

            print(' ')
            print('=================================================================================')
            print('>> disp boundary condition = {} [mm]'.format(disp))
            print('=================================================================================')

            self.presLoad.t = disp

            newton_prm = solver_ut.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 1000
            newton_prm['linear_solver'] = 'mumps'   
            newton_prm['absolute_tolerance'] = 1e-4
            newton_prm['relaxation_parameter'] = rp
 
            iteration = 0
            err = 1.

            self.H_old.assign(fe.project(history(self.x_new, self.H_old, self.I, psi_cr, self.psi_plus), WW))

            while err > staggered_tol:
                iteration += 1

                # solve phase field equation
                print('[Solving phase field equation...]')
                solver_d.solve()

                # solve momentum balance equations
                print(' ')
                print('[Solving balance equations...]')
                solver_ut.solve()

                # compute error norms
                print(' ')
                print('[Computing residuals...]')

                err_u = fe.errornorm(self.x_new, x_old, norm_type='l2', mesh=None)
                err_d = fe.errornorm(self.d_new, d_old, norm_type='l2', mesh=None)
                err = max(err_u, err_d)

                x_old.assign(self.x_new)
                d_old.assign(self.d_new)

                print('---------------------------------------------------------------------------------')
                print('>> iteration. {}, error = {:.5}'.format(iteration, err))
                print('---------------------------------------------------------------------------------')

                if err < staggered_tol or iteration >= staggered_maxiter:

                    print(
                        '=================================================================================')
                    print(' ')

                    self.x_new.rename("u", "u")
                    self.d_new.rename("d", "d")
                    vtkfile_u << self.x_new
                    vtkfile_d << self.d_new
                    break



class DoubleCircles(PDE):
    def __init__(self, args):
        super(DoubleCircles, self).__init__(args)


    def build_mesh(self):
        length = 60
        height = 30
        radius = 5
        plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(length, height))
        circle1 = mshr.Circle(fe.Point(length/3, height/3), radius)
        circle2 = mshr.Circle(fe.Point(length*2/3, height*2/3), radius)
        material_domain = plate - circle1 - circle2
        self.mesh = mshr.generate_mesh(material_domain, 50)

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
        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_left = fe.DirichletBC(self.M.sub(0).sub(0), fe.Constant(0),  self.left)
        BC_u_right = fe.DirichletBC(self.M.sub(0).sub(0), self.presLoad,  self.right )
        BC_u_corner = fe.DirichletBC(self.M.sub(0).sub(1), fe.Constant(0.0), self.corner, method='pointwise')
        self.BC = [BC_u_left, BC_u_right, BC_u_corner] 
        # self.right.mark(self.boundaries, 1)


    def build_weak_form_monolithic(self):
        self.psi_plus = partial(psi_plus_Miehe, mu=mu, beta=beta)
        self.psi_minus = partial(psi_minus_Miehe, mu=mu, beta=beta)

        PK_plus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_plus)
        PK_minus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_minus)

        G_ut = (g_d(self.d_new) * fe.inner(PK_plus, fe.grad(self.eta)) + fe.inner(PK_minus, fe.grad(self.eta)) )* fe.dx
        G_d = self.H_old * self.zeta * g_d_prime(self.d_new, g_d) * fe.dx \
            + 2 * psi_cr * (self.zeta * self.d_new + l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new))) * fe.dx  
        self.G = G_ut + G_d


    def set_bcs_staggered(self):
        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_left = fe.DirichletBC(self.U.sub(0), fe.Constant(0), self.left)
        BC_u_right = fe.DirichletBC(self.U.sub(0), self.presLoad, self.right )
        BC_u_corner = fe.DirichletBC(self.U.sub(1), fe.Constant(0.0), self.corner, method='pointwise')
        self.BC = [BC_u_left, BC_u_right, BC_u_corner]     
        self.BC_d = []


    def build_weak_form_staggered(self):
        self.psi_plus = partial(psi_plus_Miehe, mu=mu, beta=beta)
        self.psi_minus = partial(psi_minus_Miehe, mu=mu, beta=beta)

        PK_plus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_plus)
        PK_minus = first_PK_stress_plus(self.I + fe.grad(self.x_new), self.psi_minus)
 
        self.G_ut = (g_d(self.d_new) * fe.inner(PK_plus, fe.grad(self.eta)) +  fe.inner(PK_minus, fe.grad(self.eta)))* fe.dx
        self.G_d = self.H_old * self.zeta * g_d_prime(self.d_new, g_d) * fe.dx \
            + 2 * psi_cr * (self.zeta * self.d_new + l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new))) * fe.dx  


def test(args):
    args.case_name = "circular_holes"
    args.displacements = np.concatenate((np.linspace(1, 11, 6), np.linspace(12, 26.5, 30), np.linspace(27, 40, 53)))
    args.relaxation_parameters = np.concatenate((np.linspace(0.2, 0.2, 11), np.linspace(0.1, 0.1, 24), np.linspace(0.02, 0.02, 54)))

    pde = DoubleCircles(args)
    # pde.monolithic_solve()
    pde.staggered_solve()


if __name__ == '__main__':
    args = arguments.args
    test(args)