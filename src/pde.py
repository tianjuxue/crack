import fenics as fe
import dolfin_adjoint as da
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
import shutil
from functools import partial
import scipy.optimize as opt
from pyadjoint.overloaded_type import create_overloaded_object
from . import arguments
from .constitutive import *
from .mfem import distance_function_segments_ufl, distance_function_segments_normal, map_function_normal, inverse_map_function_normal, map_function_ufl


# fe.parameters["form_compiler"]["quadrature_degree"] = 4


class PDE(object):
    def __init__(self, args):
        self.args = args
        self.preparation()
        self.build_mesh()
        self.set_boundaries()
        self.l0 = 2 * self.mesh.hmin()
        self.staggered_tol = 1e-5
        self.staggered_maxiter = 1000 
        self.monolithic_tol = 1e-5
        self.monolithic_maxiter = 1000
        self.map_flag = False
        self.delta_u_recorded = []
        self.sigma_recorded = []
        self.psi_cr = 0.01


    def preparation(self):
        # files = glob.glob('data/pvd/{}/*'.format(self.case_name))
        # for f in files:
        #     try:
        #         os.remove(f)
        #     except Exception as e:
        #         print('Failed to delete {}, reason: {}' % (f, e))
        data_path = 'data/pvd/{}'.format(self.case_name)
        print("\nDelete data folder {}".format(data_path))
        shutil.rmtree(data_path, ignore_errors=True)


    def set_boundaries(self):
        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.ds = fe.Measure("ds")(subdomain_data=self.boundaries)   
        self.I = fe.Identity(self.mesh.topology().dim())
        self.normal = fe.FacetNormal(self.mesh)


    def update_history(self):
        return 0


    def monolithic_solve(self):
        self.U = fe.VectorElement('CG', self.mesh.ufl_cell(), 1)  
        self.W = fe.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.M = fe.FunctionSpace(self.mesh, self.U * self.W)

        self.WW = fe.FunctionSpace(self.mesh, 'DG', 0)
        self.EE = fe.FunctionSpace(self.mesh, 'CG', 1) 
        self.MM = fe.VectorFunctionSpace(self.mesh, 'CG', 1)

        m_test = fe.TestFunctions(self.M)
        m_delta = fe.TrialFunctions(self.M)
        m_new = da.Function(self.M)
        m_old = da.Function(self.M) 
        m_pre = da.Function(self.M) 

        self.eta, self.zeta = m_test
        self.x_new, self.d_new = fe.split(m_new) # fe.split(m_new) is used in weak form, different from m_new.split() 
        self.x_pre, self.d_pre = fe.split(m_pre)

        self.H_old = da.Function(self.WW)
        self.H_new = da.Function(self.WW)

        e = da.Function(self.EE, name="e")
        map_plot = da.Function(self.MM, name="m")

        file_results = fe.XDMFFile('data/xdmf/{}/u.xdmf'.format(self.case_name))
        file_results.parameters["functions_share_mesh"] = True

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

            m_pre.assign(m_new)

            self.presLoad.t = disp

            newton_prm = solver.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100
            # newton_prm['absolute_tolerance'] = 1e-4
            newton_prm['relaxation_parameter'] = rp

            self.H_old.assign(self.H_new)

            vtkfile_u_staggered = fe.File('data/pvd/{}/step{}/u.pvd'.format(self.case_name, i))
            vtkfile_d_staggered = fe.File('data/pvd/{}/step{}/d.pvd'.format(self.case_name, i))
            iteration = 0
            err = 1.
            while err > self.monolithic_tol:
                iteration += 1

                self.H_new.assign(fe.project(history(self.H_old, self.update_history(), self.psi_cr), self.WW))

                solver.solve()

                np_m_new = np.asarray(m_new.vector())
                np_m_old = np.asarray(m_old.vector())
                err = np.linalg.norm(np_m_new - np_m_old) / np.sqrt(len(np_m_new))
    
                m_old.assign(m_new)

                print('---------------------------------------------------------------------------------')
                print('>> iteration. {}, error = {:.5}'.format(iteration, err))
                print('---------------------------------------------------------------------------------')

                self.x_plot, self.d_plot = m_new.split()
                self.x_plot.rename("u", "u")
                self.d_plot.rename("d", "d")
                vtkfile_u_staggered << self.x_plot
                vtkfile_d_staggered << self.d_plot

                if err < self.monolithic_tol or iteration >= self.monolithic_maxiter:
                    print('=================================================================================')
                    print('\n')
                    break

            if self.map_flag:
                delta_x = self.x - self.x_hat
                map_plot.assign(fe.project(delta_x, self.MM))


            # if self.map_flag:
            #     self.update_map()


            e.assign(da.interpolate(self.H_old, self.EE))
            # e.assign(da.project(self.psi_plus(strain(fe.grad(self.x_new))), self.EE))
            # e.assign(da.project(first_PK_stress(self.I + fe.grad(x_new))[0, 0], EE))

            self.x_plot, self.d_plot = m_new.split()
            self.x_plot.rename("u", "u")
            self.d_plot.rename("d", "d")
 
            file_results.write(self.x_plot, i)
            file_results.write(self.d_plot, i)
            file_results.write(e, i)
            file_results.write(map_plot, i)

            vtkfile_u << self.x_plot
            vtkfile_d << self.d_plot
            self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)
            self.sigma = cauchy_stress(strain(fe.grad(self.x_new)), self.psi)
            force_upper = float(fe.assemble(self.sigma[1, 1]*self.ds(1)))
            print("Force upper {}".format(force_upper))
            self.delta_u_recorded.append(disp)
            self.sigma_recorded.append(force_upper)

            print('=================================================================================')


    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 

        self.EE = fe.FunctionSpace(self.mesh, 'CG', 1) 
        self.WW = fe.FunctionSpace(self.mesh, 'DG', 0) 
        self.MM = fe.VectorFunctionSpace(self.mesh, 'CG', 1)

        self.eta = fe.TestFunction(self.U)
        self.zeta = fe.TestFunction(self.W)

        del_x = fe.TrialFunction(self.U)
        del_d = fe.TrialFunction(self.W)

        self.x_new = da.Function(self.U, name="u")
        self.d_new = da.Function(self.W, name="d")
        self.d_pre = da.Function(self.W)

        x_old = da.Function(self.U)
        d_old = da.Function(self.W) 

        self.H_old = da.Function(self.WW)

        map_plot = da.Function(self.MM, name="m")
        e = da.Function(self.EE, name="e")

        self.build_weak_form_staggered()
        J_u = fe.derivative(self.G_u, self.x_new, del_x)
        J_d = fe.derivative(self.G_d, self.d_new, del_d) 

        self.set_bcs_staggered()
        p_u = fe.NonlinearVariationalProblem(self.G_u, self.x_new, self.BC_u, J_u)
        p_d  = fe.NonlinearVariationalProblem(self.G_d,  self.d_new, self.BC_d, J_d)
        solver_u = fe.NonlinearVariationalSolver(p_u)
        solver_d  = fe.NonlinearVariationalSolver(p_d)

        file_results = fe.XDMFFile('data/xdmf/{}/u.xdmf'.format(self.case_name))
        file_results.parameters["functions_share_mesh"] = True

        vtkfile_e = fe.File('data/pvd/{}/e.pvd'.format(self.case_name))
        vtkfile_u = fe.File('data/pvd/{}/u.pvd'.format(self.case_name))
        vtkfile_d = fe.File('data/pvd/{}/d.pvd'.format(self.case_name))

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):

            print('\n')
            print('=================================================================================')
            print('>> Step {}, disp boundary condition = {} [mm]'.format(i, disp))
            print('=================================================================================')

            self.d_pre.assign(self.d_new)

            self.presLoad.t = disp

            newton_prm = solver_u.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100 
            # newton_prm['absolute_tolerance'] = 1e-4
            newton_prm['relaxation_parameter'] = rp

            newton_prm = solver_d.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100 
            # newton_prm['absolute_tolerance'] = 1e-4
            newton_prm['relaxation_parameter'] = rp

            self.H_old.assign(fe.project(history(self.H_old, self.update_history(), self.psi_cr), self.WW))
            
            vtkfile_e_staggered = fe.File('data/pvd/{}/step{}/e.pvd'.format(self.case_name, i))
            vtkfile_u_staggered = fe.File('data/pvd/{}/step{}/u.pvd'.format(self.case_name, i))
            vtkfile_d_staggered = fe.File('data/pvd/{}/step{}/d.pvd'.format(self.case_name, i))
            iteration = 0
            err = 1.
            while err > self.staggered_tol:
                iteration += 1

                solver_d.solve()

                solver_u.solve()

                # dolfin (2019.1.0) errornorm function has severe bugs not behave as expected
                # The bug seems to be fixed in later versions
                # The following sometimes produces nonzero results in dolfin (2019.1.0)
                # print(fe.errornorm(self.d_new, self.d_new, norm_type='l2'))
                # We use another error measure similar in https://doi.org/10.1007/s10704-019-00372-y

                np_x_new = np.asarray(self.x_new.vector())
                np_d_new = np.asarray(self.d_new.vector())
                np_x_old = np.asarray(x_old.vector())
                np_d_old = np.asarray(d_old.vector())
                err_x = np.linalg.norm(np_x_new - np_x_old) / np.sqrt(len(np_x_new))
                err_d = np.linalg.norm(np_d_new - np_d_old) / np.sqrt(len(np_d_new))
                err = max(err_x, err_d)

                x_old.assign(self.x_new)
                d_old.assign(self.d_new)
                e.assign(fe.project(self.H_old, self.EE))

                print('---------------------------------------------------------------------------------')
                print('>> iteration. {}, err_u = {:.5}, err_d = {:.5}, error = {:.5}'.format(iteration, err_x, err_d, err))
                print('---------------------------------------------------------------------------------')

                vtkfile_e_staggered << e
                vtkfile_u_staggered << self.x_new
                vtkfile_d_staggered << self.d_new

                if err < self.staggered_tol or iteration >= self.staggered_maxiter:
                    print('=================================================================================')
                    print('\n')
                    break

            if self.map_flag:
                delta_x = self.x - self.x_hat
                map_plot.assign(fe.project(delta_x, self.MM))

            file_results.write(self.x_new, i)
            file_results.write(self.d_new, i)
            file_results.write(map_plot, i)

            vtkfile_e << e
            vtkfile_u << self.x_new
            vtkfile_d << self.d_new

            self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)
            self.sigma = cauchy_stress(strain(fe.grad(self.x_new)), self.psi)
            force_upper = float(fe.assemble(self.sigma[1, 1]*self.ds(1)))
            print("Force upper {}".format(force_upper))
            self.delta_u_recorded.append(disp)
            self.sigma_recorded.append(force_upper)

            if force_upper < 0.5 and i > 10:
                break


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













class StripeFabric(PDE):
    def __init__(self, args):
        self.case_name = "stripe_fabric"
        super(StripeFabric, self).__init__(args)
 
        self.displacements = np.linspace(0.15, 0.15, 501)
        # self.displacements = np.concatenate((np.linspace(0.15, 0.15, 51), np.linspace(0.2, 0.2, 21)))

        self.relaxation_parameters =  np.linspace(1, 1, len(self.displacements))

        self.psi_cr = 0.01
        self.l0 = 2 * self.mesh.hmin()
        self.build_lame_parameters()


    def build_lame_parameters(self):
        class MuExpression(da.UserExpression):
            def eval(self, values, x):
                if (x[0] // 50) % 2 == 0:
                    values[0] = 10*1e2
                else:
                    values[0] = 10*1e2
            def value_shape(self):
                return ()

        self.mu = MuExpression()
        self.nu = 0.4
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)




    def build_mesh(self):
        self.length = 100
        self.height = 100

        plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(self.length, self.height))
        notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1), fe.Point(0, self.height / 2 - 1), fe.Point(6, self.height / 2)])
        notch = mshr.Polygon([fe.Point(0, self.height / 2 + 1e-10), fe.Point(0, self.height / 2 - 1e-10), fe.Point(self.length / 2, self.height / 2)])
        # notch = mshr.Polygon([fe.Point(self.length / 4, self.height / 2), fe.Point(self.length / 2, self.height / 2 - 1e-10), \
        #                       fe.Point(self.length * 3 / 4, self.height / 2), fe.Point(self.length / 2, self.height / 2 + 1e-10)])
        self.mesh = mshr.generate_mesh(plate - notch, 50)

        # self.mesh = da.RectangleMesh(fe.Point(0, 0), fe.Point(self.length, self.height), 40, 40, diagonal="crossed")
 
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

        class Left(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0)

        class Right(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and fe.near(x[0], length)

        class Notch(fe.SubDomain):
            def inside(self, x, on_boundary):
                return  fe.near(x[1], height / 2) and x[0] < length / 20


        self.lower = Lower()
        self.upper = Upper()
        self.corner = Corner()

        self.left = Left()
        self.right = Right()
        self.notch = Notch()


    def set_bcs_monolithic(self):
        self.upper.mark(self.boundaries, 1)

        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.M.sub(0).sub(1), da.Constant(0),  self.lower)
        BC_u_upper = da.DirichletBC(self.M.sub(0).sub(1), self.presLoad,  self.upper)
        BC_u_corner = da.DirichletBC(self.M.sub(0).sub(0), da.Constant(0.0), self.corner, method='pointwise')
        self.BC = [BC_u_lower, BC_u_upper, BC_u_corner]

        # self.presLoad = da.Expression((0, "t"), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.M.sub(0), da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.M.sub(0), self.presLoad, self.upper) 
        # BC_u_left = da.DirichletBC(self.M.sub(0).sub(0), da.Constant(0),  self.left)
        # BC_u_right = da.DirichletBC(self.M.sub(0).sub(0), da.Constant(0),  self.right)
        # self.BC = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right]         
 
 
    def build_weak_form_monolithic(self):
        self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)

        sigma_plus = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi_plus)
        sigma_minus = cauchy_stress_minus(strain(fe.grad(self.x_new)), self.psi_minus)

        G_u = (g_d(self.d_new) * fe.inner(sigma_plus, strain(fe.grad(self.eta))) \
            + fe.inner(sigma_minus, strain(fe.grad(self.eta)))) * fe.dx

        G_d = (self.H_new * self.zeta * g_d_prime(self.d_new, g_d) \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        # G_d = (history(self.H_old, self.psi_plus(strain(fe.grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        # g_c = 0.01
        # G_d = (self.psi_plus(strain(fe.grad(self.x_new))) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + g_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        self.G = G_u + G_d


    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)

        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.lower)
        BC_u_upper = da.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        BC_u_corner = da.DirichletBC(self.U.sub(0), da.Constant(0.0), self.corner, method='pointwise')
        BC_d_notch = fe.DirichletBC(self.W, fe.Constant(1.), self.notch, method='pointwise')

        self.BC_u = [BC_u_lower, BC_u_upper, BC_u_corner]
        self.BC_d = []

        # self.presLoad = da.Expression((0, "t"), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        # BC_u_left = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.left)
        # BC_u_right = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.right)
        # self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        # self.BC_d = []


    def build_weak_form_staggered(self):
        self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)

        sigma_plus = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi_plus)
        sigma_minus = cauchy_stress_minus(strain(fe.grad(self.x_new)), self.psi_minus)

        self.G_u = (g_d(self.d_new) * fe.inner(sigma_plus, strain(fe.grad(self.eta))) \
            + fe.inner(sigma_minus, strain(fe.grad(self.eta)))) * fe.dx
 

        # self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        self.G_d = (history(self.H_old, self.psi_plus(strain(fe.grad(self.x_new))), self.psi_cr) * self.zeta * g_d_prime(self.d_new, g_d) \
            + 2 * self.psi_cr * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        # g_c = 0.01
        # self.G_d = (self.psi_plus(strain(fe.grad(self.x_new))) * self.zeta * g_d_prime(self.d_new, g_d) \
        #     + g_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx


        self.G_d += 1 * (self.d_new - self.d_pre) * self.zeta * fe.dx


    def update_history(self):
        psi_new = self.psi_plus(strain(fe.grad(self.x_new)))  
        return psi_new











class HalfCrackSqaure(PDE):
    def __init__(self, args):
        self.case_name = "half_crack_square"
        super(HalfCrackSqaure, self).__init__(args)

        self.displacements = np.concatenate((np.linspace(0, 0.08, 11), np.linspace(0.08, 0.15, 101)))

        # self.displacements = np.linspace(0.0, 0.2, 51)
 
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))

        self.psi_cr = 0.01

        self.mu = 1e3
        self.nu = 0.4
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)
        self.l0 = 2 * self.mesh.hmin()

        self.rho_default = 30
        self.initialize_control_points_and_impact_radii()
        self.d_integrals = []
        self.finish_flag = False
        self.map_flag = True

        # For MFEM
        self.l0 /= 2


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
        self.x = map_function_ufl(self.x_hat, self.control_points, self.impact_radii)  
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

        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.lower)
        BC_u_upper = da.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        BC_u_corner = da.DirichletBC(self.U.sub(0), da.Constant(0.0), self.corner, method='pointwise')
        BC_d_notch = fe.DirichletBC(self.W, fe.Constant(1.), self.notch, method='pointwise')
        self.BC_u = [BC_u_lower, BC_u_upper, BC_u_corner]
        self.BC_d = []

        # self.presLoad = da.Expression((0, "t"), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        # BC_u_left = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.left)
        # BC_u_right = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.right)
        # self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        # self.BC_d = []


        # self.presLoad = da.Expression(("t", 0), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        # BC_u_left = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.left)
        # BC_u_right = da.DirichletBC(self.U.sub(0), da.Constant(0),  self.right)
        # self.BC_u = [BC_u_lower, BC_u_upper] 
        # self.BC_d = []


    def build_weak_form_staggered(self):
        self.x_hat = fe.variable(fe.SpatialCoordinate(self.mesh))
        self.x = map_function_ufl(self.x_hat, self.control_points, self.impact_radii)  
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


        # self.G_d += 0.5 * (self.d_new - self.d_pre) * self.zeta * fe.det(self.grad_gamma) * fe.dx



    def update_history(self):
        psi_new = self.psi_plus(strain(self.mfem_grad(self.x_new)))  
        return psi_new


    def initialize_control_points_and_impact_radii(self):
        # self.control_points = []
        # self.impact_radii = []
        # control_points = np.asarray([[self.length/2, self.height/2]])
        # for new_tip_point in control_points:
        #     self.compute_impact_radii(new_tip_point)

        self.control_points = np.asarray([[self.length/2, self.height/2], [self.length, self.height/2]])
        self.impact_radii = np.array([self.height/4, self.height/4])


    def compute_impact_radius_tip_point(self, P, direct_vec=None):
        radii = np.array([P[0], self.length - P[0], P[1], self.height - P[1]])
        vectors = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
        impact_radius = self.rho_default
        for i in range(len(radii)):
            if direct_vec is not None:
                if np.dot(direct_vec, vectors[i]) >= 0 and radii[i] < impact_radius:
                    impact_radius = radii[i]
            else:
                if radii[i] < impact_radius:
                    impact_radius = radii[i]

        return impact_radius


    def middle_vector(self, v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v_mid = v1 + v2
        if np.linalg.norm(v_mid) < fe.DOLFIN_EPS:
            return np.array([-v1[1], v1[0]])
        else:
            return v_mid / np.linalg.norm(v_mid)


    def inside_domain(self, P):
        return P[0] >= 0 and P[1] >= 0 and P[0] <= self.length and P[1] <= self.height


    def binary_search(self, start_point, end_point):
        assert self.inside_domain(start_point) and not self.inside_domain(end_point)
        tol = 1e-5
        val = np.min(np.array([start_point[0], self.length - start_point[0], start_point[1], self.height - start_point[1]]))
        while val < 0 or val > tol:
            mid_point = (start_point + end_point) / 2
            if self.inside_domain(mid_point):
                start_point = np.array(mid_point)
            else:
                end_point = np.array(mid_point)
            val = np.min(np.array([mid_point[0], self.length - mid_point[0], mid_point[1], self.height - mid_point[1]]))
        return mid_point


    def compute_impact_radius_middle_point(self, P, angle_vec):
        start_point1 = np.array(P)
        end_point1 = start_point1 + angle_vec*1e3
        mid_point1 = self.binary_search(start_point1, end_point1)
        radius1 = np.linalg.norm(mid_point1 - P)

        start_point2 = np.array(P)
        end_point2 = start_point2 - angle_vec*1e3
        mid_point2 = self.binary_search(start_point2, end_point2)
        radius2 = np.linalg.norm(mid_point2 - P)

        return np.min(np.array([radius1, radius2, self.rho_default]))


    def compute_impact_radii(self, new_tip_point):
        assert len(self.control_points) == len(self.impact_radii)
        assert self.inside_domain(new_tip_point)
        self.impact_radii_old = np.array(self.impact_radii)
        if len(self.control_points) == 0:
            self.impact_radii = np.array([self.compute_impact_radius_tip_point(new_tip_point)])
            self.control_points = new_tip_point.reshape(1, -1)
        elif len(self.control_points) == 1:
            self.impact_radii = np.append(self.impact_radii, self.compute_impact_radius_tip_point(new_tip_point, new_tip_point - self.control_points[-1]))
            self.control_points = np.concatenate((self.control_points, new_tip_point.reshape(1, -1)), axis=0)
        else:
            self.impact_radii = self.impact_radii[:-1]
            v1 = self.control_points[-2] - self.control_points[-1]
            v2 = new_tip_point - self.control_points[-1]
            self.impact_radii = np.append(self.impact_radii, self.compute_impact_radius_middle_point(self.control_points[-1], self.middle_vector(v1, v2)))
            self.impact_radii = np.append(self.impact_radii, self.compute_impact_radius_tip_point(new_tip_point, new_tip_point - self.control_points[-1]))
            self.control_points = np.concatenate((self.control_points, new_tip_point.reshape(1, -1)), axis=0)


    def identify_crack_tip(self):

        print('\n')
        print('=================================================================================')
        print('>> Identifying crack tip')
        print('=================================================================================')

        def obj(x):
            p = da.Constant(x)
            x_coo = fe.SpatialCoordinate(self.mesh)
            control_points = list(self.control_points)
            control_points.append(p)
            pseudo_radii = np.zeros(len(control_points))
            distance_field, _ = distance_function_segments_ufl(x_coo, control_points, pseudo_radii)
            d_artificial = fe.exp(-distance_field/self.l0)
            L_tape = fe.assemble((self.d_new - d_artificial)**2 * fe.det(self.grad_gamma) * fe.dx)
            L = float(L_tape)
            return L


        if len(self.control_points) > 1:
            x_initial = self.control_points[-1] + (self.control_points[-1] - self.control_points[-2])
        else:
            raise NotImplementedError("To be implemented!")

        options = {'eps': 1e-15, 'maxiter': 1000, 'disp': True}  # CG > BFGS > Newton-CG
        res = opt.minimize(fun=obj,
                           x0=x_initial,
                           method='CG',
                           callback=None,
                           options=options)

        print("Optimized x is {}".format(res.x))
        print('=================================================================================')

        new_tip_point = map_function_normal(res.x, self.control_points, self.impact_radii)

        return new_tip_point


    def interpolate_H(self):
        control_points_new_map = self.control_points
        control_points_old_map = self.control_points[:-1]
        impact_radii_new_map = self.impact_radii
        impact_radii_old_map = self.impact_radii_old

        inside_domain = self.inside_domain

        class InterpolateExpression(fe.UserExpression):
            def __init__(self, H_old, control_points_new_map, control_points_old_map, impact_radii_new_map, impact_radii_old_map):
                # Construction method of base class has to be called first
                super(InterpolateExpression, self).__init__()
                self.H_old = H_old
                self.control_points_new_map = control_points_new_map
                self.control_points_old_map = control_points_old_map
                self.impact_radii_new_map = impact_radii_new_map
                self.impact_radii_old_map = impact_radii_old_map

            def eval(self, values, x_hat_new):
                x = map_function_normal(x_hat_new, self.control_points_new_map, self.impact_radii_new_map)
                x_hat_old = inverse_map_function_normal(x, self.control_points_old_map, self.impact_radii_old_map)
                point = fe.Point(x_hat_old)

                if not inside_domain(point):
                    print("x_hat_new {}".format(x_hat_new))
                    print("x {}".format(x))
                    print("x_hat_old {}".format(x_hat_old))
                    print("control_points_new_map {}".format(self.control_points_new_map))
                    print("control_points_old_map {}".format(self.control_points_old_map))
                    print("impact_radii_new_map {}".format(self.impact_radii_new_map))
                    print("impact_radii_old_map {}".format(self.impact_radii_old_map))

                values[0] = self.H_old(point)

 
            def value_shape(self):
                return ()

        H_exp = InterpolateExpression(self.H_old, control_points_new_map, control_points_old_map, impact_radii_new_map, impact_radii_old_map)
        # self.H_old.assign(fe.project(H_exp, self.WW))
        self.H_old.assign(fe.interpolate(H_exp, self.WW))


    def update_map(self):
        d_int = fe.assemble(self.d_new * fe.det(self.grad_gamma) * fe.dx)
        print("d_int {}".format(float(d_int)))

        d_integral_interval_initial = 1

        d_integral_interval = 50

        update_flag = False

        while len(self.d_integrals) < len(self.control_points):
            self.d_integrals.append(d_int)

        if len(self.d_integrals) == 0:
            if d_int > d_integral_interval_initial:
                update_flag = True
        else:
            if d_int - self.d_integrals[-1] > d_integral_interval:
                update_flag = True

        if update_flag and not self.finish_flag:
            print('\n')
            print('=================================================================================')
            print('>> Updating map...')
            print('=================================================================================')

            new_tip_point = self.identify_crack_tip()
            if self.inside_domain(new_tip_point):
                v1 = self.control_points[-1] - self.control_points[-2]
                v2 = new_tip_point - self.control_points[-1]
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)

                print("new_tip_point is {}".format(new_tip_point))
                print("v1 is {}".format(v1))
                print("v2 is {}".format(v2))
                print("control points are \n{}".format(self.control_points))
                print("impact_radii are {}".format(self.impact_radii))
                assert np.dot(v1, v2) > np.sqrt(2)/2, "Crack propogration angle not good"
                self.compute_impact_radii(new_tip_point)
                self.interpolate_H()
                self.d_integrals.append(d_int)
                print('=================================================================================')
            else:
                self.finish_flag = True

        else:
            print("Do not modify map")

        print("d_integrals {}".format(self.d_integrals))


def test(args):

    # pde_dc = DoubleCircles(args)
    # pde_dc.monolithic_solve()
    # pde_dc.staggered_solve()

    pde_hc = HalfCrackSqaure(args)
    # pde_hc.monolithic_solve()
    pde_hc.staggered_solve()
 
    # pde_sf = StripeFabric(args)
    # pde_sf.monolithic_solve()
    # pde_sf.staggered_solve()

    fig = plt.figure()
    plt.plot(pde_hc.delta_u_recorded, pde_hc.sigma_recorded, linestyle='--', marker='o', color='red')
    plt.tick_params(labelsize=14)
    plt.xlabel("Vertical displacement of top side", fontsize=14)
    plt.ylabel("Force on top side", fontsize=14)
    fig.savefig('data/pdf/{}/force_load.pdf'.format(pde_hc.case_name), bbox_inches='tight')
    plt.show()


def show_map():
    x1 = np.linspace(0, 0.5, 101)
    y1 = 0.5*x1
    x2 = np.linspace(0.5, 1, 101)
    y2 = -6*x2**3 + 14*x2**2 -9*x2 + 2
    x3 = np.linspace(1, 2, 101)
    y3 = x3
    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x3, y3)
    plt.show()

if __name__ == '__main__':
    args = arguments.args
    test(args)
    # show_map()