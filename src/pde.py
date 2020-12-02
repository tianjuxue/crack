import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
from . import arguments
from .constitutive import *


fe.parameters["form_compiler"]["quadrature_degree"] = 4

# sigma_c = 2
# psi_cr = sigma_c**2 / (2 * E)

psi_cr = 0.03

Gc_0 = 0.1
l0 = 1.


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

        self.set_bcs_mono()

        WW = fe.FunctionSpace(self.mesh, 'DG', 0) 
        EE = fe.FunctionSpace(self.mesh, 'CG', 1) 

        m_test = fe.TestFunctions(self.M)
        m_delta = fe.TrialFunctions(self.M)
        m_new = fe.Function(self.M)

        self.eta, self.zeta = m_test
        self.x_new, self.d_new = fe.split(m_new)

        self.H_old = fe.Function(WW)
        E = fe.Function(EE)

        self.build_weak_form_mono()

        dG = fe.derivative(self.G, m_new)
        p = fe.NonlinearVariationalProblem(self.G, m_new, self.BC, dG)
        solver = fe.NonlinearVariationalSolver(p)

        vtkfile_u = fe.File('data/pvd/{}/u.pvd'.format(self.args.case_name))
        vtkfile_d = fe.File('data/pvd/{}/d.pvd'.format(self.args.case_name))
        vtkfile_e = fe.File('data/pvd/{}/e.pvd'.format(self.args.case_name))

        for disp in self.args.displacements:
            print(' ')
            print('=================================================================================')
            print('>> disp boundary condition = {} [mm]'.format(disp))
            print('=================================================================================')

            self.presLoad.t = disp

            newton_prm = solver.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 1000
            newton_prm['linear_solver'] = 'mumps'   
            newton_prm['absolute_tolerance'] = 1e-4

            if disp > 11 and disp <= 14 :
                newton_prm['relaxation_parameter'] = 0.2
            elif disp > 14 and disp <= 26.5:
                newton_prm['relaxation_parameter'] = 0.1
            elif disp > 26.5:
                newton_prm['relaxation_parameter'] = 0.02

            solver.solve()

            self.H_old.assign(fe.project(H(self.x_new, self.H_old, self.I, psi_cr), WW))

            E.assign(fe.project(psi(self.I + fe.grad(self.x_new)), EE))
            # E.assign(fe.project(first_PK_stress(self.I + fe.grad(x_new))[0, 0], EE))
            
            print('=================================================================================')
            print(' ')

            x_plot, d_plot = m_new.split()
            x_plot.rename("Displacement", "label")
            d_plot.rename("Phase field", "label")

            vtkfile_u << x_plot
            vtkfile_d << d_plot
            vtkfile_e << self.H_old 


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
        self.length = length

    def set_bcs_mono(self):
        length = self.length
        class Left(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0)

        class Right(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], length)

        class Corner(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return fe.near(x[0], 0) and fe.near(x[1], 0)

        self.presLoad = fe.Expression("t", t=0.0, degree=1)
        BC_u_left = fe.DirichletBC(self.M.sub(0).sub(0), fe.Constant(0),  Left())
        BC_u_right = fe.DirichletBC(self.M.sub(0).sub(0), self.presLoad,  Right())
        BC_u_corner = fe.DirichletBC(self.M.sub(0).sub(1), fe.Constant(0.0), Corner(), method='pointwise')
        self.BC = [BC_u_left, BC_u_right, BC_u_corner] 
        # right.mark(self.boundaries, 1)

    def build_weak_form_mono(self):
        G_ut = g_d(self.d_new) * fe.inner(first_PK_stress(self.I + fe.grad(self.x_new)), fe.grad(self.eta)) * fe.dx
        G_d = self.H_old * self.zeta * g_d_prime(self.d_new, g_d) * fe.dx \
            + 2 * psi_cr * (self.zeta * self.d_new + l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new))) * fe.dx  
        self.G = G_ut + G_d


def test(args):
    args.case_name = "circular_holes"
    args.displacements = np.concatenate((np.linspace(1, 11, 6), np.linspace(12, 26.5, 30), np.linspace(27, 40, 53)))
    pde = DoubleCircles(args)
    pde.monolithic_solve()


if __name__ == '__main__':
    args = arguments.args
    test(args)