'''To compare with Matlab results
'''
import fenics as fe
import dolfin_adjoint as da
import meshio
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import h5py
import os
from scipy.io import loadmat
from functools import partial
import scipy.optimize as opt
from pyadjoint.overloaded_type import create_overloaded_object
from .. import arguments
from ..constitutive import *
from ..pde import PDE


class InternalCrack(PDE):
    def __init__(self, args):
        self.case_name = "matlab_internal_crack"
        super(InternalCrack, self).__init__(args)

        self.displacements = 1e-5*np.linspace(0, 1, 11)
        # self.displacements = 1e-5*np.concatenate((np.arange(0.3, 0.6980), np.arange(0.6980, 0.699, 0.0001)))
        
        self.relaxation_parameters =  np.linspace(1, 1, len(self.displacements))

        self.l0 = 0.3
        self.E = 210e3
        self.nu = 0.3
        self.mu = self.E / (2 * (1 + self.nu))
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)

        self.staggered_tol = 1e-10
        self.staggered_maxiter = 10000


    def build_mesh(self):
        # Read mesh from matlab file .mat and convert it to FEniCS mesh
        mesh_matlab = loadmat('data/mat/mesh/PF-1.mat')
        points = mesh_matlab['p']
        cells = mesh_matlab['t']
        points = points.T
        cells = [("triangle", (cells[:-1, :] - 1).T)]
        meshio.write_points_cells('data/xdmf/{}/mesh.xdmf'.format(self.case_name), points, cells)
        xdmf_mesh = fe.XDMFFile('data/xdmf/{}/mesh.xdmf'.format(self.case_name))
        self.mesh = fe.Mesh()
        xdmf_mesh.read(self.mesh)

        self.length = 1
        self.height = 1
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

        class Middle(fe.SubDomain):
            def inside(self, x, on_boundary):
                return  np.absolute(x[1] - 0.5) < 1e-2 and (x[0] - 0.3) * (x[0] - 0.7) < 1e-7

        self.lower = Lower()
        self.upper = Upper()
        self.corner = Corner()
        self.middle = Middle()


    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)

        self.presLoad = da.Expression((0, "t"), t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 

        BC_d_middle = fe.DirichletBC(self.W, fe.Constant(1.), self.middle, method='pointwise')

        self.BC_u = [BC_u_lower, BC_u_upper]
        self.BC_d = [BC_d_middle]


    def build_weak_form_staggered(self):
        self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)

        sigma_plus = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi_plus)
        sigma_minus = cauchy_stress_minus(strain(fe.grad(self.x_new)), self.psi_minus)

        self.G_u = (g_d(self.d_new) * fe.inner(sigma_plus, strain(fe.grad(self.eta))) \
            + fe.inner(sigma_minus, strain(fe.grad(self.eta)))) * fe.dx
 
        g_c = 2.7e-6;
        self.G_d = (self.psi_plus(strain(fe.grad(self.x_new))) * self.zeta * g_d_prime(self.d_new, g_d) \
            + g_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx


def test(args):
    pde = InternalCrack(args)
    pde.staggered_solve()

    plt.figure()
    plt.plot(pde.delta_u_recorded, pde.sigma_recorded, linestyle='--', marker='o', color='red')
    plt.tick_params(labelsize=14)
    plt.xlabel("Vertical displacement of top side", fontsize=14)
    plt.ylabel("Force on top side", fontsize=14)
    plt.show()


if __name__ == '__main__':
    args = arguments.args
    test(args)