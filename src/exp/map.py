import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os


class FractureExpression(fe.UserExpression):

    def eval(self, values, x):
        if x[0] > -1 and x[0] < 1 and x[1] > -0.05 and x[1] < 0.05:
            values[0] = 1
        else:
            values[0] = 0

    def value_shape(self):
        return ()


class LevelSetExpression(fe.UserExpression):

    def eval(self, values, x):
        # values[0] = fe.sqrt(x[0]**2 + x[1]**2) - 1
        values[0] = x[0]**2 + x[1]**2

    def value_shape(self):
        return ()


def d3(grad_u):
    norm = fe.sqrt(fe.inner(grad_u, grad_u))
    return fe.conditional(fe.gt(norm, 1), 1 - 1/norm, 1 - (2 - norm))
    # return 1 - 1/norm


def map():
    '''
    Try to reproduce https://doi.org/10.1016/j.cma.2014.11.016
    '''
    files = glob.glob('data/pvd/map/*')
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print('Failed to delete {}, reason: {}' % (f, e))

    # plate = mshr.Rectangle(fe.Point(-2, -2), fe.Point(2, 2))
    # mesh = mshr.generate_mesh(plate, 50)
    mesh = fe.RectangleMesh(fe.Point(-2, -2), fe.Point(2, 2), 50, 50)


    V = fe.FunctionSpace(mesh, "CG", 1)
    d = fe.interpolate(FractureExpression(), V)

    u = fe.TrialFunction(V)
    v  = fe.TestFunction(V)

    u_k = fe.interpolate(LevelSetExpression(), V)
    # u_k = fe.project(1 - d, V)
 
    class CenterPoint(fe.SubDomain):
        def inside(self, x, on_boundary):                    
            return x[0] > -1 and x[0] < 1 and x[1] > -0.05 and x[1] < 0.05

    BC_u = fe.DirichletBC(V, fe.Constant(0.0), CenterPoint(), method='pointwise')
    # BC = [BC_u]     
    BC = []

    a = fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    a += 100 * u * v * d * fe.dx
    L = fe.inner(fe.grad(u_k), fe.grad(v)) * (1 - d3(fe.grad(u_k))) * fe.dx 
    # L += -100*d * v * fe.dx 

    # E = 0.5*(fe.sqrt(fe.inner(fe.grad(u), fe.grad(u))) - 1)**2 * fe.dx
    # dE = fe.derivative(E, u, v)
    # jacE = fe.derivative(dE, u, du)
    # fe.solve(dE == 0, u, [], J=jacE, solver_parameters={'newton_solver': {'relaxation_parameter': 0.1, 'maximum_iterations':1000}})

    u = fe.Function(V)
    eps = 1.0           # error measure ||u-u_k||
    tol = 1e-5       # tolerance
    iteration = 0            # iteration counter
    maxiter = 25        # max no of iterations allowed
    while eps > tol and iteration < maxiter:
        iteration += 1
        fe.solve(a == L, u, BC)
        diff = np.array(u.vector()) - np.array(u_k.vector())
        eps = np.linalg.norm(diff, ord=np.Inf)
        # eps = fe.errornorm(u, u_k, norm_type='l2', mesh=None)
        print('iteration={}: norm={}'.format(iteration, eps))
        u_k.assign(u)   # update for next iteration


    vtkfile_u = fe.File('data/pvd/map/u.pvd')
    u.rename("phi", "phi")
    vtkfile_u << u

    vtkfile_d = fe.File('data/pvd/map/d.pvd')
    d.rename("d", "d")
    vtkfile_d << d


if __name__ == '__main__':
    map()
