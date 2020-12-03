import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os


def distance_function(P, A=[-1, 0], B=[1, 0]):     
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
  
    BP = [None, None]
    BP[0] = P[0] - B[0]
    BP[1] = P[1] - B[1]

    AP = [None, None]
    AP[0] = P[0] - A[0]
    AP[1] = P[1] - A[1]
  
    AB_BP = AB[0] * BP[0] + AB[1] * BP[1]
    AB_AP = AB[0] * AP[0] + AB[1] * AP[1]
  
    y = P[1] - B[1]
    x = P[0] - B[0]
    df1 = fe.sqrt(x * x + y * y) 

    y = P[1] - A[1]
    x = P[0] - A[0]
    df2 = fe.sqrt(x * x + y * y)

    x1 = AB[0]
    y1 = AB[1]
    x2 = AP[0]
    y2 = AP[1]
    mod = fe.sqrt(x1 * x1 + y1 * y1)
    df3 = np.absolute(x1 * y2 - y1 * x2) / mod

    df = fe.conditional(fe.gt(AB_BP, 0), df1, fe.conditional(fe.lt(AB_AP, 0), df2, df3))

    return df


def ratio_function(ratio):
    return fe.conditional(fe.lt(ratio, 1), ratio**2, ratio)

# def ratio_function(ratio):
#     return fe.conditional(fe.lt(ratio, 1), 1 - fe.exp(1 + 1 / (ratio**3 - 1)), ratio)


def map_function(x_hat):
    rho = 1
    x_hat = fe.variable(x_hat)
    df = distance_function(x_hat)
    grad_x_hat = fe.diff(df, x_hat)
 
    ratio = df / rho
    delta_x = grad_x_hat * (rho * ratio_function(ratio) - df)
    return delta_x



def mfem():
    files = glob.glob('data/pvd/mfem/*')
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print('Failed to delete {}, reason: {}' % (f, e))

    plate = mshr.Rectangle(fe.Point(-2, -2), fe.Point(2, 2))
    mesh = mshr.generate_mesh(plate, 30)
    # mesh = fe.RectangleMesh(fe.Point(-2, -2), fe.Point(2, 2), 50, 50)

    U = fe.VectorFunctionSpace(mesh, 'CG', 2)

    V = fe.FunctionSpace(mesh, "CG", 1)
    d = fe.interpolate(fe.Constant(0), V)

    u = fe.TrialFunction(U)
    v = fe.TestFunction(U)

    x_hat = fe.SpatialCoordinate(mesh)
    delta_x = map_function(x_hat)

    u = fe.project(delta_x, U)

    # a = fe.dot(u, v) * fe.dx
    # L = fe.dot(delta_x, v) * fe.dx

    # u = fe.Function(U)
    # fe.solve(a == L, u, [])

    vtkfile_u = fe.File('data/pvd/mfem/u.pvd')
    u.rename("u", "u")
    vtkfile_u << u

    vtkfile_d = fe.File('data/pvd/mfem/d.pvd')
    d.rename("d", "d")
    vtkfile_d << d


if __name__ == '__main__':
    mfem()
