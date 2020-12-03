import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
import ufl

fe.parameters["form_compiler"]["quadrature_degree"] = 4


def distance_function_line_segement(P, A=[-1, 0], B=[1, 0]):     
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


def distance_function_point(P, A=[0, 0]):
    x = P[0] - A[0]
    y = P[1] - A[1]
    return fe.sqrt(x * x + y * y)


def distance_function_segments(P, points):
    if len(points) == 1:
        return distance_function_point(P, points[0])
    else:
        distance = distance_function_line_segement(P, points[0], points[1])
        for i in range(len(points) - 1):
            tmp = distance_function_line_segement(P, points[i], points[i + 1])
            distance = ufl.Min(distance, tmp)

    return distance
 

def ratio_function(ratio):
    return fe.conditional(fe.lt(ratio, 1), ratio**2, ratio)


def map_function(x_hat):
    rho = 0.5
    x_hat = fe.variable(x_hat)
    df = distance_function_line_segement(x_hat)
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
    mesh = mshr.generate_mesh(plate, 50)
    # mesh = fe.RectangleMesh(fe.Point(-2, -2), fe.Point(2, 2), 50, 50)

    x_hat = fe.SpatialCoordinate(mesh)

    U = fe.VectorFunctionSpace(mesh, 'CG', 2)
    V = fe.FunctionSpace(mesh, "CG", 1)

    points = [[-2, 0], [-1, 0], [0, 0], [1, 1]]
    n = 100
    points_long = np.stack((np.linspace(0, 1, n), np.linspace(0, 1, n)), axis=1)

    dist = distance_function_segments(x_hat, points_long)
    d = fe.project(dist, V)

    # u = fe.TrialFunction(V)
    # v = fe.TestFunction(V)
    # a = fe.dot(u, v) * fe.dx
    # L = fe.dot(dist, v) * fe.dx
    # d = fe.Function(V)
    # fe.solve(a == L, d, [])

    delta_x = map_function(x_hat)
    u = fe.project(delta_x, U)

    vtkfile_u = fe.File('data/pvd/mfem/u.pvd')
    u.rename("u", "u")
    vtkfile_u << u

    vtkfile_d = fe.File('data/pvd/mfem/d.pvd')
    d.rename("d", "d")
    vtkfile_d << d


if __name__ == '__main__':
    mfem()
