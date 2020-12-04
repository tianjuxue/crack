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
    df1 = fe.sqrt(x**2 + y**2) 
    xi1 = 1

    y = P[1] - A[1]
    x = P[0] - A[0]
    df2 = fe.sqrt(x**2 + y**2)
    xi2 = 0

    x1 = AB[0]
    y1 = AB[1]
    x2 = AP[0]
    y2 = AP[1]
    mod = fe.sqrt(x1**2 + y1**2)
    df3 = np.absolute(x1 * y2 - y1 * x2) / mod
    xi3 = fe.conditional(fe.gt(x2**2 + y2**2 - df3**2, 0), fe.sqrt(x2**2 + y2**2 - df3**2) / mod, 0)

    df = fe.conditional(fe.gt(AB_BP, 0), df1, fe.conditional(fe.lt(AB_AP, 0), df2, df3))
    xi = fe.conditional(fe.gt(AB_BP, 0), xi1, fe.conditional(fe.lt(AB_AP, 0), xi2, xi3))

    return df, xi


def distance_function_point(P, A=[0, 0]):
    x = P[0] - A[0]
    y = P[1] - A[1]
    return fe.sqrt(x * x + y * y)


def distance_function_segments(P, points, impact_radii):

    if len(points) == 1:
        return distance_function_point(P, points[0]), impact_radii[0]
    else:
        rho1 = impact_radii[0]
        rho2 = impact_radii[1]
        df, xi = distance_function_line_segement(P, points[0], points[1])
        for i in range(len(points) - 1):
            tmp_df, tmp_xi = distance_function_line_segement(P, points[i], points[i + 1])
            xi = fe.conditional(fe.lt(tmp_df, df), tmp_xi, xi)
            rho1 = fe.conditional(fe.lt(tmp_df, df), impact_radii[i], rho1)
            rho2 = fe.conditional(fe.lt(tmp_df, df), impact_radii[i + 1], rho2)
            df = ufl.Min(tmp_df, df)

        return df, (1 - xi) * rho1 + xi * rho2
 

def ratio_function(ratio):
    return fe.conditional(fe.lt(ratio, 1), ratio**2, ratio)


def map_function(x_hat, points, impact_radii):
    x_hat = fe.variable(x_hat)
    df, rho = distance_function_segments(x_hat, points, impact_radii)
    grad_x_hat = fe.diff(df, x_hat)
    # TODO: no division
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

    n = 101
    points = np.stack((np.linspace(0, 1, n), np.linspace(0, 1, n)), axis=1)
    impact_radii = np.linspace(1e-10, 0.5, n)

    dist, xi = distance_function_segments(x_hat, points, impact_radii)
    d = fe.project(dist, V)

    delta_x = map_function(x_hat, points, impact_radii)
    u = fe.project(delta_x, U)

    vtkfile_u = fe.File('data/pvd/mfem/u.pvd')
    u.rename("u", "u")
    vtkfile_u << u

    vtkfile_d = fe.File('data/pvd/mfem/d.pvd')
    d.rename("d", "d")
    vtkfile_d << d


if __name__ == '__main__':
    mfem()
