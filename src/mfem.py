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


def ratio_function(ratio):
    return ratio**2


def inverse_ratio_function(ratio):
    return fe.sqrt(ratio)


def distance_function_line_segement_ufl(P, A=[-1, 0], B=[1, 0]):     
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


def distance_function_point_ufl(P, A=[0, 0]):
    x = P[0] - A[0]
    y = P[1] - A[1]
    return fe.sqrt(x * x + y * y)


def distance_function_segments_ufl(P, control_points, impact_radii):
    if len(control_points) == 1:
        return distance_function_point_ufl(P, control_points[0]), impact_radii[0]
    else:
        rho1 = impact_radii[0]
        rho2 = impact_radii[1]
        df, xi = distance_function_line_segement_ufl(P, control_points[0], control_points[1])
        for i in range(len(control_points) - 1):
            tmp_df, tmp_xi = distance_function_line_segement_ufl(P, control_points[i], control_points[i + 1])
            xi = fe.conditional(fe.lt(tmp_df, df), tmp_xi, xi)
            rho1 = fe.conditional(fe.lt(tmp_df, df), impact_radii[i], rho1)
            rho2 = fe.conditional(fe.lt(tmp_df, df), impact_radii[i + 1], rho2)
            df = ufl.Min(tmp_df, df)
        return df, (1 - xi) * rho1 + xi * rho2
 

def map_function_ufl(x_hat, control_points, impact_radii):
    if len(control_points) == 0:
        return x_hat
    x_hat = fe.variable(x_hat)
    df, rho = distance_function_segments_ufl(x_hat, control_points, impact_radii)
    grad_x_hat = fe.diff(df, x_hat)
    delta_x_hat = fe.conditional(fe.gt(df, rho), fe.Constant((0., 0.)), grad_x_hat * (rho * ratio_function(df / rho) - df))
    return delta_x_hat + x_hat


def inverse_map_function_ufl(x, control_points, impact_radii):
    if len(control_points) == 0:
        return x
    x = fe.variable(x)
    df, rho = distance_function_segments_ufl(x, control_points, impact_radii)
    grad_x = fe.diff(df, x)
    delta_x = fe.conditional(fe.gt(df, rho), fe.Constant((0., 0.)), grad_x * (rho * inverse_ratio_function(df / rho) - df))
    return delta_x + x


def distance_function_line_segement_normal(P, A=[-1, 0], B=[1, 0]):     
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
    df1 = np.sqrt(x**2 + y**2) 
    xi1 = 1

    y = P[1] - A[1]
    x = P[0] - A[0]
    df2 = np.sqrt(x**2 + y**2)
    xi2 = 0

    x1 = AB[0]
    y1 = AB[1]
    x2 = AP[0]
    y2 = AP[1]
    mod = np.sqrt(x1**2 + y1**2)
    df3 = np.absolute(x1 * y2 - y1 * x2) / mod
    xi3 = np.sqrt(x2**2 + y2**2 - df3**2) / mod

    if AB_BP > 0:
        df = df1
        xi = xi1
    elif AB_AP < 0:
        df = df2
        xi = xi2
    else:
        df = df3
        xi = xi3
 
    return df, xi


def distance_function_point_normal(P, A=[0, 0]):
    x = P[0] - A[0]
    y = P[1] - A[1]
    return np.sqrt(x * x + y * y)


def distance_function_segments_normal(P, control_points, impact_radii):
    if len(control_points) == 1:
        return distance_function_point_normal(P, control_points[0]), impact_radii[0], control_points[0]
    else:
        rho1 = impact_radii[0]
        rho2 = impact_radii[1]
        point1 = control_points[0]
        point2 = control_points[1]
        df, xi = distance_function_line_segement_normal(P, control_points[0], control_points[1])
        for i in range(len(control_points) - 1):
            tmp_df, tmp_xi = distance_function_line_segement_normal(P, control_points[i], control_points[i + 1])
            xi =  tmp_xi if tmp_df < df else xi
            rho1 = impact_radii[i] if tmp_df < df else rho1
            rho2 = impact_radii[i + 1] if tmp_df < df else rho2
            point1 = control_points[i] if tmp_df < df else point1
            point2 = control_points[i + 1] if tmp_df < df else point2
            df = min(tmp_df, df)

        return df, (1 - xi) * rho1 + xi * rho2, (1 - xi) * point1 + xi * point2
 


def map_function_normal(x_hat, control_points, impact_radii):
    if len(control_points) == 0:
        return x_hat
    df, rho, point = distance_function_segments_normal(x_hat, control_points, impact_radii)
    vec_dist = x_hat - point
    if df <= 0 or df >= rho:
        delta_x_hat = np.array([0., 0.])
    else:
        delta_x_hat =  vec_dist / df * (rho * ratio_function(df / rho) - df)
    return delta_x_hat + x_hat


def inverse_map_function_normal(x, control_points, impact_radii):
    if len(control_points) == 0:
        return x
    df, rho, point = distance_function_segments_normal(x, control_points, impact_radii)
    vec_dist = x - point
    if df <= 0 or df >= rho:
        delta_x = np.array([0., 0.])
    else:
        delta_x =  vec_dist / df * (rho * inverse_ratio_function(df / rho) - df)
    return delta_x + x



class InterpolateExpression(fe.UserExpression):

    def __init__(self, e, control_points, impact_radii):
        # Construction method of base class has to be called first
        super(InterpolateExpression, self).__init__()
        self.e = e
        self.control_points = control_points
        self.impact_radii = impact_radii

    def eval(self, values, x_hat):
        x = inverse_map_function_normal(map_function_normal(x_hat, self.control_points, self.impact_radii), self.control_points, self.impact_radii)
        point = fe.Point(x)
        values[0] = self.e(point)
        # delta_x_hat = x - x_hat
        # values[0] = delta_x_hat[0]
        # values[1] = delta_x_hat[1] 

    def value_shape(self):
        return ()



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
    W = fe.FunctionSpace(mesh, "DG", 0)

    n = 21
    control_points = np.stack((np.linspace(0, 1, n), np.linspace(0, 1, n)), axis=1)
    impact_radii = np.linspace(0., 0.5, n)


    df, xi = distance_function_segments_ufl(x_hat, control_points, impact_radii)
    d = fe.project(df, V)

    delta_x = map_function_ufl(x_hat, control_points, impact_radii) - x_hat
    u = fe.project(delta_x, U)

    # int_exp = InterpolateExpression(u, control_points, impact_radii)
    # e = fe.interpolate(int_exp, U)

    e = fe.Function(W)
    int_exp = InterpolateExpression(e, control_points, impact_radii)
    e = fe.project(int_exp, W)

 
    vtkfile_u = fe.File('data/pvd/mfem/u.pvd')
    u.rename("u", "u")
    vtkfile_u << u

    vtkfile_d = fe.File('data/pvd/mfem/d.pvd')
    d.rename("d", "d")
    vtkfile_d << d

    vtkfile_e = fe.File('data/pvd/mfem/e.pvd')
    e.rename("e", "e")
    vtkfile_e << e



if __name__ == '__main__':
    mfem()
