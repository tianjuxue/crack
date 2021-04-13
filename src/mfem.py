import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
import ufl
from . import arguments

fe.parameters["form_compiler"]["quadrature_degree"] = 4


def smooth_combo(x):
    if x < -1:
        y = x
    elif x < -1./2.:
        y = -6 * x**3 - 14 * x**2 - 9 * x - 2
    elif x < 1./2.:
        y = 1./2. * x
    elif x < 1:
        y = -6 * x**3 + 14 * x**2 - 9 * x + 2
    else:
        y = x
    return y


def inverse_smooth_combo(y):
    assert y >=0 and y <= 1
    if y <= 1./4.:
        return 2 * y
    tol = 1e-10
    start_x = y
    end_x = 1.
    assert smooth_combo(start_x) <= y and y <= smooth_combo(end_x)
    abs_err = 1e10
    while abs_err > tol:
        mid_x = (start_x + end_x) / 2.
        signed_error = smooth_combo(mid_x) - y
        if signed_error > 0:
            end_x = mid_x
        else:
            start_x = mid_x
        abs_err = np.absolute(signed_error)
    assert mid_x >= 0 and mid_x <= 1
    return mid_x


def ratio_function_ufl(ratio, map_type):
    if map_type == 'linear':
        return fe.conditional(fe.lt(ratio, -1./2.), 3./2.* ratio + 1./2., fe.conditional(fe.gt(ratio, 1./2.), 3./2.* ratio - 1./2., 1./2.* ratio))
    elif map_type == 'power':
        return ratio**2
    elif map_type == 'identity':
        return ratio
    elif map_type == 'smooth':
        return fe.conditional(fe.lt(ratio, -1./2.), -6*ratio**3 - 14*ratio**2 - 9*ratio - 2, \
               fe.conditional(fe.gt(ratio, 1./2.), -6*ratio**3 + 14*ratio**2 - 9*ratio + 2, 1./2.* ratio))       
    else:
        raise NotImplementedError("To be implemented")


def inverse_ratio_function_ufl(ratio, map_type):
    if map_type == 'linear':
        return fe.conditional(fe.lt(ratio, -1./4.), 2./3.* ratio - 1./3., fe.conditional(fe.gt(ratio, 1./4.), 2./3.* ratio + 1./3., 2. * ratio))
    elif map_type == 'power':
        return ratio**2
    elif map_type == 'identity':
        return ratio
    else:
        raise NotImplementedError("To be implemented")


def ratio_function_normal(ratio, map_type):
    if map_type == 'linear':
        if ratio <= 1./2 and ratio >= -1./2:
            return 1./2.*ratio
        elif ratio > 1/2:
            return 3./2.*ratio - 1./2.
        else:
            return 3./2.*ratio + 1./2.
    elif map_type == 'power':
        return ratio**(1.5)
    elif map_type == 'identity':
        return ratio
    elif map_type == 'smooth':
        return smooth_combo(ratio)
    else:
        raise NotImplementedError("To be implemented")


def inverse_ratio_function_normal(ratio, map_type):
    if map_type == 'linear':
        if ratio <= 1./4 and ratio >= -1./4:
            return 2.*ratio
        elif ratio > 1/4:
            return 2./3.*ratio + 1./3.
        else:
            return 2./3.*ratio - 1./3.
    elif map_type == 'power':
        return ratio**(1/1.5)
    elif map_type == 'identity':
        return ratio
    elif map_type == 'smooth':
        return inverse_smooth_combo(ratio)
    else:
        raise NotImplementedError("To be implemented")


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
 

def map_function_ufl(x_hat, control_points, impact_radii, map_type, boundary_info=None):
    if len(control_points) == 0:
        return x_hat
    x_hat = fe.variable(x_hat)
    df, rho = distance_function_segments_ufl(x_hat, control_points, impact_radii)
    grad_x_hat = fe.diff(df, x_hat)
    delta_x_hat = fe.conditional(fe.gt(df, rho), fe.Constant((0., 0.)), grad_x_hat * (rho * ratio_function_ufl(df / rho, map_type) - df))
    if boundary_info is None:
        return delta_x_hat + x_hat
    else: 
        last_control_point = control_points[-1]
        points, directions, rho_default = boundary_info
        mid_point, mid_point1, mid_point2 = points
        direct_vec, rotated_vec = directions
        aux_control_point1 = last_control_point + rho_default * rotated_vec
        aux_control_point2 = last_control_point - rho_default * rotated_vec

        w1 = np.linalg.norm(mid_point1 - aux_control_point1)
        w2 = np.linalg.norm(mid_point2 - aux_control_point2)
        w0 = np.linalg.norm(mid_point - last_control_point)

        assert np.absolute(2*w0 - w1 - w2) < 1e-5

        AB = mid_point - last_control_point
        AP = x_hat - last_control_point

        x1 = AB[0]
        y1 = AB[1]
        x2 = AP[0]
        y2 = AP[1]

        mod = fe.sqrt(x1**2 + y1**2)
        df_to_direct = (x1 * y2 - y1 * x2) / mod  # AB x AP
        df_to_rotated = (x1 * x2 + y1 * y2) / mod

        k1 = rho_default * (w1 + w2) / (rho_default * (w1 + w2) + df_to_direct * (w1 - w2))

        new_df_to_direct = rho_default * ratio_function_ufl(df_to_direct / rho_default, map_type)

        k2 = rho_default * (w1 + w2) / (rho_default * (w1 + w2) + new_df_to_direct * (w1 - w2))

        new_df_to_rotated = df_to_rotated * k1 / k2
 
        x = fe.as_vector(last_control_point + direct_vec * new_df_to_rotated + rotated_vec * new_df_to_direct) 

        return fe.conditional(fe.gt(df_to_rotated, 0), fe.conditional(fe.gt(np.absolute(df_to_direct), rho), x_hat, x), delta_x_hat + x_hat)
 

def inverse_map_function_ufl(x, control_points, impact_radii, map_type):
    if len(control_points) == 0:
        return x
    x = fe.variable(x)
    df, rho = distance_function_segments_ufl(x, control_points, impact_radii)
    grad_x = fe.diff(df, x)
    delta_x = fe.conditional(fe.gt(df, rho), fe.Constant((0., 0.)), grad_x * (rho * inverse_ratio_function_ufl(df / rho, map_type) - df))
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

    if x2**2 + y2**2 - df3**2 > 0:
        xi3 = np.sqrt(x2**2 + y2**2 - df3**2) / mod
    else:
        xi3 = 0

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
 


def map_function_normal(x_hat, control_points, impact_radii, map_type, boundary_info=None):
    if len(control_points) == 0:
        return x_hat
    df, rho, point = distance_function_segments_normal(x_hat, control_points, impact_radii)
    vec_dist = x_hat - point
    if df <= 0 or df >= rho:
        delta_x_hat = np.array([0., 0.])
    else:
        delta_x_hat =  vec_dist / df * (rho * ratio_function_normal(df / rho, map_type) - df)
 
    if boundary_info is None:
        return delta_x_hat + x_hat
    else: 
        last_control_point = control_points[-1]
        points, directions, rho_default = boundary_info
        mid_point, mid_point1, mid_point2 = points
        direct_vec, rotated_vec = directions
        aux_control_point1 = last_control_point + rho_default * rotated_vec
        aux_control_point2 = last_control_point - rho_default * rotated_vec

        w1 = np.linalg.norm(mid_point1 - aux_control_point1)
        w2 = np.linalg.norm(mid_point2 - aux_control_point2)
        w0 = np.linalg.norm(mid_point - last_control_point)

        assert np.absolute(2*w0 - w1 - w2) < 1e-5

        AB = mid_point - last_control_point
        AP = x_hat - last_control_point

        x1 = AB[0]
        y1 = AB[1]
        x2 = AP[0]
        y2 = AP[1]

        mod = np.sqrt(x1**2 + y1**2)
        df_to_direct = (x1 * y2 - y1 * x2) / mod  # AB x AP
        df_to_rotated = (x1 * x2 + y1 * y2) / mod

        k1 = rho_default * (w1 + w2) / (rho_default * (w1 + w2) + df_to_direct * (w1 - w2))

        new_df_to_direct = rho_default * ratio_function_normal(df_to_direct / rho_default, map_type)

        k2 = rho_default * (w1 + w2) / (rho_default * (w1 + w2) + new_df_to_direct * (w1 - w2))

        new_df_to_rotated = df_to_rotated * k1 / k2
 
        x = last_control_point + direct_vec * new_df_to_rotated + rotated_vec * new_df_to_direct

        if df_to_rotated > 0:
            if np.absolute(df_to_direct) > rho:
                return x_hat
            else:
                return x
        else:
            return delta_x_hat + x_hat


def inverse_map_function_normal(x, control_points, impact_radii, map_type):
    if len(control_points) == 0:
        return x
    df, rho, point = distance_function_segments_normal(x, control_points, impact_radii)
    vec_dist = x - point
    if df <= 0 or df >= rho:
        delta_x = np.array([0., 0.])
    else:
        delta_x =  vec_dist / df * (rho * inverse_ratio_function_normal(df / rho, map_type) - df)
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
        # values[0] = self.e(point)
        # delta_x_hat = x - x_hat
        values = x - x_hat
 
    def value_shape(self):
        return (2,)


def mfem():
    files = glob.glob('data/pvd/mfem/*')
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print('Failed to delete {}, reason: {}' % (f, e))

    plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(100, 100))
    mesh = mshr.generate_mesh(plate, 50)
    # mesh = fe.RectangleMesh(fe.Point(-2, -2), fe.Point(2, 2), 50, 50)

    x_hat = fe.SpatialCoordinate(mesh)

    U = fe.VectorFunctionSpace(mesh, 'CG', 2)
    V = fe.FunctionSpace(mesh, "CG", 1)
    W = fe.FunctionSpace(mesh, "DG", 0)

    # n = 21
    # control_points = np.stack((np.linspace(0, 1, n), np.linspace(0, 1, n)), axis=1)
    # impact_radii = np.linspace(0., 0.5, n)

    rho_default = 25. / np.sqrt(5) * 2 
    control_points = np.array([[50., 50.], [62.5, 25.]])
    impact_radii = np.array([rho_default, rho_default])

    mid_point = np.array([75., 0.])
    mid_point1 = np.array([100., 0.])
    mid_point2 = np.array([50., 0.])
    points = [mid_point, mid_point1, mid_point2]
    direct_vec = np.array([1., -2])
    rotated_vec = np.array([2., 1.])

    direct_vec /= np.linalg.norm(direct_vec)
    rotated_vec /= np.linalg.norm(rotated_vec)

    directions = [direct_vec, rotated_vec]
    boundary_info = [points, directions, rho_default]


    # df, xi = distance_function_segments_ufl(x_hat, control_points, impact_radii)
    # d = fe.project(df, V)

    delta_x = map_function_ufl(x_hat, control_points, impact_radii, boundary_info) - x_hat
    u = fe.project(delta_x, U)

    # e = fe.Function(U)
    # int_exp = InterpolateExpression(u, control_points, impact_radii)
    # e = fe.interpolate(int_exp, U)
    # int_exp = InterpolateExpression(e, control_points, impact_radii)
    # e = fe.project(int_exp, U)

 
    vtkfile_u = fe.File('data/pvd/mfem/u.pvd')
    u.rename("u", "u")
    vtkfile_u << u

    # vtkfile_d = fe.File('data/pvd/mfem/d.pvd')
    # d.rename("d", "d")
    # vtkfile_d << d

    # vtkfile_e = fe.File('data/pvd/mfem/e.pvd')
    # e.rename("e", "e")
    # vtkfile_e << e


def show_map_helper(radius, control_points):
    domain_size = 1.
    # coarse_division = 51
    # fine_division = 501
    coarse_division = 51
    fine_division = 51

    x_coo = []
    y_coo = []
    for i in range(coarse_division + 1):
        x_coo.append(np.linspace(0, domain_size, fine_division + 1))
        y_coo.append(np.linspace(i*domain_size/coarse_division, i*domain_size/coarse_division, fine_division + 1))

    for i in range(coarse_division + 1):
        y_coo.append(np.linspace(0, domain_size, fine_division + 1))
        x_coo.append(np.linspace(i*domain_size/coarse_division, i*domain_size/coarse_division, fine_division + 1))

    x_coo = np.array(x_coo)
    y_coo = np.array(y_coo)

    impact_radii = radius*np.ones(len(control_points))

    x_coo_mapped = np.array(x_coo)
    y_coo_mapped = np.array(y_coo)

    for i in range(len(x_coo)):
        for j in range(len(x_coo[0])):
            x_mapped = map_function_normal([x_coo[i][j], y_coo[i][j]], control_points, impact_radii, 'smooth')
            x_coo_mapped[i][j] = x_mapped[0]
            y_coo_mapped[i][j] = x_mapped[1] 

    return x_coo, y_coo, x_coo_mapped, y_coo_mapped


def show_fixed_map():
    radius = 1./4.
    p1 = [3./8., 3./8.]
    p2 = [5./8., 5./8.]
    control_points = np.array([p1, p2])
    x_coo, y_coo, x_coo_mapped, y_coo_mapped = show_map_helper(radius, control_points)

    angles1 = np.linspace(3./4.*np.pi, 7./4.*np.pi, 100)
    x_curve1 = radius * np.cos(angles1) + p1[0]
    y_curve1 = radius * np.sin(angles1) + p1[0]

    angles2 = np.linspace(-np.pi/4., 3./4.*np.pi, 100)
    x_curve2 = radius * np.cos(angles2) + p2[0]
    y_curve2 = radius * np.sin(angles2) + p2[1]

    fig1 = plt.figure(num=0, figsize=(8, 8))
    for i in range(len(x_coo)):
        plt.plot(x_coo[i], y_coo[i], color='black', linewidth=0.5)
    plt.plot(x_curve1, y_curve1, linestyle='--', color='red')
    plt.plot(x_curve2, y_curve2, linestyle='--', color='red')
    plt.plot([x_curve1[0], x_curve2[-1]], [y_curve1[0], y_curve2[-1]], linestyle='--', color='red')
    plt.plot([x_curve1[-1], x_curve2[0]], [y_curve1[-1], y_curve2[0]], linestyle='--', color='red')
    plt.plot(control_points[:, 0], control_points[:, 1], linestyle='-', marker='o', color='red')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    fig1.savefig('data/pdf/{}/omega.pdf'.format(case_name), bbox_inches='tight')

    fig2 = plt.figure(num=1, figsize=(8, 8))
    for i in range(len(x_coo)):
        plt.plot(x_coo_mapped[i], y_coo_mapped[i], color='black', linewidth=0.5)
    plt.plot(x_curve1, y_curve1, linestyle='--', color='red')
    plt.plot(x_curve2, y_curve2, linestyle='--', color='red')
    plt.plot([x_curve1[0], x_curve2[-1]], [y_curve1[0], y_curve2[-1]], linestyle='--', color='red')
    plt.plot([x_curve1[-1], x_curve2[0]], [y_curve1[-1], y_curve2[0]], linestyle='--', color='red')
    plt.plot(control_points[:, 0], control_points[:, 1], linestyle='-', marker='o', color='red')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    fig2.savefig('data/pdf/{}/omega_hat.pdf'.format(case_name), bbox_inches='tight')


def show_adaptive_map():
    radius = 1./4.
    p1 = [6./16., 5./16.]
    p2 = [7./16., 7./16.]
    p3 = [9./16., 9./16.]
    control_points1 = np.array([p1, p2, p3])
    x_coo, y_coo, x_coo_mapped1, y_coo_mapped1 = show_map_helper(radius, control_points1)

    fig1 = plt.figure(num=0, figsize=(8, 8))
    for i in range(len(x_coo)):
        plt.plot(x_coo_mapped1[i], y_coo_mapped1[i], color='black', linewidth=0.5)
    plt.plot(control_points1[:, 0], control_points1[:, 1], linestyle='-', marker='o', color='red')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    fig1.savefig('data/pdf/{}/adaptive1.pdf'.format(case_name), bbox_inches='tight')

    p1 = [6./16., 5./16.]
    p2 = [7./16., 7./16.]
    p3 = [9./16., 9./16.]
    p4 = [11./16., 10./16.]
    control_points2 = np.array([p1, p2, p3, p4])
    x_coo, y_coo, x_coo_mapped2, y_coo_mapped2 = show_map_helper(radius, control_points2)

    fig2 = plt.figure(num=1, figsize=(8, 8))
    for i in range(len(x_coo)):
        plt.plot(x_coo_mapped2[i], y_coo_mapped2[i], color='black', linewidth=0.5)
    plt.plot(control_points2[:, 0], control_points2[:, 1], linestyle='-', marker='o', color='red')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    fig2.savefig('data/pdf/{}/adaptive2.pdf'.format(case_name), bbox_inches='tight')


def plot_ratio():
    x1 = np.linspace(0, 0.5, 101)
    y1 = 0.5*x1
    x2 = np.linspace(0.5, 1, 101)
    y2 = -6*x2**3 + 14*x2**2 -9*x2 + 2
    x3 = np.linspace(1, 2, 101)
    y3 = x3
    x_id = np.linspace(0, 2, 101)
    y_id = x_id
    fig = plt.figure()
    plt.plot(x1, y1, color='black')
    plt.plot(x2, y2, color='black')
    plt.plot(x3, y3, color='black', label=r'$q(\eta)$')
    plt.plot(x_id, y_id, color='black', linestyle='--', label=r'id')
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.axis('equal')
    plt.tick_params(labelsize=14)
    plt.xlabel(r'$\eta$', fontsize=18)
    # plt.ylabel(r'$q(\eta)$', fontsize=14)
    fig.savefig('data/pdf/{}/ratio.pdf'.format(case_name), bbox_inches='tight')


def plot_1d_map_helper(x, y, fig_id, label_x, label_y):
    fig = plt.figure(fig_id)
    plt.plot(x, y, color='black')
    plt.axis('equal')
    plt.tick_params(labelsize=18)
    plt.xlabel(label_x, fontsize=22)
    plt.ylabel(label_y, fontsize=22)
    plt.xlim([-1, 1])
    plt.ylim([-0.1, 1])
    plt.grid(True)
    fig.savefig('data/pdf/{}/1d_map_{}_ratio.pdf'.format(case_name, fig_id), bbox_inches='tight')


# To produce the igure in the section "Introduction"
def plot_1d_map():
    x = np.linspace(-1, 1, 1001)
    y = np.exp(-np.absolute(x)/0.1)
    x_hat = np.where(x > 0, x**0.5, -(-x)**0.5)
    plot_1d_map_helper(x, y, 0, r'$x$', r'$d(x)$')
    plot_1d_map_helper(x_hat, y, 1, r'$\hat{x}$', r'$d(\hat{x})$')


if __name__ == '__main__':
    case_name = 'mfem'
    # mfem()
    # show_fixed_map()
    # show_adaptive_map()
    plot_ratio()
    # plot_1d_map()
    plt.show()