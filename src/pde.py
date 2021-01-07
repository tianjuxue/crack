import fenics as fe
import sys
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
import scipy.optimize as opt
import shutil
from functools import partial
from . import arguments
from .constitutive import *
from .mfem import distance_function_segments_ufl, map_function_normal, map_function_ufl, inverse_map_function_normal


# fe.parameters["form_compiler"]["quadrature_degree"] = 4

#TODO: Change x to u if for displacement

class PDE(object):
    def __init__(self, args):
        self.args = args
        self.preparation()
        self.build_mesh()
        self.set_boundaries()
        self.staggered_tol = 1e-5
        self.staggered_maxiter = 1000 
        self.map_flag = False
        self.delta_u_recorded = []
        self.force_recorded = []
        self.update_weak_form = True
        self.display_intermediate_results = False


    def preparation(self):
        # files = glob.glob('data/pvd/{}/*'.format(self.case_name))
        # for f in files:
        #     try:
        #         os.remove(f)
        #     except Exception as e:
        #         print('Failed to delete {}, reason: {}' % (f, e))
        data_path_pvd = 'data/pvd/{}'.format(self.case_name)
        print("\nDelete data folder {}".format(data_path_pvd))
        shutil.rmtree(data_path_pvd, ignore_errors=True)
        # data_path_xdmf = 'data/xdmf/{}'.format(self.case_name)
        # print("\nDelete data folder {}".format(data_path_xdmf))
        # shutil.rmtree(data_path_xdmf, ignore_errors=True)


    def set_boundaries(self):
        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.ds = fe.Measure("ds")(subdomain_data=self.boundaries)   
        self.I = fe.Identity(self.mesh.topology().dim())
        self.normal = fe.FacetNormal(self.mesh)


    def update_history(self):
        return 0

    def update_weak_form_due_to_Model_C_bug(self):
        pass

    def create_custom_xdmf_files(self):
        pass

    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 

        self.WW = fe.FunctionSpace(self.mesh, 'DG', 0) 
        self.EE = fe.TensorFunctionSpace(self.mesh, 'DG', 0) 
        self.MM = fe.VectorFunctionSpace(self.mesh, 'CG', 1)

        self.eta = fe.TestFunction(self.U)
        self.zeta = fe.TestFunction(self.W)
        q = fe.TestFunction(self.WW)

        del_x = fe.TrialFunction(self.U)
        del_d = fe.TrialFunction(self.W)
        p = fe.TrialFunction(self.WW)

        self.x_new = fe.Function(self.U, name="u")
        self.d_new = fe.Function(self.W, name="d")
        self.d_pre = fe.Function(self.W)
        self.x_pre = fe.Function(self.U)

        x_old = fe.Function(self.U)
        d_old = fe.Function(self.W) 

        self.H_old = fe.Function(self.WW)

        self.map_plot = fe.Function(self.MM, name="m")
        e = fe.Function(self.EE, name="e")

        self.create_custom_xdmf_files() 

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
            self.i = i
            self.update_weak_form_due_to_Model_C_bug()

            if self.update_weak_form:
                print("Update weak form...")
                self.build_weak_form_staggered()
                J_u = fe.derivative(self.G_u, self.x_new, del_x)
                J_d = fe.derivative(self.G_d, self.d_new, del_d) 

                self.set_bcs_staggered()
                p_u = fe.NonlinearVariationalProblem(self.G_u, self.x_new, self.BC_u, J_u)
                p_d  = fe.NonlinearVariationalProblem(self.G_d,  self.d_new, self.BC_d, J_d)
                solver_u = fe.NonlinearVariationalSolver(p_u)
                solver_d  = fe.NonlinearVariationalSolver(p_d)
                self.update_weak_form = False

                a = p * q * fe.dx
                L = history(self.H_old, self.update_history(), self.psi_cr) * q * fe.dx

                self.interpolate_map()
                # delta_x = self.x - self.x_hat
                # self.map_plot.assign(fe.project(delta_x, self.MM))

            self.presLoad.t = disp

            newton_prm = solver_u.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100 
            # newton_prm['absolute_tolerance'] = 1e-8
            newton_prm['relaxation_parameter'] = rp

            newton_prm = solver_d.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100 
            # newton_prm['absolute_tolerance'] = 1e-8
            newton_prm['relaxation_parameter'] = rp
           
            vtkfile_e_staggered = fe.File('data/pvd/{}/step{}/e.pvd'.format(self.case_name, i))
            vtkfile_u_staggered = fe.File('data/pvd/{}/step{}/u.pvd'.format(self.case_name, i))
            vtkfile_d_staggered = fe.File('data/pvd/{}/step{}/d.pvd'.format(self.case_name, i))
            iteration = 0
            err = 1.
            while err > self.staggered_tol:
                iteration += 1

                solver_d.solve()

                solver_u.solve()

                if self.solution_scheme == 'explicit':
                    break

                # # Remarks(Tianju): self.x_new.vector() does not behave as expected: producing nan values
                # The following lines of codes cause issues
                # We use an error measure similar in https://doi.org/10.1007/s10704-019-00372-y
                # np_x_new = np.asarray(self.x_new.vector())
                # np_d_new = np.asarray(self.d_new.vector())
                # np_x_old = np.asarray(x_old.vector())
                # np_d_old = np.asarray(d_old.vector())
                # err_x = np.linalg.norm(np_x_new - np_x_old) / np.sqrt(len(np_x_new))
                # err_d = np.linalg.norm(np_d_new - np_d_old) / np.sqrt(len(np_d_new))
                # err = max(err_x, err_d)

                # # Remarks(Tianju): dolfin (2019.1.0) errornorm function has severe bugs not behave as expected
                # The bug seems to be fixed in later versions
                # The following sometimes produces nonzero results in dolfin (2019.1.0)
                # print(fe.errornorm(self.d_new, self.d_new, norm_type='l2'))

                err_x = fe.errornorm(self.x_new, x_old, norm_type='l2')
                err_d = fe.errornorm(self.d_new, d_old, norm_type='l2')
                err = max(err_x, err_d) 

                x_old.assign(self.x_new)
                d_old.assign(self.d_new)
                e.assign(fe.project(strain(self.mfem_grad(self.x_new)), self.EE))

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

            fe.solve(a == L, self.H_old, [])      

            # self.d_pre.assign(self.d_new)
            # self.H_old.assign(fe.project(history(self.H_old, self.update_history(), self.psi_cr), self.WW))

            if self.map_flag:
                self.update_map()

            file_results.write(e, i)
            file_results.write(self.x_new, i)
            file_results.write(self.d_new, i)
            file_results.write(self.map_plot, i)

            vtkfile_e << e
            vtkfile_u << self.x_new
            vtkfile_d << self.d_new

            self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)
            self.sigma = cauchy_stress(strain(self.mfem_grad(self.x_new)), self.psi)
            if self.case_name == 'pure_shear':
                force_upper = float(fe.assemble(self.sigma[0, 1]*self.ds(1)))
            else:
                force_upper = float(fe.assemble(self.sigma[1, 1]*self.ds(1)))
            print("Force is {}".format(force_upper))
            self.delta_u_recorded.append(disp)
            self.force_recorded.append(force_upper)

            # if force_upper < 0.5 and i > 10:
            #     break

            if i %10 == 0:
                self.show_force_displacement()

            self.save_data_in_loop()

        plt.ioff()
        plt.show()
 

    def show_force_displacement(self):
        fig = plt.figure(0)
        plt.ion()
        plt.plot(self.delta_u_recorded, self.force_recorded, linestyle='--', marker='o', color='red')
        plt.tick_params(labelsize=14)
        plt.xlabel("Vertical displacement of top side", fontsize=14)
        plt.ylabel("Force on top side", fontsize=14)
        plt.grid(True)
        fig.savefig('data/pdf/{}/force_load.pdf'.format(self.case_name), bbox_inches='tight')
        if self.display_intermediate_results:
            plt.show()
            plt.pause(0.001)
            

class MappedPDE(PDE):
    def __init__(self, args):
        super(MappedPDE, self).__init__(args)
        self.d_integrals = []
        self.finish_flag = False
        self.map_flag = True
        self.boundary_info = None
        self.rho_default = 15.
        self.d_integral_interval = 1.5*self.rho_default


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
        tol = 1e-10
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
            impact_radius_middle_point = self.compute_impact_radius_middle_point(self.control_points[-1], self.middle_vector(v1, v2))
            impact_radius_tip_point = self.compute_impact_radius_tip_point(new_tip_point, new_tip_point - self.control_points[-1])
            if impact_radius_middle_point < self.rho_default or impact_radius_tip_point < self.rho_default:
                self.impact_radii = np.append(self.impact_radii, self.rho_default)

                self.compute_boundary_info(self.control_points[-1], -v1)
                self.finish_flag = True
                self.update_weak_form = True

                print("Setting finish_flag True because it is near the boundary")
            else:
                self.impact_radii = np.append(self.impact_radii, impact_radius_middle_point)
                self.impact_radii = np.append(self.impact_radii, impact_radius_tip_point)
                self.control_points = np.concatenate((self.control_points, new_tip_point.reshape(1, -1)), axis=0)

        assert len(self.control_points) == len(self.impact_radii)


    def set_to_boundary(self, points):
        for P in points:
            if np.absolute(P[0] - 0) < 1e-3:
                P[0] = 0
            if np.absolute(P[0] - self.length) < 1e-3:
                P[0] = self.length            
            if np.absolute(P[1] - 0) < 1e-3:
                P[1] = 0
            if np.absolute(P[1] - self.height) < 1e-3:
                P[1] = self.height 


    def three_points_same_line(self, points):
        assert len(points) == 3
        cross_product = np.cross(points[1] - points[0], points[2] - points[1])
        print("The three points tested are {}".format(points))
        return True if cross_product < 1e-10 else False


    def compute_boundary_info(self, P, direct_vec):
        direct_vec /= np.linalg.norm(direct_vec)
        rotated_vec = np.array(direct_vec)
        rotated_vec[0] = -direct_vec[1]
        rotated_vec[1] = direct_vec[0]

        start_point = np.array(P)
        end_point = start_point + direct_vec*1e3
        mid_point = self.binary_search(start_point, end_point)

        start_point1 = np.array(P + rotated_vec*self.rho_default)
        end_point1 = start_point1 + direct_vec*1e3
        mid_point1 = self.binary_search(start_point1, end_point1)

        start_point2 = np.array(P - rotated_vec*self.rho_default)
        end_point2 = start_point2 + direct_vec*1e3
        mid_point2 = self.binary_search(start_point2, end_point2)


        points = [mid_point, mid_point1, mid_point2]
        print(points)
        directions = [direct_vec, rotated_vec]
        self.set_to_boundary(points)
        print(points)
        assert self.three_points_same_line(points), "Assumption violated: The three boundary points are not on the same line!"

        self.boundary_info = [points, directions, self.rho_default]


    def identify_crack_tip(self):

        print('\n')
        print('=================================================================================')
        print('>> Identifying crack tip')
        print('=================================================================================')

        def obj(x):
            p = fe.Constant(x)
            x_coo = fe.SpatialCoordinate(self.mesh)
            control_points = list(self.control_points)
            control_points.append(p)
            pseudo_radii = np.zeros(len(control_points))
            distance_field, _ = distance_function_segments_ufl(x_coo, control_points, pseudo_radii)
            d_artificial = fe.exp(-distance_field/self.l0)

            d_clipped = fe.conditional(fe.gt(self.d_new, 0.5), self.d_new, 0.)

            L_tape = fe.assemble((d_clipped - d_artificial)**2 * fe.det(self.grad_gamma) * fe.dx)
            # L_tape = fe.assemble((self.d_new - d_artificial)**2 * fe.det(self.grad_gamma) * fe.dx)

            L = float(L_tape)
            return L

        if len(self.control_points) > 1:
            x_initial = self.control_points[-1] + (self.control_points[-1] - self.control_points[-2])
        elif len(self.control_points) == 1:
            if self.case_name == 'half_crack_square':
                x_initial = np.array([self.length/2 + 1, self.height/2])
            elif self.case_name == 'L_shape':
                x_initial = np.array([self.length/2 - 1, self.height/2 + 1])
            else:
                raise NotImplementedError("To be implemented!")
        else:
            raise NotImplementedError("To be implemented!")

        options = {'eps': 1e-15, 'maxiter': 1000, 'disp': True}  # CG > BFGS > Newton-CG
        res = opt.minimize(fun=obj,
                           x0=x_initial,
                           method='CG',
                           callback=None,
                           options=options)

        new_tip_point = map_function_normal(res.x, self.control_points, self.impact_radii, self.map_type, self.boundary_info)

        print("Optimized x is {}".format(res.x))
        print("New tip point is {}".format(new_tip_point))
        print('=================================================================================')

        return new_tip_point


    def interpolate_map(self):
        print("Interploating map...")
        control_points = self.control_points
        impact_radii  = self.impact_radii
        boundary_info = self.boundary_info
        map_type = self.map_type

        class InterpolateExpression(fe.UserExpression):
            def eval(self, values, x_hat):
                x = map_function_normal(x_hat, control_points, impact_radii, map_type, boundary_info)
                # x = np.array(map_function_ufl(x_hat, control_points, impact_radii, map_type, boundary_info))
                values[0] = (x - x_hat)[0]
                values[1] = (x - x_hat)[1]

            def value_shape(self):
                return (2,)

        map_exp = InterpolateExpression()
        self.map_plot.assign(fe.interpolate(map_exp, self.MM))
        print("Finish interploating map")


    def interpolate_H(self):
        print("Interploating H_old...")
        control_points_new_map = self.control_points
        impact_radii_new_map = self.impact_radii
        map_type = self.map_type

        inside_domain = self.inside_domain
        H_old = self.H_old
        boundary_info = self.boundary_info

        if boundary_info is None:
            control_points_old_map = self.control_points[:-1]
            impact_radii_old_map = self.impact_radii_old
        else:
            control_points_old_map = self.control_points
            impact_radii_old_map = self.impact_radii

        # print("boundary_info {}".format(boundary_info))
        # print("control_points_new_map {}".format(control_points_new_map))
        # print("control_points_old_map {}".format(control_points_old_map))
        # print("impact_radii_new_map {}".format(impact_radii_new_map))
        # print("impact_radii_old_map {}".format(impact_radii_old_map))

        class InterpolateExpression(fe.UserExpression):
            def eval(self, values, x_hat_new):
                x = map_function_normal(x_hat_new, control_points_new_map, impact_radii_new_map, map_type, boundary_info)
                x_hat_old = inverse_map_function_normal(x, control_points_old_map, impact_radii_old_map, map_type)

                point = fe.Point(x_hat_old)

                if not inside_domain(point):
                    print("x_hat_new is ({}, {})".format(float(x_hat_new[0]), float(x_hat_new[1])))
                    print("x is ({}, {})".format(float(x[0]), float(x[1])))
                    print("x_hat_old is ({}, {})".format(float(x_hat_old[0]), float(x_hat_old[1])))
            
                values[0] = H_old(point)

            def value_shape(self):
                return ()

        H_exp = InterpolateExpression()
        # self.H_old.assign(fe.project(H_exp, self.WW)) # two slow
        self.H_old.assign(fe.interpolate(H_exp, self.WW))

        print("Finish interploating H_old")


    def update_map(self):
        d_clipped = fe.conditional(fe.gt(self.d_new, 0.5), self.d_new, 0.)        
        d_int_full = fe.assemble(self.d_new * fe.det(self.grad_gamma) * fe.dx)
        d_int = fe.assemble(d_clipped * fe.det(self.grad_gamma) * fe.dx)

        print("d_int_clipped {}".format(float(d_int)))
        print("d_int_full {}".format(float(d_int_full)))

        d_integral_interval_initial = 1

        update_flag = False

        while len(self.d_integrals) < len(self.control_points):
            self.d_integrals.append(d_int)

        if len(self.d_integrals) == 0:
            if d_int > d_integral_interval_initial:
                update_flag = True
        else:
            if d_int - self.d_integrals[-1] > self.d_integral_interval:
                update_flag = True

        if update_flag and not self.finish_flag:
            print('\n')
            print('=================================================================================')
            print('>> Updating map...')
            print('=================================================================================')

            new_tip_point = self.identify_crack_tip()
            if self.inside_domain(new_tip_point):
                if len(self.control_points) > 1:
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
            self.update_weak_form = True

        else:
            print("Do not modify map")

        print("d_integrals {}".format(self.d_integrals))
