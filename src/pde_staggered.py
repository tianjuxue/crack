import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
  

staggered_tol     = 1e-6 # tolerance for the staggered scheme
staggered_maxiter = 20   # max. iteration for the staggered scheme
newton_Rtol       = 1e-8 # relative tolerance for Newton solver (balance eq.)
newton_Atol       = 1e-8 # absoulte tolerance for Newton solver (balance eq.)
newton_maxiter    = 20   # max. iteration for Newton solver (balance eq.)
snes_Rtol         = 1e-9 # relative tolerance for SNES solver (phase field eq.)
snes_Atol         = 1e-9 # absolute tolerance for SNES solver (phase field eq.)
snes_maxiter      = 30   # max. iteration for SNEs solver (phase field eq.)



fe.parameters["form_compiler"]["quadrature_degree"] = 4
dim = 2


t_i = 0.0    # Initial t [sec]
t_f = 50    # Final t [sec]
dt = 1      # dt [sec]
disp_rate = 1    # Displacement rate [mm/s]

       
G  = 0.19          # Shear modulus [Mpa]
nu = 0.45          # Poisson's ratio
lamda = G * ((2. * nu) / (1. - 2. * nu))
mu = G
kappa = lamda + 2. / 3. * mu
E = 3 * kappa * (1 - 2 * nu)
beta = 2 * nu / (1 - 2 * nu)

sigma_c = 2
psi_cr = sigma_c**2 / (2 * E)

psi_cr = 0.03

Gc_0 = 0.1
l0 = 1.

displacements = np.concatenate((np.linspace(1, 11, 6), np.linspace(12, 26.5, 30), np.linspace(27, 40, 53)))


def H(u_new, H_old):
    I = fe.Identity(dim)
    psi_new = psi(I + fe.grad(u_new))  
    history_max_tmp = fe.conditional(fe.gt(psi_new - psi_cr, 0), psi_new - psi_cr, 0)
    history_max = fe.conditional(fe.gt(history_max_tmp, H_old), history_max_tmp, H_old)
    return history_max


def strain(grad_u):
    return 0.5*(grad_u + grad_u.T)

def linear_elasticity_psi_plus(epsilon):
    return linear_elasticity_psi(epsilon)

def linear_elasticity_psi_minus(epsilon):
    return 0

def linear_elasticity_psi(epsilon):
    return lamda / 2 * fe.tr(epsilon)**2 + mu * fe.inner(epsilon, epsilon)

def cauchy_stress_plus(epsilon):
    epsilon = fe.variable(epsilon)
    energy_plus = linear_elasticity_psi_plus(epsilon)
    sigma_plus = fe.diff(energy_plus, epsilon)
    return sigma_plus

    
def cauchy_stress_minus(epsilon):
    epsilon = fe.variable(epsilon)
    energy_minus = linear_elasticity_psi_minus(epsilon)
    sigma_minus = fe.diff(energy_minus, epsilon)
    return sigma_minus

def cauchy_stress(epsilon):
    epsilon = fe.variable(epsilon)
    energy = cauchy_stress_plus(epsilon)
    sigma = fe.diff(energy, epsilon)
    return sigma



def psi_aux(F):
    J = fe.det(F)
    C = F.T * F
    Jinv = J**(-2 / 3)
    U = 0.5 * kappa * (0.5 * (J**2 - 1) - fe.ln(J))
    Wbar = 0.5 * mu * (Jinv * (fe.tr(C) + 1) - 3)
    return U, Wbar


def psi_plus(F):
    J = fe.det(F)
    U, Wbar = psi_aux(F)
    return fe.conditional(fe.lt(J, 1), Wbar, U + Wbar)


def psi_minus(F):
    J = fe.det(F)
    U, Wbar = psi_aux(F)
    return fe.conditional(fe.lt(J, 1), U, 0)


# def psi(F):
#     J = fe.det(F)
#     U, Wbar = psi_aux(F)
#     return  U + Wbar 


def psi(F):
    J = fe.det(F)
    C = F.T * F
    W = mu / 2 * (fe.tr(C) + 1 - 3) + mu / beta * (J**(-beta) - 1)
    return W



def g_d(d):
    m = 2
    degrad = m * ((1 - d)**3 - (1 - d)**2) + 3 * (1 - d)**2 - 2 * (1 - d)**3
    return degrad 


def g_d_prime(d, degrad_func):
    d = fe.variable(d)
    degrad = degrad_func(d)
    degrad_prime = fe.diff(degrad, d)
    return degrad_prime


def first_PK_stress_plus(F):
    F = fe.variable(F)
    energy_plus = psi_plus(F)
    P_plus = fe.diff(energy_plus, F)
    return P_plus

    
def first_PK_stress_minus(F):
    F = fe.variable(F)
    energy_minus = psi_minus(F)
    P_minus = fe.diff(energy_minus, F)
    return P_minus


def first_PK_stress(F):
    F = fe.variable(F)
    energy = psi(F)
    P = fe.diff(energy, F)
    return P


def phase_field():
    '''
    Try to reproduce https://doi.org/10.1016/j.cma.2014.11.016
    '''
    files = glob.glob('data/pvd/circular_holes/*')
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print('Failed to delete {}, reason: {}' % (f, e))

    length = 60
    height = 30
    radius = 5
    plate = mshr.Rectangle(fe.Point(0, 0), fe.Point(length, height))
    circle1 = mshr.Circle(fe.Point(length/3, height/3), radius)
    circle2 = mshr.Circle(fe.Point(length*2/3, height*2/3), radius)
    material_domain = plate - circle1 - circle2
    mesh = mshr.generate_mesh(material_domain, 50)


    class Left(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], 0)

    class Right(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], length)

    class Corner(fe.SubDomain):
        def inside(self, x, on_boundary):                    
            return fe.near(x[0], 0) and fe.near(x[1], 0)

    U = fe.VectorFunctionSpace(mesh, 'CG', 2)
    W = fe.FunctionSpace(mesh, 'CG', 1) 
    WW = fe.FunctionSpace(mesh, 'DG', 0) 

    left = Left()
    right = Right()
    corner = Corner()

    presLoad = fe.Expression("t", t=0.0, degree=1)
    BC_u_left = fe.DirichletBC(U.sub(0), fe.Constant(0), left)
    BC_u_right = fe.DirichletBC(U.sub(0), presLoad, right)
    BC_u_corner = fe.DirichletBC(U.sub(1), fe.Constant(0.0), corner, method='pointwise')

    BC = [BC_u_left, BC_u_right, BC_u_corner]     
    BC_d = []

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    right.mark(boundaries, 1)
    ds = fe.Measure("ds")(subdomain_data=boundaries)

    I = fe.Identity(dim)
    normal = fe.FacetNormal(mesh)

    eta = fe.TestFunction(U)
    zeta = fe.TestFunction(W)

    del_x = fe.TrialFunction(U)
    del_d = fe.TrialFunction(W)

    x_new = fe.Function(U)
    x_old = fe.Function(U)

    d_new = fe.Function(W)
    d_old = fe.Function(W) 

    H_old = fe.Function(WW)

    G_ut = g_d(d_new) * fe.inner(first_PK_stress(I + fe.grad(x_new)), fe.grad(eta)) * fe.dx
  
    G_d = H_old * zeta * g_d_prime(d_new, g_d) * fe.dx \
        + 2 * psi_cr * (zeta * d_new + l0**2 * fe.inner(fe.grad(zeta), fe.grad(d_new))) * fe.dx  


    J_ut = fe.derivative(G_ut, x_new, del_x)
    J_d = fe.derivative(G_d, d_new, del_d) 

    d_min = fe.interpolate(fe.Constant(fe.DOLFIN_EPS), W) 
    d_max = fe.interpolate(fe.Constant(1.0), W)      

    p_ut = fe.NonlinearVariationalProblem(G_ut, x_new, BC,   J_ut)
    p_d  = fe.NonlinearVariationalProblem(G_d,  d_new, BC_d, J_d)

    # p_d.set_bounds(d_min, d_max)  

    solver_ut = fe.NonlinearVariationalSolver(p_ut)
    solver_d  = fe.NonlinearVariationalSolver(p_d)

    # newton_prm = solver_ut.parameters['newton_solver']
    # newton_prm['relative_tolerance'] = newton_Rtol
    # newton_prm['absolute_tolerance'] = newton_Atol
    # newton_prm['maximum_iterations'] = newton_maxiter
    # newton_prm['error_on_nonconvergence'] = False
    # newton_prm['linear_solver'] = 'mumps'

    # snes_prm = {"nonlinear_solver": "snes",
    #             "snes_solver"     : { "method": "vinewtonssls",
    #                                   "line_search": "basic",
    #                                   "maximum_iterations": snes_maxiter,
    #                                   "relative_tolerance": snes_Rtol,
    #                                   "absolute_tolerance": snes_Atol,
    #                                   "report": True,
    #                                   "error_on_nonconvergence": False,
    #                                 }}
    # solver_d.parameters.update(snes_prm)


    vtkfile_u = fe.File('data/pvd/circular_holes/u.pvd')
    vtkfile_d = fe.File('data/pvd/circular_holes/d.pvd')

    t = t_i
    sigmas = []
    deltaUs = []
    forceForm = (first_PK_stress(I + fe.grad(x_new))[0, 0])*ds(1)

    for disp in displacements:

        t += dt

        print(' ')
        print('=================================================================================')
        print('>> disp boundary condition = {} [mm]'.format(disp))
        print('=================================================================================')

        presLoad.t = disp

        newton_prm = solver_ut.parameters['newton_solver']
        newton_prm['maximum_iterations'] = 1000
        newton_prm['linear_solver'] = 'mumps'   
        newton_prm['absolute_tolerance'] = 1e-4

        if disp > 11 and disp <= 14 :
            newton_prm['relaxation_parameter'] = 0.2
        elif disp > 14 and disp <= 26:
            newton_prm['relaxation_parameter'] = 0.1
        elif disp > 26:
            newton_prm['relaxation_parameter'] = 0.02

        iteration = 0
        err = 1.

        H_old.assign(fe.project(H(x_new, H_old), WW))

        while err > staggered_tol:
            iteration += 1

            # solve phase field equation
            print('[Solving phase field equation...]')
            solver_d.solve()

            # solve momentum balance equations
            print(' ')
            print('[Solving balance equations...]')
            solver_ut.solve()

            # compute error norms
            print(' ')
            print('[Computing residuals...]')

            err_u = fe.errornorm(x_new, x_old, norm_type='l2', mesh=None)
            err_d = fe.errornorm(d_new, d_old, norm_type='l2', mesh=None)
            err = max(err_u, err_d)

            x_old.assign(x_new)
            d_old.assign(d_new)


            print('---------------------------------------------------------------------------------')
            print('>> iteration. {}, error = {:.5}'.format(iteration, err))
            print('---------------------------------------------------------------------------------')


            if err < staggered_tol or iteration >= staggered_maxiter:

                print(
                    '=================================================================================')
                print(' ')

                x_new.rename("Displacement", "label")
                d_new.rename("Phase field", "label")

                vtkfile_u << x_new
                vtkfile_d << d_new
                deltaUs.append(t * disp_rate)
                sigmas.append(fe.assemble(forceForm))

                break

    plt.clf()
    plt.plot(deltaUs, np.array(sigmas)/G)
    plt.savefig("data/png/phase_field/stress-strain-curve.png")
 
    # u = fe.Function(U)
    # file = fe.File('data/pvd/circular_holes/u.pvd')
    # u.rename('u', 'u')
    # file << u



if __name__ == '__main__':
    phase_field()
    plt.show()