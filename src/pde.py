import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os


fe.parameters["form_compiler"]["quadrature_degree"] = 4


# Material parameters 1 (micropolar elasticity)
G  = 12.5e3    # Shear modulus [MPa]
nu = 0.2       # Poisson's ratio
 

# Material parameters 2 (phase field fracture)
Gc     = 0.1     # Critical energy release rate [N/mm]
lc     = 0.75    # Length scale [mm]
psi_cr = 0.001   # Threshold strain energy per unit volume [MJ/m3]
p      = 10.0    # Shape parameter

# Solver parameters
t_i       = 0.0     # Initial t [sec]
t_f       = 0.05     # Final t [sec]
dt        = 0.001  # dt [sec]
disp_rate = 1     # Displacement rate [mm/s]

staggered_tol     = 1e-6 # tolerance for the staggered scheme
staggered_maxiter = 10   # max. iteration for the staggered scheme
newton_Rtol       = 1e-8 # relative tolerance for Newton solver (balance eq.)
newton_Atol       = 1e-8 # absoulte tolerance for Newton solver (balance eq.)
newton_maxiter    = 20   # max. iteration for Newton solver (balance eq.)
snes_Rtol         = 1e-9 # relative tolerance for SNES solver (phase field eq.)
snes_Atol         = 1e-9 # absolute tolerance for SNES solver (phase field eq.)
snes_maxiter      = 30   # max. iteration for SNEs solver (phase field eq.)

lamda = G*((2.*nu)/(1.-2.*nu))
mu = G
m = 3.*Gc/(8.*lc*psi_cr)


def DeformationGradient(u):
    I = fe.Identity(u.geometric_dimension())
    return fe.variable(I + fe.grad(u))


def epsilon(u):
  strain = fe.as_tensor([[ u[0].dx(0), u[1].dx(0)],
                      [ u[0].dx(1), u[1].dx(1)]])  
  return strain
  

def epsilon_sym(u):
    strain_sym = fe.as_tensor([[u[0].dx(0),  (1./2.)*(u[0].dx(1) + u[1].dx(0))],
                            [(1./2.)*(u[0].dx(1) + u[1].dx(0)), u[1].dx(1)]])
    return strain_sym


def sigma(u):
    eps_sym = epsilon_sym(u)
    stress_B = lamda*fe.tr(eps_sym)*fe.Identity(2) + (2.*mu)*eps_sym
    return stress_B


def psi(u):
    eps_sym = epsilon_sym(u)
    eps1 = (1./2.)*fe.tr(eps_sym) + fe.sqrt((1./4.)*(fe.tr(eps_sym)**2) - fe.det(eps_sym))
    eps2 = (1./2.)*fe.tr(eps_sym) - fe.sqrt((1./4.)*(fe.tr(eps_sym)**2) - fe.det(eps_sym))
    tr_eps_plus = (1./2.)*(fe.tr(eps_sym) + abs(fe.tr(eps_sym)))
    eps_plus_doubledot_eps_plus = ((1./2.)*(eps1 + abs(eps1)))**2 + ((1./2.)*(eps2 + abs(eps2)))**2
    energy = (1./2.)*lamda*(tr_eps_plus**2) + (mu)*eps_plus_doubledot_eps_plus
    return energy


def H(u_old, u_new, H_old):
    psi_i_new = psi(u_new) - psi_cr
    psi_i_old = psi(u_old) - psi_cr
    psi_new = psi_cr + (1./2.) * (psi_i_new + abs(psi_i_new))
    psi_old = psi_cr + (1./2.) * (psi_i_old + abs(psi_i_old))
    return fe.conditional(fe.lt(psi_old, psi_new), psi_new, H_old)


def g_d(d):
    numerator = (1. - d)**2
    denominator = (1. - d)**2 + m * d * (1. + p * d)
    g_d_val = numerator / denominator
    return g_d_val


def g_d_prime(d):
    numerator = (d - 1.) * (d * (2. * p + 1.) + 1.) * m
    denominator = ((d**2) * (m * p + 1.) + d * (m - 2.) + 1.)**2
    g_d_prime_val = numerator/denominator
    return g_d_prime_val


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
    mesh = mshr.generate_mesh(material_domain, 100)

    class Left(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], 0)

    class Right(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], length)

    U = fe.VectorFunctionSpace(mesh, 'CG', 2)
    W = fe.FunctionSpace(mesh, 'CG', 1) 
    WW = fe.FunctionSpace(mesh, 'DG', 0) 

    left = Left()
    right = Right()
    presLoad = fe.Expression("t", t=0.0, degree=1)
    BC_u_left = fe.DirichletBC(U, fe.Constant((0.0, 0.0)), left)
    BC_u1_right = fe.DirichletBC(U.sub(0), presLoad, right)
    BC_u2_right = fe.DirichletBC(U.sub(1), 0.0, Right())
    BC = [BC_u_left, BC_u1_right, BC_u2_right]     
    BC_d = []

    normal = fe.FacetNormal(mesh)

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    right.mark(boundaries, 1)
    ds = fe.Measure("ds")(subdomain_data=boundaries)

    eta = fe.TestFunction(U)
    zeta = fe.TestFunction(W)

    del_x = fe.TrialFunction(U)
    del_d = fe.TrialFunction(W)

    x_new = fe.Function(U)
    x_old = fe.Function(U)

    d_new = fe.Function(W)
    d_old = fe.Function(W) 

    H_old = fe.Function(W)

    G_ut = g_d(d_new) * fe.inner(epsilon(eta), sigma(x_new)) * fe.dx
    J_ut = fe.derivative(G_ut, x_new, del_x)

    # Weak form: phase-field equation
    G_d = H(x_old, x_new, H_old)*fe.inner(zeta, g_d_prime(d_new)) * fe.dx \
        + (3.*Gc/(8.*lc)) * (zeta + (2.*lc**2)*fe.inner(fe.grad(zeta), fe.grad(d_new))) * fe.dx  

    J_d = fe.derivative(G_d, d_new, del_d) 

    d_min = fe.interpolate(fe.Constant(fe.DOLFIN_EPS), W) 
    d_max = fe.interpolate(fe.Constant(1.0), W)      

    # Problem definition
    p_ut = fe.NonlinearVariationalProblem(G_ut, x_new, BC,   J_ut)
    p_d  = fe.NonlinearVariationalProblem(G_d,  d_new, BC_d, J_d)
    p_d.set_bounds(d_min, d_max) # set bounds for the phase field

    # Construct solvers
    solver_ut = fe.NonlinearVariationalSolver(p_ut)
    solver_d  = fe.NonlinearVariationalSolver(p_d)

    # Set nonlinear solver parameters
    newton_prm = solver_ut.parameters['newton_solver']
    newton_prm['relative_tolerance'] = newton_Rtol
    newton_prm['absolute_tolerance'] = newton_Atol
    newton_prm['maximum_iterations'] = newton_maxiter
    newton_prm['error_on_nonconvergence'] = False
    newton_prm['linear_solver'] = 'mumps'


    snes_prm = {"nonlinear_solver": "snes",
                "snes_solver"     : { "method": "vinewtonssls",
                                      "line_search": "basic",
                                      "maximum_iterations": snes_maxiter,
                                      "relative_tolerance": snes_Rtol,
                                      "absolute_tolerance": snes_Atol,
                                      "report": True,
                                      "error_on_nonconvergence": False,
                                    }}
    solver_d.parameters.update(snes_prm)


    vtkfile_u = fe.File('data/pvd/circular_holes/u.pvd')
    vtkfile_d = fe.File('data/pvd/circular_holes/d.pvd')

    t = t_i
    sigmas = []
    deltaUs = []
    forceForm = sigma(x_new)[1, 1]*ds(1)

    while t <= t_f:

        t += dt

        print(' ')
        print('=================================================================================')
        print('>> t =', t, '[sec]')
        print('=================================================================================')

        presLoad.t = t*disp_rate

        iteration = 0
        err = 1

        while err > staggered_tol:
            iteration += 1

            print('---------------------------------------------------------------------------------')
            print('>> iteration. %d, error = %.5g' % (iteration, err))
            print('---------------------------------------------------------------------------------')

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
            H_old.assign(fe.project(fe.conditional(fe.lt(H_old, psi_cr + (1./2.) * (psi(x_new)-psi_cr + abs(psi(x_new)-psi_cr))),
                    psi_cr + (1./2.)*(psi(x_new)-psi_cr + abs(psi(x_new)-psi_cr)),
                    H_old
                ), WW))

            if err < staggered_tol or iteration >= staggered_maxiter:

                print(
                    '=================================================================================')
                print(' ')

                x_new.rename("Displacement", "label")
                d_new.rename("Phase field", "label")

                vtkfile_u << x_new
                vtkfile_d << d_new
                deltaUs.append(t*disp_rate)
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