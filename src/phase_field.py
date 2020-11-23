# ---------------------------------------------------------------- 
# FEniCS implementation: Micropolar phase field fracture   
# Written by: Hyoung Suk Suh (h.suh@columbia.edu)     
# ----------------------------------------------------------------       
import matplotlib
import matplotlib.pyplot as plt
from dolfin import *
import sys
import time
import numpy as np
matplotlib.use('Agg')

parameters["form_compiler"]["quadrature_degree"] = 4

tic = time.time()



# ---------------------------------------------------------------- 
# Input parameters
# ----------------------------------------------------------------
# Mesh and result file names
file_name = 'double_notch'  # Input/output directory name
degradation = 'B'           # Energy parts to be degraded: B, C, R, B+R, B+C, B+C+R

# Material parameters 1 (micropolar elasticity)
G  = 12.5e3    # Shear modulus [MPa]
nu = 0.2       # Poisson's ratio
l  = 30.0      # Characteristic length (bending) [mm]
N  = 0.5       # Coupling parameter 

# Material parameters 2 (phase field fracture)
Gc     = 0.1     # Critical energy release rate [N/mm]
lc     = 0.75    # Length scale [mm]
psi_cr = 0.001   # Threshold strain energy per unit volume [MJ/m3]
p      = 10.0    # Shape parameter

# Solver parameters
t_i       = 0.0     # Initial t [sec]
t_f       = 0.05     # Final t [sec]
dt        = 0.0005  # dt [sec]
disp_rate = 4.0     # Displacement rate [mm/s]

staggered_tol     = 1e-6 # tolerance for the staggered scheme
staggered_maxiter = 10   # max. iteration for the staggered scheme
newton_Rtol       = 1e-8 # relative tolerance for Newton solver (balance eq.)
newton_Atol       = 1e-8 # absoulte tolerance for Newton solver (balance eq.)
newton_maxiter    = 20   # max. iteration for Newton solver (balance eq.)
snes_Rtol         = 1e-9 # relative tolerance for SNES solver (phase field eq.)
snes_Atol         = 1e-9 # absolute tolerance for SNES solver (phase field eq.)
snes_maxiter      = 30   # max. iteration for SNEs solver (phase field eq.)


# ---------------------------------------------------------------- 
# Read mesh
# ----------------------------------------------------------------
mesh = Mesh('data/xml/double_notch.xml')

dim = mesh.geometry().dim()
mesh_coord = mesh.coordinates()
mesh_xmin  = min(mesh_coord[:,0])
mesh_xmax  = max(mesh_coord[:,0])
mesh_ymin  = min(mesh_coord[:,1])
mesh_ymax  = max(mesh_coord[:,1])


# ---------------------------------------------------------------- 
# Define function spaces
# ----------------------------------------------------------------
u_elem     = VectorElement('CG', mesh.ufl_cell(), 2) # displacement

U = FunctionSpace(mesh, u_elem)

# function spaces for the phase field, history variable
W  = FunctionSpace(mesh, 'CG', 1) # phase field
WW = FunctionSpace(mesh, 'DG', 0) # history variable



# ---------------------------------------------------------------- 
# Define boundary conditions
# ----------------------------------------------------------------
top    = CompiledSubDomain("near(x[1], mesh_ymax) && on_boundary", mesh_ymax = mesh_ymax)
bottom = CompiledSubDomain("near(x[1], mesh_ymin) && on_boundary", mesh_ymin = mesh_ymin)
left   = CompiledSubDomain("near(x[0], mesh_xmin) && on_boundary", mesh_xmin = mesh_xmin)
right  = CompiledSubDomain("near(x[0], mesh_xmax) && on_boundary", mesh_xmax = mesh_xmax)

# constrained displacement boundary
BC_bottom     = DirichletBC(U, Constant((0.0,0.0)), bottom)

# prescribed displacement boundary
presLoad    = Expression("t", t = 0.0, degree=1)
BC_top_pres1 = DirichletBC(U.sub(1), presLoad, top)
BC_top_pres2 = DirichletBC(U.sub(0), 0.0, top)

# displacement & micro-rotation boundary condition
BC = [BC_bottom,
      BC_top_pres1,  BC_top_pres2]     
  
# phase-field boundary condition   
BC_d = []
  
# mark boundaries
boundaries = MeshFunction('size_t', mesh, dim - 1)
boundaries.set_all(0)

top.mark(boundaries, 1)
  
ds = Measure("ds")(subdomain_data=boundaries)
n  = FacetNormal(mesh)



# ---------------------------------------------------------------- 
# Define variables
# ----------------------------------------------------------------
# Elastic material parameters -- conversion
lamda = G*((2.*nu)/(1.-2.*nu))
mu    = G
gamma = 4.*G*l**2

# Degradation function parameter
m = 3.*Gc/(8.*lc*psi_cr)


# Micropolar strain & micro-curvature -------------
def epsilon(u):
  
  strain = as_tensor([[ u[0].dx(0), u[1].dx(0)],
                      [ u[0].dx(1), u[1].dx(1)]])  

  return strain
  

def epsilon_sym(u):

    strain_sym = as_tensor([[u[0].dx(0),  (1./2.)*(u[0].dx(1) + u[1].dx(0))],
                            [(1./2.)*(u[0].dx(1) + u[1].dx(0)), u[1].dx(1)]])

    return strain_sym


# -------------------------------------------------


# Force stress & couple stress --------------------
def sigma(u):

    eps_sym = epsilon_sym(u)

    stress_B = lamda*tr(eps_sym)*Identity(2) + (2.*mu)*eps_sym

    return stress_B


# -------------------------------------------------


# Strain energy densities -------------------------
def psi(u):

    eps_sym = epsilon_sym(u)

    eps1 = (1./2.)*tr(eps_sym) + sqrt((1./4.)*(tr(eps_sym)**2) - det(eps_sym))
    eps2 = (1./2.)*tr(eps_sym) - sqrt((1./4.)*(tr(eps_sym)**2) - det(eps_sym))

    tr_eps_plus = (1./2.)*(tr(eps_sym) + abs(tr(eps_sym)))
    eps_plus_doubledot_eps_plus = ((1./2.)*(eps1 + abs(eps1)))**2 + ((1./2.)*(eps2 + abs(eps2)))**2

    energy = (1./2.)*lamda*(tr_eps_plus**2) + (mu)*eps_plus_doubledot_eps_plus

    return energy


# -------------------------------------------------
  

# Driving force -----------------------------------
def H(u_old, u_new, H_old):

    psi_i_new = psi(u_new) - psi_cr
    psi_i_old = psi(u_old) - psi_cr

    psi_new = psi_cr + (1./2.) * (psi_i_new + abs(psi_i_new))
    psi_old = psi_cr + (1./2.) * (psi_i_old + abs(psi_i_old))

    return conditional(lt(psi_old, psi_new), psi_new, H_old)

# -------------------------------------------------


# Degradation function & its derivative -----------
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

# -------------------------------------------------




# ---------------------------------------------------------------- 
# Define variational form
# ----------------------------------------------------------------
# Define test & trial spaces 
eta   = TestFunction(U)
zeta  = TestFunction(W)

del_x = TrialFunction(U)
del_d = TrialFunction(W)

x_new = Function(U)
x_old = Function(U)

d_new = Function(W)
d_old = Function(W) 

H_old = Function(W)

# Weak form: balance equations
G_ut = g_d(d_new) * inner(epsilon(eta), sigma(x_new)) * dx

J_ut = derivative(G_ut, x_new, del_x) # jacobian

# Weak form: phase-field equation
G_d = H(x_old, x_new, H_old)*inner(zeta, g_d_prime(d_new)) * dx \
    + (3.*Gc/(8.*lc)) * (zeta + (2.*lc**2)*inner(grad(zeta), grad(d_new))) * dx  

J_d = derivative(G_d, d_new, del_d) # jacobian

# Constraints for the phase field
d_min = interpolate(Constant(DOLFIN_EPS), W) # lower bound
d_max = interpolate(Constant(1.0), W)        # upper bound

# Problem definition
p_ut = NonlinearVariationalProblem(G_ut, x_new, BC,   J_ut)
p_d  = NonlinearVariationalProblem(G_d,  d_new, BC_d, J_d)
p_d.set_bounds(d_min, d_max) # set bounds for the phase field

# Construct solvers
solver_ut = NonlinearVariationalSolver(p_ut)
solver_d  = NonlinearVariationalSolver(p_d)

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





# ---------------------------------------------------------------- 
# Solve system & output results
# ----------------------------------------------------------------
vtkfile_u     = File('data/pvd/double_notch/u.pvd')
vtkfile_d     = File('data/pvd/double_notch/d.pvd')

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

    iter = 0
    err = 1

    while err > staggered_tol:
        iter += 1

        print('---------------------------------------------------------------------------------')
        print('>> iter. %d, error = %.5g' % (iter, err))
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

        err_u = errornorm(x_new, x_old, norm_type='l2', mesh=None)
        err_d = errornorm(d_new, d_old, norm_type='l2', mesh=None)
        err = max(err_u, err_d)

        x_old.assign(x_new)
        d_old.assign(d_new)
        H_old.assign(project(conditional(lt(H_old, psi_cr + (1./2.) * (psi(x_new)-psi_cr + abs(psi(x_new)-psi_cr))),
                psi_cr + (1./2.)*(psi(x_new)-psi_cr + abs(psi(x_new)-psi_cr)),
                H_old
            ), WW))

        if err < staggered_tol or iter >= staggered_maxiter:

            print(
                '=================================================================================')
            print(' ')

            x_new.rename("Displacement", "label")
            d_new.rename("Phase field", "label")

            vtkfile_u << x_new
            vtkfile_d << d_new
            deltaUs.append(t*disp_rate)
            sigmas.append(assemble(forceForm))

            break

toc = time.time() - tic
plt.clf()
plt.plot(deltaUs, np.array(sigmas)/G)
plt.savefig("stress-strain-curve.png")
print('Elapsed CPU time: ', toc, '[sec]')


