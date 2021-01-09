import fenics as fe
import numpy as np
import ufl

# ---------------------------------------------------------------- 
# History functions
def history(H_old, psi_new, psi_cr):
    history_max_tmp = fe.conditional(fe.gt(psi_new - psi_cr, 0), psi_new - psi_cr, 0)
    history_max = fe.conditional(fe.gt(history_max_tmp, H_old), history_max_tmp, H_old)
    return history_max


# ---------------------------------------------------------------- 
# Degradation functions
def g_d(d):
    degrad = (1 - d)**2 + 1e-10;
    return degrad 


def g_d_prime(d, degrad_func):
    d = fe.variable(d)
    degrad = degrad_func(d)
    degrad_prime = fe.diff(degrad, d)
    return degrad_prime


# ---------------------------------------------------------------- 
# Linear elasticity
def strain(grad_u):
    return 0.5*(grad_u + grad_u.T)


def psi_linear_elasticity(epsilon, lamda, mu):
    return lamda / 2 * fe.tr(epsilon)**2 + mu * fe.inner(epsilon, epsilon)


# ----------------------------------------------------------------
# Model A
def psi_plus_linear_elasticity_model_A(epsilon, lamda, mu):
    return psi_linear_elasticity(epsilon, lamda, mu)


def psi_minus_linear_elasticity_model_A(epsilon, lamda, mu):
    return 0.


# ----------------------------------------------------------------
# Model B: Amor paper https://doi.org/10.1016/j.jmps.2009.04.011
# TODO: Check if bulk_mod is correct under plane strain assumption
def psi_plus_linear_elasticity_model_B(epsilon, lamda, mu):
    bulk_mod = lamda + 2. / 3. * mu
    tr_epsilon_plus = ufl.Max(fe.tr(epsilon), 0)
    return bulk_mod / 2. * tr_epsilon_plus**2 + mu * fe.inner(fe.dev(epsilon), fe.dev(epsilon))


def psi_minus_linear_elasticity_model_B(epsilon, lamda, mu):
    bulk_mod = lamda + 2. / 3. * mu
    tr_epsilon_minus = ufl.Min(fe.tr(epsilon), 0)
    return bulk_mod / 2. * tr_epsilon_minus**2


# ----------------------------------------------------------------
# Model C: Miehe paper https://doi.org/10.1002/nme.2861
# Eigenvalue decomposition for 2x2 matrix
# See https://yutsumura.com/express-the-eigenvalues-of-a-2-by-2-matrix-in-terms-of-the-trace-and-determinant/

# Remarks(Tianju): The ufl functions Max and Min do not seem to behave as expected
# For example, the following line of code works
# tr_epsilon_plus = (fe.tr(epsilon) + np.absolute(fe.tr(epsilon))) / 2
# However, the following line of code does not work (Newton solver never converges)
# tr_epsilon_plus = ufl.Max(fe.tr(epsilon), 0)

# Remarks(Tianju): If Newton solver fails to converge, consider using a non-zero initial guess for the displacement field
# For example, use Model A to solve for one step and then switch back to Model C
# The reason for the failure is not clear.
# It may be because of the singular nature of Model C that causes trouble for UFL to take derivatives at the kink.
def psi_plus_linear_elasticity_model_C(epsilon, lamda, mu):
    sqrt_delta = fe.conditional(fe.gt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon), 0), fe.sqrt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon)), 0)
    eigen_value_1 = (fe.tr(epsilon) + sqrt_delta) / 2
    eigen_value_2 = (fe.tr(epsilon) - sqrt_delta) / 2

    # tr_epsilon_plus = (fe.tr(epsilon) + np.absolute(fe.tr(epsilon))) / 2
    # eigen_value_1_plus = (eigen_value_1 + np.absolute(eigen_value_1)) / 2
    # eigen_value_2_plus = (eigen_value_2 + np.absolute(eigen_value_2)) / 2


    tr_epsilon_plus = fe.conditional(fe.gt(fe.tr(epsilon), 0.), fe.tr(epsilon), 0.)
    eigen_value_1_plus = fe.conditional(fe.gt(eigen_value_1, 0.), eigen_value_1, 0.)
    eigen_value_2_plus = fe.conditional(fe.gt(eigen_value_2, 0.), eigen_value_2, 0.)


    return lamda / 2 * tr_epsilon_plus**2 + mu * (eigen_value_1_plus**2 + eigen_value_2_plus**2)


def psi_minus_linear_elasticity_model_C(epsilon, lamda, mu):
    sqrt_delta = fe.conditional(fe.gt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon), 0), fe.sqrt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon)), 0)
    eigen_value_1 = (fe.tr(epsilon) + sqrt_delta) / 2
    eigen_value_2 = (fe.tr(epsilon) - sqrt_delta) / 2

    # tr_epsilon_minus = (fe.tr(epsilon) - np.absolute(fe.tr(epsilon))) / 2
    # eigen_value_1_minus = (eigen_value_1 - np.absolute(eigen_value_1)) / 2
    # eigen_value_2_minus = (eigen_value_2 - np.absolute(eigen_value_2)) / 2

    tr_epsilon_minus = fe.conditional(fe.lt(fe.tr(epsilon), 0.), fe.tr(epsilon), 0.)
    eigen_value_1_minus = fe.conditional(fe.lt(eigen_value_1, 0.), eigen_value_1, 0.)
    eigen_value_2_minus = fe.conditional(fe.lt(eigen_value_2, 0.), eigen_value_2, 0.)


    return lamda / 2 * tr_epsilon_minus**2 + mu * (eigen_value_1_minus**2 + eigen_value_2_minus**2)


# TODO(Tianju): Collapse the three functions into one
# ---------------------------------------------------------------- 
# Cauchy stress
def cauchy_stress_plus(epsilon, psi_plus):
    epsilon = fe.variable(epsilon)
    energy_plus = psi_plus(epsilon)
    sigma_plus = fe.diff(energy_plus, epsilon)
    return sigma_plus

    
def cauchy_stress_minus(epsilon, psi_minus):
    epsilon = fe.variable(epsilon)
    energy_minus = psi_minus(epsilon)
    sigma_minus = fe.diff(energy_minus, epsilon)
    return sigma_minus


def cauchy_stress(epsilon, psi):
    epsilon = fe.variable(epsilon)
    energy = psi(epsilon)
    sigma = fe.diff(energy, epsilon)
    return sigma


# ---------------------------------------------------------------- 
# Nonlinear material models


# ---------------------------------------------------------------- 
# Borden2016_plasticity: https://doi.org/10.1016/j.cma.2016.09.005
def psi_aux_Borden(F, mu, kappa):
    J = fe.det(F)
    C = F.T * F
    Jinv = J**(-2 / 3)
    U = 0.5 * kappa * (0.5 * (J**2 - 1) - fe.ln(J))
    Wbar = 0.5 * mu * (Jinv * (fe.tr(C) + 1) - 3)
    return U, Wbar


def psi_plus_Borden(F, mu, kappa):
    J = fe.det(F)
    U, Wbar = psi_aux_Borden(F, mu, kappa)
    return fe.conditional(fe.lt(J, 1), Wbar, U + Wbar)


def psi_minus_Borden(F, mu, kappa):
    J = fe.det(F)
    U, Wbar = psi_aux_Borden(F, mu, kappa)
    return fe.conditional(fe.lt(J, 1), U, 0)


def psi_Borden(F, mu, kappa):
    J = fe.det(F)
    U, Wbar = psi_aux_Borden(F, mu, kappa)
    return  U + Wbar 
 

# ---------------------------------------------------------------- 
# Miehe2014_finie_strain: https://doi.org/10.1016/j.cma.2014.11.016
def psi_Miehe(F, mu, beta):
    J = fe.det(F)
    C = F.T * F
    W = mu / 2 * (fe.tr(C) + 1 - 3) + mu / beta * (J**(-beta) - 1)
    return W


def psi_plus_Miehe(F, mu, beta):
    return  psi_Miehe(F, mu, beta)


def psi_minus_Miehe(F, mu, beta):
    return 0

    
# ---------------------------------------------------------------- 
# first Piola-Kirchhoff stress
def first_PK_stress_plus(F, psi_plus):
    F = fe.variable(F)
    energy_plus = psi_plus(F)
    P_plus = fe.diff(energy_plus, F)
    return P_plus


def first_PK_stress_minus(F, psi_minus):
    F = fe.variable(F)
    energy_minus = psi_minus(F)
    P_minus = fe.diff(energy_minus, F)
    return P_minus


def first_PK_stress(F, psi):
    F = fe.variable(F)
    energy = psi(F)
    P = fe.diff(energy, F)
    return P
