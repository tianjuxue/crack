import fenics as fe


# ---------------------------------------------------------------- 
# History functions

def history(u_new, H_old, I, psi_cr, psi_plus):
    psi_new = psi_plus(I + fe.grad(u_new))  
    history_max_tmp = fe.conditional(fe.gt(psi_new - psi_cr, 0), psi_new - psi_cr, 0)
    history_max = fe.conditional(fe.gt(history_max_tmp, H_old), history_max_tmp, H_old)
    return history_max


# ---------------------------------------------------------------- 
# Degradation functions

def g_d(d):
    m = 2
    degrad = m * ((1 - d)**3 - (1 - d)**2) + 3 * (1 - d)**2 - 2 * (1 - d)**3
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