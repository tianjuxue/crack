import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os



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


    U = fe.VectorElement('CG', mesh.ufl_cell(), 2)  
    W = fe.FiniteElement("CG", mesh.ufl_cell(), 1)
    M = fe.FunctionSpace(mesh, U * W)

    WW = fe.FunctionSpace(mesh, 'DG', 0) 
    EE = fe.FunctionSpace(mesh, 'CG', 1) 

    presLoad = fe.Expression("t", t=0.0, degree=1)
    BC_u_left = fe.DirichletBC(M.sub(0).sub(0), fe.Constant(0),  Left())
    BC_u_right = fe.DirichletBC(M.sub(0).sub(0), presLoad,  Right())
    BC_u_corner = fe.DirichletBC(M.sub(0).sub(1), fe.Constant(0.0), Corner(), method='pointwise')

    BC = [BC_u_left, BC_u_right, BC_u_corner]     

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    right.mark(boundaries, 1)
    ds = fe.Measure("ds")(subdomain_data=boundaries)

    I = fe.Identity(dim)
    normal = fe.FacetNormal(mesh)

    m_test = fe.TestFunctions(M)
    m_delta = fe.TrialFunctions(M)
    m_new = fe.Function(M)

    (eta, zeta) = m_test
    (x_new, d_new) = fe.split(m_new)

    H_old = fe.Function(WW)

    E = fe.Function(EE)

    # G_ut = (g_d(d_new) * fe.inner(first_PK_stress_plus(I + fe.grad(x_new)), fe.grad(eta)) \
    #      + fe.inner(first_PK_stress_minus(I + fe.grad(x_new)), fe.grad(eta))) * fe.dx

    G_ut = g_d(d_new) * fe.inner(first_PK_stress(I + fe.grad(x_new)), fe.grad(eta)) * fe.dx
  

    # G_d = H(x_new, H_old) * zeta * g_d_prime(d_new, g_d) * fe.dx \
    #     + 2 * psi_cr * (zeta * d_new + l0**2 * fe.inner(fe.grad(zeta), fe.grad(d_new))) * fe.dx  

    G_d = H_old * zeta * g_d_prime(d_new, g_d) * fe.dx \
        + 2 * psi_cr * (zeta * d_new + l0**2 * fe.inner(fe.grad(zeta), fe.grad(d_new))) * fe.dx  

    # G_d = psi_plus(I + fe.grad(x_new)) * zeta * g_d_prime(d_new, g_d) * fe.dx \
    #     + Gc_0 * (1 / (2 * l0) * zeta * d_new + 2 * l0 * fe.inner(fe.grad(zeta), fe.grad(d_new))) * fe.dx  


    # G_ut = (g_d(d_new) * fe.inner(cauchy_stress_plus(strain(fe.grad(x_new))), strain(fe.grad(eta))) \
    #      + fe.inner(cauchy_stress_minus(strain(fe.grad(x_new))), strain(fe.grad(eta)))) * fe.dx
 
    # G_d = linear_elasticity_psi_plus(strain(fe.grad(x_new))) * zeta * g_d_prime(d_new, g_d) * fe.dx \
    #     + Gc_0 * (1 / (2 * l0) * zeta * d_new + 2 * l0 * fe.inner(fe.grad(zeta), fe.grad(d_new))) * fe.dx  


    G = G_ut + G_d

    dG = fe.derivative(G, m_new)
    p = fe.NonlinearVariationalProblem(G, m_new, BC, dG)
    solver = fe.NonlinearVariationalSolver(p)
 
    vtkfile_u = fe.File('data/pvd/circular_holes/u.pvd')
    vtkfile_d = fe.File('data/pvd/circular_holes/d.pvd')
    vtkfile_e = fe.File('data/pvd/circular_holes/e.pvd')

    t = t_i
    sigmas = []
    deltaUs = []
    forceForm = (first_PK_stress(I + fe.grad(x_new))[0, 0])*ds(1)

    # while t <= t_f:
    for disp in displacements:

        t += dt

        print(' ')
        print('=================================================================================')
        print('>> disp boundary condition = {} [mm]'.format(disp))
        print('=================================================================================')

        # presLoad.t = t*disp_rate
        presLoad.t = disp

        newton_prm = solver.parameters['newton_solver']
        newton_prm['maximum_iterations'] = 1000
        newton_prm['linear_solver'] = 'mumps'   
        newton_prm['absolute_tolerance'] = 1e-4

        if disp > 11 and disp <= 14 :
            newton_prm['relaxation_parameter'] = 0.2
        elif disp > 14 and disp <= 26.5:
            newton_prm['relaxation_parameter'] = 0.1
        elif disp > 26.5:
            newton_prm['relaxation_parameter'] = 0.02

        solver.solve()

        H_old.assign(fe.project(H(x_new, H_old), WW))

        E.assign(fe.project(psi(I + fe.grad(x_new)), EE))
        # E.assign(fe.project(first_PK_stress(I + fe.grad(x_new))[0, 0], EE))
        
 
        print(
            '=================================================================================')
        print(' ')


        (x_plot, d_plot) = m_new.split()
        x_plot.rename("Displacement", "label")
        d_plot.rename("Phase field", "label")

        vtkfile_u << x_plot
        vtkfile_d << d_plot
        vtkfile_e << H_old
    #     deltaUs.append(t * disp_rate)
    #     sigmas.append(fe.assemble(forceForm))



    # plt.clf()
    # plt.plot(deltaUs, np.array(sigmas)/G)
    # plt.savefig("data/png/phase_field/stress-strain-curve.png")
 

if __name__ == '__main__':
    phase_field()
    plt.show()