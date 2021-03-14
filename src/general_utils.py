import meshio
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from . import arguments


def write_vtk(case_name, step):
    if case_name == 'analysis':
        xdmf_filename = 'data/xdmf/{}/u_refine_0_mfem_True_model_0.xdmf'.format(case_name)
    else:
        xdmf_filename = 'data/xdmf/{}/u_refine_0_mfem_True.xdmf'.format(case_name)
    with meshio.xdmf.TimeSeriesReader(xdmf_filename) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k) 
            if step == k:
                meshio.write_points_cells('data/pvd/post_processing/{}/u{}.vtu'.format(case_name, k), points, cells, point_data={'m':point_data['m']})


# To produce the figures in the section "Numerical examples"
def show_solution(case_name, step=None, physical_domain=True):
    control_points = np.load('data/numpy/{}/control_points.npy'.format(case_name))
    if case_name == 'analysis':
        step = 0
    else:
        delta_u_recorded_coarse = np.load('data/numpy/{}/displacement_refine_0_mfem_True.npy'.format(case_name))
        if step is None:
            step = len(delta_u_recorded_coarse)-1

    write_vtk(case_name, step)

    path = 'data/pvd/post_processing/{}/u{}.vtu'.format(case_name, step)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    # reader.SetFileName("u000000.vtu")
    reader.Update()

    data = reader.GetOutput()

    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    u = vtk_to_numpy(data.GetPointData().GetVectors('m'))

    fig = plt.figure(figsize=(8, 8))

    if physical_domain:
        x_ = x + u
        markersize = 4 if case_name == 'three_point_bending' else None
        plt.plot(control_points[:, 0], control_points[:, 1], linestyle='-', marker='o', markersize=markersize, color='red')
    else:
        x_ = x

    print(x.shape)

    triangles = vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size // 4  # number of cells
    tri = np.take(triangles, [n for n in range(
        triangles.size) if n % 4 != 0]).reshape(ntri, 3)

    # fig = plt.figure(figsize=(8, 8))
    plt.triplot(x_[:, 0], x_[:, 1], tri, color='black', alpha=1., linewidth=0.3)

    # colors = u[:, 0]
    # tpc = plt.tripcolor(x_[:, 0], x_[:, 1], tri, colors, cmap='bwr', shading='flat', vmin=None, vmax=None)
    plt.gca().set_aspect('equal')
    plt.axis('off')

    fig.savefig('data/pdf/{}/mesh_step_{}_physical_{}.pdf'.format(case_name, step, physical_domain), bbox_inches='tight')

    return x[:, :2], x_[:, :2]


def post_processing_mesh():
    # show_solution('analysis', step=0, physical_domain=False)
    # show_solution('analysis', step=0, physical_domain=True)

    show_solution('pure_tension')
    show_solution('pure_shear')
    show_solution('three_point_bending')
    show_solution('L_shape')


if __name__ == '__main__':
    args = arguments.args
    # write_vtk('pure_shear')
    # show_solution('pure_shear', step=0, physical_domain=False)\
    post_processing_mesh()
    plt.show()