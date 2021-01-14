import meshio
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from . import arguments


def show_map():
    x1 = np.linspace(0, 0.5, 101)
    y1 = 0.5*x1
    x2 = np.linspace(0.5, 1, 101)
    y2 = -6*x2**3 + 14*x2**2 -9*x2 + 2
    x3 = np.linspace(1, 2, 101)
    y3 = x3
    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x3, y3)
    plt.show()


def write_vtk(case_name):
    xdmf_filename = 'data/xdmf/{}/u_refine_0_mfem_True.xdmf'.format(case_name)
    with meshio.xdmf.TimeSeriesReader(xdmf_filename) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k) 
            meshio.write_points_cells('data/pvd/post_processing/{}/u{}.vtu'.format(case_name, k), points, cells, point_data={'m':point_data['m']})


def show_solution(case_name):

    control_points = np.load('data/numpy/{}/control_points.npy'.format(case_name))
    delta_u_recorded_coarse = np.load('data/numpy/{}/displacement_refine_0_mfem_True.npy'.format(case_name))

    plt.plot(control_points[:, 0], control_points[:, 1], linestyle='--', marker='o', color='red')

    print(control_points)

    path = 'data/pvd/post_processing/{}/u{}.vtu'.format(case_name, len(delta_u_recorded_coarse)-1)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    # reader.SetFileName("u000000.vtu")
    reader.Update()

    data = reader.GetOutput()

    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    u = vtk_to_numpy(data.GetPointData().GetVectors('m'))

    x_ = x + u

    print(x.shape)

    triangles = vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size // 4  # number of cells
    tri = np.take(triangles, [n for n in range(
        triangles.size) if n % 4 != 0]).reshape(ntri, 3)

    # fig = plt.figure(figsize=(8, 8))
    plt.triplot(x_[:, 0], x_[:, 1], tri, color='black', alpha=1., linewidth=0.5)

    # colors = u[:, 0]
    # tpc = plt.tripcolor(x_[:, 0], x_[:, 1], tri, colors, cmap='bwr', shading='flat', vmin=None, vmax=None)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.show()
    return x[:, :2], x_[:, :2]


if __name__ == '__main__':
    args = arguments.args
    # write_vtk('pure_shear')
    show_solution('pure_shear')