#!/usr/bin/env python
# coding: utf-8
"""
Created Thu Dec 20 17:36:35 EST 2018

@author: ShengMao
Edited by J. Jaslove 5/5/20

This file creates geometry and mesh for trilayer 3D cylinders with a 
spherical cap at the end

it has a refine option

"""

##################################################################################
# import pkgs
##################################################################################

import os, meshio
import numpy as np
import dolfin as dl

##################################################################################
# functions: create_geometry, gen_mesh
##################################################################################

def gen_mesh(fileDir, problemDict, info = True):

    # buffer data from problem description
    Rout = problemDict["Rout"]
    Rin = problemDict["Rin"]
    tEp = problemDict["tEp"]
    tMed = problemDict["tMed"]
    length = problemDict["Height"] #problemDict["length"]
    Nin = problemDict["Nin"]
    Nout = problemDict["Nout"]
    Nmed = problemDict["Nmed"]
    dimension = 3
    
    # create geo files
    create_geometry(fileDir, Rout, Rin, tEp, tMed, length, Nin, Nmed, Nout)
    # convert geofile to gmsh and then xdmf
    cleanDir = fileDir.replace(" ", "\ ") # os.system command needs spaces in paths escaped
    os.system('gmsh -%d -v 1 %sgeometry.geo %sgeometry.msh' % (dimension, cleanDir, cleanDir))
    meshFile = os.path.join(cleanDir, "geometry")
    save_with_meshio(meshFile, dimension)
    
    # create a refined mesh for postprocessing
    create_geometry(fileDir, Rout, Rin, tEp, tMed, length, Nin*1.5, Nmed*1.5, Nout*1.5, refined=True)
    os.system('gmsh -%d -v 1 %sgeometry_refined.geo %sgeometry_refined.msh' % (dimension, cleanDir, cleanDir))
    meshFile = os.path.join(cleanDir, "geometry_refined")
    save_with_meshio(meshFile, dimension)

    # indicate the mesh process has been completed
    if info:
        print("--------------------------- mesh generation completed! -------------------------------")

    return



def save_with_meshio(meshFile, dimension):
    # input mesh file name and path witout file extension
    
    # Developed with massive hints from here:
    # https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-
    # to-mesh-xdmf-from-dolfin-convert-to-meshio/412/79
    # facet_region = mf
    # physical_region = cf

    # Select the correct element types for the domain and boundary based on
    # the spatial dimension of the mesh
    if dimension == 2:
        domainType = "triangle"
        boundaryType = "line"
        
    elif dimension == 3:
        domainType = "tetra"
        boundaryType = "triangle"

    else:
        raise ValueError("Mesh dimension must be 2 or 3")
        
    
    msh = meshio.read(meshFile + ".msh")

    # identify all triangular cells and line edges
    domain_cells = np.vstack(np.array([cells.data for cells in msh.cells
                                if cells.type == domainType]))
    
    facet_cells = np.vstack(np.array([cells.data for cells in msh.cells
                                      if cells.type == boundaryType]))
    

    # get the physical regions and physical boundaries
    facet_data = []
    domain_data = []

    if "gmsh:physical" in msh.cell_data_dict.keys(): 
        for key in msh.cell_data_dict["gmsh:physical"].keys():
            if key == boundaryType:
                if len(facet_data) == 0:
                    facet_data = msh.cell_data_dict["gmsh:physical"][key]
                else:
                    facet_data = np.vstack([facet_data, msh.cell_data_dict["gmsh:physical"][key]])
            elif key == domainType:
                if len(domain_data) == 0:
                    domain_data = msh.cell_data_dict["gmsh:physical"][key]
                else:
                    domain_data = np.vstack([domain_data, msh.cell_data_dict["gmsh:physical"][key]])

    
    if dimension == 2:
    # make the mesh 2D by removing the third column of points before saving the data to files
        points = msh.points[:,:2]
        
    else:
        points = msh.points

    domain_mesh = meshio.Mesh(points=points,
                           cells=[(domainType, domain_cells)],
                           cell_data={"name_to_read":[domain_data]})

    facet_mesh = meshio.Mesh(points=points,
                           cells=[(boundaryType, facet_cells)],
                           cell_data={"name_to_read":[facet_data]})
    

    meshio.write(meshFile + ".xdmf", domain_mesh)
    meshio.write(meshFile + "_facet_region.xdmf", facet_mesh)


def load_with_meshio(meshFile, dimension):
    # input mesh file name and path witout file extension
    # dimension = max spatial dimension of the mesh
    # (typically 2 or 3 for 2D or 3D)

    # read in the mesh
    mesh = dl.Mesh()
    with dl.XDMFFile(meshFile + ".xdmf") as infile:
        infile.read(mesh)
        
    # read in the facet boundaries
    mvc = dl.MeshValueCollection("size_t", mesh, dimension - 1)
    with dl.XDMFFile(meshFile + "_facet_region.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    facet_regions = dl.cpp.mesh.MeshFunctionSizet(mesh,mvc)

    # read in the defined physical regions
    mvc2 = dl.MeshValueCollection("size_t", mesh, dimension)
    with dl.XDMFFile(meshFile + ".xdmf") as infile:
        infile.read(mvc2, "name_to_read")
    physical_regions = dl.cpp.mesh.MeshFunctionSizet(mesh, mvc2)

    return mesh, facet_regions, physical_regions


def create_geometry(fileDir, Rout, Rin, tEp, tMed, length, Nin, Nmed, Nout, refined=False):
    # middle radius
    Rmus = Rin + tEp
    Rmed = Rmus + tMed
    # mesh fineness -> can be tuned
    meshSizeOut = (Rout-Rin)/Nout
    meshSizeEp  = tEp/Nin
    meshSizeMus = tMed/Nmed

    # open the .geo file for writing
    if refined:
        fileName = 'geometry_refined.geo'
    else:
        fileName = 'geometry.geo'

    fullFile = os.path.join(fileDir,fileName)

    string = """
    // Gmsh project created on Fri Apr 17 15:56:20 2020

    Rin = DefineNumber[ %f ];
    Rmed = DefineNumber[ %f ];
    Rmus = DefineNumber[ %f ];
    Rout = DefineNumber[ %f ];
    length = DefineNumber[ %f ];
    meshSize = DefineNumber[ %f ];
    meshSize2 = DefineNumber[ %f ];
    meshSize3 = DefineNumber[ %f ];

    Point(1) = {0.000000, 0.000000, 0.000000, meshSize};
    Point(2) = {0.000000, 0.000000, length, meshSize};
    Point(11) = {Rout, 0.000000, 0.000000, meshSize3};
    Point(12) = {-Rout, 0.000000, 0.000000, meshSize3};
    Point(13) = {Rout, 0.000000, length, meshSize3};
    Point(14) = {-Rout, 0.000000, length, meshSize3};
    Point(15) = {0.000000, 0.000000, length + Rout, meshSize3};
    Point(21) = {Rmed, 0.000000, 0.000000, meshSize2};
    Point(22) = {-Rmed, 0.000000, 0.000000, meshSize2};
    Point(23) = {Rmed, 0.000000, length, meshSize2};
    Point(24) = {-Rmed, 0.000000, length, meshSize2};
    Point(25) = {0.000000, 0.000000, length + Rmed, meshSize2};
    Point(31) = {Rmus, 0.000000, 0.000000, meshSize};
    Point(32) = {-Rmus, 0.000000, 0.000000, meshSize};
    Point(33) = {Rmus, 0.000000, length, meshSize};
    Point(34) = {-Rmus, 0.000000, length, meshSize};
    Point(35) = {0.000000, 0.000000, length + Rmus, meshSize};
    Point(41) = {Rin, 0.000000, 0.000000, meshSize};
    Point(42) = {-Rin, 0.000000, 0.000000, meshSize};
    Point(43) = {Rin, 0.000000, length, meshSize};
    Point(44) = {-Rin, 0.000000, length, meshSize};
    Point(45) = {0.000000, 0.000000, length + Rin, meshSize};

    Line(121) = {11, 21};
    Line(122) = {12, 22};
    Line(123) = {13, 23};
    Line(124) = {14, 24};
    Line(231) = {21, 31};
    Line(232) = {22, 32};
    Line(233) = {23, 33};
    Line(234) = {24, 34};
    Line(341) = {31, 41};
    Line(342) = {32, 42};
    Line(343) = {33, 43};
    Line(344) = {34, 44};

    Circle(11) = {11, 1, 12};
    Circle(12) = {12, 1, 11};
    Circle(13) = {13, 2, 14};
    Circle(14) = {14, 2, 13};
    Line(15) = {13, 11};
    Line(16) = {14, 12};
    Circle(17) = {13, 2, 15};
    Circle(18) = {15, 2, 14};
    Circle(21) = {21, 1, 22};
    Circle(22) = {22, 1, 21};
    Circle(23) = {23, 2, 24};
    Circle(24) = {24, 2, 23};
    Line(25) = {23, 21};
    Line(26) = {24, 22};
    Circle(27) = {23, 2, 25};
    Circle(28) = {25, 2, 24};
    Circle(31) = {31, 1, 32};
    Circle(32) = {32, 1, 31};
    Circle(33) = {33, 2, 34};
    Circle(34) = {34, 2, 33};
    Line(35) = {33, 31};
    Line(36) = {34, 32};
    Circle(37) = {33, 2, 35};
    Circle(38) = {35, 2, 34};
    Circle(41) = {41, 1, 42};
    Circle(42) = {42, 1, 41};
    Circle(43) = {43, 2, 44};
    Circle(44) = {44, 2, 43};
    Line(45) = {43, 41};
    Line(46) = {44, 42};
    Circle(47) = {43, 2, 45};
    Circle(48) = {45, 2, 44};

    Line Loop(101) = {11,122,-21,-121};
    Plane Surface(1) = {101};
    Line Loop(102) = {-12,122,22,-121};
    Plane Surface(2) = {102};
    Line Loop(103) = {17, 18, -13};
    Ruled Surface(3) = {103};
    Line Loop(104) = {17, 18, 14};
    Ruled Surface(4) = {104};
    Line Loop(105) = {15, 11, -16, -13};
    Ruled Surface(5) = {105};
    Line Loop(106) = {15, -12, -16, 14};
    Ruled Surface(6) = {106};
    Line Loop(107) = {25, 21, -26, -23};
    Ruled Surface(7) = {107};
    Line Loop(108) = {25, -22, -26, 24};
    Ruled Surface(8) = {108};
    Line Loop(109) = {35, 31, -36, -33};
    Ruled Surface(9) = {109};
    Line Loop(110) = {35, -32, -36, 34};
    Ruled Surface(10) = {110};
    Line Loop(111) = {45, 41, -46, -43};
    Ruled Surface(11) = {111};
    Line Loop(112) = {45, -42, -46, 44};
    Ruled Surface(12) = {112};
    Line Loop(201) = {21,232,-31,-231};
    Plane Surface(21) = {201};
    Line Loop(202) = {-22,232,32,-231};
    Plane Surface(22) = {202};
    Line Loop(203) = {27, 28, -23};
    Ruled Surface(23) = {203};
    Line Loop(204) = {27, 28, 24};
    Ruled Surface(24) = {204};
    Line Loop(301) = {31,342,-41,-341};
    Plane Surface(31) = {301};
    Line Loop(302) = {-32,342,42,-341};
    Plane Surface(32) = {302};
    Line Loop(303) = {37, 38, -33};
    Ruled Surface(33) = {303};
    Line Loop(304) = {37, 38, 34};
    Ruled Surface(34) = {304};
    Line Loop(403) = {47, 48, -43};
    Ruled Surface(43) = {403};
    Line Loop(404) = {47, 48, 44};
    Ruled Surface(44) = {404};

    Surface Loop(101) = {1,2,3,4,5,6,7,8,23,24};
    Volume(2) = {101};
    Surface Loop(201) = {21,22,23,24,7,8,9,10,33,34};
    Volume(3) = {201};
    Surface Loop(301) = {31,32,33,34,9,10,11,12,43,44};
    Volume(1) = {301};
    Physical Surface(1) = {11, 12, 43, 44};
    Physical Surface(2) = {2, 1, 22, 21, 31, 32};

    Physical Volume(3) = {1}; // epithelium
    Physical Volume(1) = {2}; // mesenchyme
    Physical Volume(2) = {3}; // smooth muscle
    """

    fileContents = string % (Rin, Rmed, Rmus, Rout,
                             length, meshSizeEp, meshSizeMus, meshSizeOut)
    
    with open( fullFile, "wt") as fid:
        print(fileContents, file=fid)

    return

        
