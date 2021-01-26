import numpy as np
import time
from .L_shape import LShape
from .three_point_bending import ThreePointBending
from .pure_shear import PureShear
from .pure_tension import PureTension
from .. import arguments


def run_case(args, CaseClass):
    post_processing_flag = True

    if post_processing_flag:

        args.map_type = 'smooth'
        args.local_refinement_iteration = 0
        pde = CaseClass(args)
        mfem_time = np.load('data/numpy/{}/time_refine_{}_mfem_{}.npy'.format(pde.case_name,  0, True))
        mfem_dof = pde.mesh.num_vertices()

        args.map_type = 'identity'
        args.local_refinement_iteration = 0
        pde = CaseClass(args)
        coarse_time = np.load('data/numpy/{}/time_refine_{}_mfem_{}.npy'.format(pde.case_name,  0, False))
        coarse_dof = pde.mesh.num_vertices()

        args.map_type = 'identity'
        args.local_refinement_iteration = 1
        pde = CaseClass(args)
        fine_time = np.load('data/numpy/{}/time_refine_{}_mfem_{}.npy'.format(pde.case_name,  1, False))
        fine_dof = pde.mesh.num_vertices()

        pde.post_processing()
        print("MFEM time {}, coarse time {}, fine time {}".format(mfem_time, coarse_time, fine_time))
        print("MFEM dof {}, coarse dof {}, fine dof {}".format(mfem_dof, coarse_dof, fine_dof))

    else:
        time_break0 = time.time()

        args.map_type = 'smooth'
        args.local_refinement_iteration = 0
        pde = CaseClass(args)
        pde.staggered_solve()

        time_break1 = time.time()

        args.map_type = 'identity'
        args.local_refinement_iteration = 0
        pde = CaseClass(args)
        pde.staggered_solve()

        time_break2 = time.time()

        args.map_type = 'identity'
        args.local_refinement_iteration = 1
        pde = CaseClass(args)
        pde.staggered_solve()    

        time_break3 = time.time()

        print("MFEM time {}, coarse time {}, fine time {}".format(time_break1 - time_break0, time_break2 - time_break1, time_break3 - time_break2))
        np.save('data/numpy/{}/time_refine_{}_mfem_{}.npy'.format(pde.case_name,  0, True), time_break1 - time_break0)
        np.save('data/numpy/{}/time_refine_{}_mfem_{}.npy'.format(pde.case_name,  0, False), time_break2 - time_break1)
        np.save('data/numpy/{}/time_refine_{}_mfem_{}.npy'.format(pde.case_name,  1, False), time_break3 - time_break2)


def main(args):
    run_case(args, PureTension)
    run_case(args, PureShear)
    run_case(args, ThreePointBending)
    run_case(args, LShape)


if __name__ == '__main__':
    args = arguments.args
    main(args)
