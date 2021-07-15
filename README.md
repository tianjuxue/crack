### Mapped Phase field method for simulating quasi-static brittle fracture based on FEniCS

This repository contains an implementation of our paper (to appear) on _Computer Methods in Applied Mechanics and Engineering_: "[Mapped phase field method for brittle fracture](https://www.journals.elsevier.com/computer-methods-in-applied-mechanics-and-engineering)". We proposed to use a local reparametrization of the physical domain near crack, so that without modifying the finite element mesh we obtain higher accuracy for crack simulation based on the phase field method.

## Dependency and Usage

To run the cases, do
```bash
python -m src.cases.driver
```
under the root directory (```/crack```).

You may need to create folders to hold data, e.g.,
```bash
crack/data/pvd/brittle
```

## Demos

The following vedios correspond to the four MPFM numerical examples in "Section 5. Numerical examples" in the manuscript.
 
https://user-images.githubusercontent.com/45647025/125759760-9913d7bb-0369-4895-8d23-597e7ee8c83e.mp4

The left image shows x10 displacement, and the right image shows the reconstruced discretization of the physical domain.

https://user-images.githubusercontent.com/45647025/125759774-570c3f7c-e11a-4540-a1eb-b24980bace53.mp4

The left image shows x10 displacement, and the right image shows the reconstruced discretization of the physical domain.

https://user-images.githubusercontent.com/45647025/125759790-37906315-d34f-427e-ba70-91b434f0ee1f.mp4

The left image shows x5 displacement, and the right image shows the reconstruced discretization of the physical domain.

https://user-images.githubusercontent.com/45647025/125759803-821dab77-dab0-4715-baba-f861f65271fb.mp4

The left image shows x100 displacement, and the right image shows the reconstruced discretization of the physical domain.

## Citation

If you find our work useful in your research, please consider citing us.
