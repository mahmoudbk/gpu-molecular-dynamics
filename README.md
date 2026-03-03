# GPU-Accelerated Molecular Dynamics Simulation

## Authors

- **Nicodème Rouger** — System architecture & algorithm design  
- **Mahmoud Becki** — CPU and CUDA implementation, optimization and benchmarking  

This project was developed over several months as part of a GPU computing project.

---

## Overview

This repository contains a high-performance 3D Molecular Dynamics (MD) simulation of particles interacting via a Lennard–Jones potential.

The objective was to:
- Model particle interactions in a periodic box
- Validate thermodynamic behavior (energy conservation, Maxwell–Boltzmann distribution)
- Design and implement a GPU-accelerated version
- Measure and analyze performance gains compared to a single-threaded CPU baseline

The implementation includes both sequential (CPU) and parallel (CUDA) solvers.

---

## Physical Model

- 3D cubic periodic domain
- Lennard–Jones potential (WCA cutoff)
- Velocity Verlet time integration
- Reduced units formulation
- Periodic Boundary Conditions (minimum image convention)

The simulation validates:
- Conservation of mechanical energy
- Convergence of temperature
- Maxwell–Boltzmann velocity distribution

---

## Numerical Approach

### Spatial Decomposition

A **block-centric spatial decomposition** strategy is used:

- Space is divided into cubic cells
- Each CUDA block corresponds to one spatial cell
- Two data structures are built:
  - **Duplicated candidate list** (particles may appear in up to 8 neighboring cells)
  - **Home ownership list** (each particle belongs to exactly one cell)

This reduces force computation complexity and enables locality-aware parallelization.

---

## GPU Implementation

The CUDA pipeline follows a structured workflow:

1. Count particle insertions per block (atomic operations)
2. Exclusive prefix-sum using **CUB**
3. Scatter particles into block lists
4. Block-centric force kernel
5. Velocity Verlet integration
6. Diagnostics & reductions (CUB-based)

### Kernel Mapping

- 1 CUDA block (CTA) = 1 spatial cell
- Threads stride over owned particles
- Force accumulation is thread-independent
- Designed for high thread-level parallelism

---

## Performance Results

Test configuration:
- ~1,000,000 particles
- NVIDIA Quadro RTX 6000
- Single-thread CPU baseline

| Version | Runtime |
|----------|----------|
| CPU (1 thread) | ~1497 s |
| GPU (CUDA) | ~36.7 s |

**≈ 40× speedup**

Performance was decomposed into:
- List construction
- Force computation
- Diagnostics

Scalability interpretation uses Amdahl’s law and GPU hardware concurrency analysis.

---

## Repository Structure
```
gpu-molecular-dynamics/
│
├── README.md
├── LICENSE
│
├── docs/
│ └── MD_GPU_Project_Report.pdf
│
├── cpu/
│ └── md_cpu.cpp
│
└── gpu/
└── md_cuda.cu

```
---

## Build Instructions

### CPU Version

```
g++ -O3 -std=c++17 md_cpu.cpp -o md_cpu
nvcc -O3 -std=c++17 md_cuda.cu -o md_cuda
```

References

Dennis Rapaport — The Art of Molecular Dynamics Simulation

Computational Soft Matter (NIC Series, Vol. 23)
