# MCTLASS 2.10

_MCTLASS 2.10 - August 2022_

MCTLASS is a collection of C++ template abstractions for implementing
high-performance matrix-multiplication (GEMM) and related computations at all levels
and scales within MACA. It incorporates strategies for hierarchical decomposition and
data movement similar to those used to implement mcBLAS and mcDNN.  MCTLASS decomposes
these "moving parts" into reusable, modular software components abstracted by C++ template
classes.  These thread-wide, warp-wide, block-wide, and device-wide primitives can be specialized
and tuned via custom tiling sizes, data types, and other algorithmic policy. The
resulting flexibility simplifies their use as building blocks within custom kernels
and applications.

MCTLASS implements high-performance Convolution via the implicit GEMM algorithm.
Implicit GEMM is the formulation of a convolution operation as a GEMM thereby taking advantage of
MCTLASS's modular GEMM pipeline.
This allows MCTLASS to build convolutions by reusing highly optimized warp-wide GEMM components and below.

## MCTLASS Template Library

```
include/                     # client applications should target this directory in their build's include paths

  mctlass/                   #  Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of architecture features (including instruction-level GEMMs)

    conv/                    # code specialized for convolution

    gemm/                    # code specialized for general matrix product computations

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory

    platform/                # Standard Library components

    reduction/               # bandwidth-limited reduction kernels that do not fit the "gemm" model

    transform/               # code specialized for layout, type, and domain transformations

    *                        # core vocabulary types, containers, and basic numeric operations
```
