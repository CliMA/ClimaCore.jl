---
title: 'TBD: ClimaCore: the dycore library for CliMA's proposed ESM'
tags:
  - earth system modeling
  - dycore
  - flexible discretizations
  - spectral elements
  - staggered finite differences
  - matrix-free
authors:
  - name: Simon Byrne
    orcid: XXXX
    affiliation: 1
  - name: Charles Kawczynski
    orcid: XXXX
    affiliation: 1
  - name: Valeria Barra
    orcid: 0000-0003-1129-2056
    affiliation: 1
  - name: Jake Bolewski
    orcid: XXXX
    affiliation: 1
  - name: Sriharsha Kandala
    orcid: XXXX
    affiliation: 1
  - name: Zhaoyi Shen
    orcid: XXXX
    affiliation: 1
  - name: Jia He
    orcid: XXXX
    affiliation: 1
  - name: Akshay Sridhar
    orcid: XXXX
    affiliation: 1
  - name: Dennis Yatunin
    orcid: XXXX
    affiliation: 1
  - name: Ben Mackay
    orcid: XXXX
    affiliation: 1
  - name: Simon Danish ?
    orcid: XXXX
    affiliation: 2
  - name: Kiran Pamnany
    orcid: XXXX
    affiliation: 3
  - name: Julia Sloan
    orcid: XXXX
    affiliation: 1
  - name: Toby Bischoff
    orcid: XXXX
    affiliation: 1
  - name: Lenka Novak
    orcid: XXXX
    affiliation: 1
  - name: Daniel (Zhengyu) Huang
    orcid: XXXX
    affiliation: 1
  - name: Oswald Knoth
    orcid: XXXX
    affiliation: 4
  - name: Paul Ullrich
    orcid: XXXX
    affiliation: 5
  -name: Tapio Schneider
    orcid: XXXX
    affiliation: 1

affiliations:
 - name: California Institute of Technology
   index: 1
 - name: ?
   index: 2
 - name: Relational AI
   index: 3
 - name: TROPOS
   index: 4
 - name: University of California Davis
   index: 5
date: 31 May 2023
bibliography: paper.bib
---

<!-- The list of authors of the software and their affiliations, using the correct format was generated from https://github.com/CliMA/ClimaCore.jl/graphs/contributors + Paul, Oswald and Tapio. The order of co-authors reflects the order of GitHub contributions by commits. Contributors with only one commit are not included at the moment. We can revise this policy if anyone does not agree with it. Also, anyone included who does not wish to be included can be removed, of course. -->

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

The Climate Modelling Alliance ([CliMA](https://clima.caltech.edu/)) is developing a new Earth System Model (ESM), entirely written in the [Julia](https://julialang.org/) programming language [@julia-paper]. The main goal of the project is to build an ESM that automatically learns from diverse data sources to produce accurate climate predictions with quantified uncertainties.

`ClimaCore.jl` is a new open-source software library that provides a suite of tools for constructing spatial discretizations, underlying many of the CliMA model components and providing the dynamical core (_dycore_) for the atmosphere and land components of the ESM. It is designed with a high-level application programming interface (API), which facilitates modularity, composition of differential operators, definition of flexible discretizations, and library reuse.

# Statement of need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

Earth system model dynamical cores are traditionally hard-coded to specific equation sets, with fixed spatial and temporal discretizations, and specific geometries, such as spherical geometries for general circulation models (GCM) in the atmosphere or Cartesian ones for large-eddy simulations (LES) (see, for instance, the High Order Method Modeling Environment (HOMME) used by the Energy Exascale Earth System Model (E3SM) [@E3SM]).

`ClimaCore.jl` aims to be a more flexible approach, inspired by other mathematical software libraries for constructing spatial discretizations of partial differential equations (PDEs), such as PETSc [@petsc-web-page; @petsc-user-ref; @petsc-efficient], libCEED [@libceed-joss-paper; @libceed-user-manual], MFEM [@MFEMlibrary; @mfem-paper], deal.II [@dealII92], Firedrake [@firedrake], and FeniCS [@FeniCS].

However, ESMs tend to have some specific properties, some of which can leverage modern heterogenous architectures (including CPUs and GPUs) or modern ML/AI tools, that there are advantages to developing a new library.

Firstly, ESMs often use a very skewed aspect ratio: when performing global simulations, it is common to use a resolution of O(100km) in the horizontal, compared with O(100m) in the vertical. This leads to several other design considerations:
- Implicit-explicit (IMEX) timestepping are typically used, with only the vertical components of the governing equations handled implicitly, known as horizontally-explicit, vertically-implicit (HEVI) schemes.
- Distributed memory parallelism is only used in the horizontal direction, which avoids the need for communication inside the implicit step.
- Meshes are not fuly unstructured, instead the 3D meshes are constructed by extruding the 2D horizontal mesh. Finally
- Different discretizations may be used in each dimension, for example our current atmosphere model uses a specral element discretization in the horizontal, with a staggered finite difference discretization in the verfical.

Secondly, we aim to support both local high-resolution (box) configurations and global lower-resolution (spherical) simulations, using a unified equation set and discretizations. Specifically, we define the equations using a local Cartesian coordinate system: in a box configuration, this corresponds to the usual global coordinate system, but use latitude, longitude and altitude on a sphere: this means that, `z` is used to refer to  the Cartesian coordinate in the box, or the altitude from the surface on the sphere. Similarly, for vector quantities, `u`, `v`, `w` refer to the Cartesian components in the box, or the zonal, meridonal and radial components on the sphere. Additionally, for our atmosphere model we make use of the so called "vector invariant form", which specifies the equations directly in covariant and contravariant components (with respect to the mesh elements).




<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->

`ClimaCore.jl` is currently being used as the basis for the atmosphere and land model components of the CliMA earth system model.



# Introduction


<!-- from README -->
`ClimaCore.jl` is a the dynamical core (_dycore_) of the atmosphere and land models, providing discretization tools to solve the governing equations of the ESM component models.

`ClimaCore.jl`'s high-level API facilitates modularity and composition of differential operators and the definition of flexible discretizations. This, in turn, is coupled with low-level APIs that support different data layouts, specialized implementations, and flexible models for threading, to better face high-performance optimization, data storage, and scalability challenges on modern HPC architectures. `ClimaCore.jl` is designed to be performance portable and can be used in a distributed setting with CPU and GPU clusters.

## Why Julia?
Julia is a compiled, dynamic programming language with great potential in numerical analysis and applied sciences. One of its defining features is multiple dispatch, in which the method of a function to be called is dynamically chosen based on run-time types. Multiple dispatch both increases run-time speed, and allows users to easily extend existing functions to suit their own needs. Another of Julia's useful qualities is array broadcasting, which facilitates efficient operations involving large data structures without extra memory allocations. Julia balances minimal barrier to entry with superior performance compared to similar easy-to-use languages, such as Python or Matlab.

## Technical aims and current support


* Support both large-eddy simulation (LES) and general circulation model (GCM) configurations for the atmosphere.
* A suite of tools for constructing space discretizations.
* Horizontal spatial discretization:
    - Supports both continuous Galerkin (CG) and discontinuous Galerkin (DG) spectral element discretizations.
* Flexible choice of vertical discretization: currently staggered finite differences.
* Support for different geometries (Cartesian, spherical), with governing equations discretized in terms of covariant/contravariant  vectors for curvilinear, non-orthogonal systems and Cartesian vectors for Euclidean spaces.
* `Field` abstraction: a data structure to describe a mathematical field defined on a given space. It can be:
    - Scalar, vector or struct-valued
    - Stores values, geometry, and mesh information
    - Flexible memory layouts: Array-of-Structs (AoS), Struct-of-Arrays (SoA),Array-of-Struct-of-Arrays (AoSoA)
    - Useful overloads: `sum` (integral), `norm`, etc.
    - Compatible with [`DifferentialEquations.jl`](https://diffeq.sciml.ai/stable/) time steppers.
* Composable operators via broadcasting: apply a function element-wise to an array; scalar values are broadcast over arrays
* Fusion of multiple operations; can be specialized for custom functions or argument types (e.g. `CuArray` compiles and applies a custom CUDA kernel).
* Operators (`grad`, `div`, `interpolate`) are “pseudo-functions”: Act like functions when broadcasted over a `Field`; fuse operators and function calls.
* Add element node size dimensions to type domain
    - i.e., specialize on polynomial degree
    - important for GPU kernel performance.
* Flexible memory layouts allow for flexible threading models (upcoming):
    - CPU thread over elements
    - GPU thread over nodes.

# Example [Nat]


# Results
(scaling/visualization)






# Tentative TODOs to have the package in order
Draft of a "white paper" that briefly mentions all the nice properties of the library:  extensibility, composability, ease-of-use, library-reuse, performance-portability, scalability, GPU support.

Improve Docs:
- [x] Getting started/How to guide
- [x] Contributing guide + Code of Conduct
- [ ] Examples documentation (equations set, what to expect from each example, artifacts, if included)


Improve Unit Tests:
- [x] Unit tests: strive for best code coverage: e.g., double check that all operators are tested


Performance:
- [ ] Distributed computing capability (re-run latest scaling studies)

Cleanup: [Dennis]
- [ ] Remove dead code, comments and TODOs


# TODOs for the paper
- [x] Abstract/Summary
- [x] Statement of need
- [x] Introduction
  - [x] Why Julia? [Julia]
  - [ ] Why ClimaCore?
- [ ] APIs
   - [x] high-level API [Valeria]
   - [ ] low-level API [Charlie]
- [ ] Examples
- Performance Portability (ClimaComms) [Sriharsha]
  - [ ] Multi-threading capability
  - [ ] Include initial GPU support


- [ ] References

# API
## High-level API [Valeria]
### Spatial discretizations
To construct a spatial discretization, in ClimaCore's API, we need 4 elements:

- Domain: defines the bounding box of a domain. It can be an `IntervalDomain` (1D), a `RectangleDomain` (2D), `SphereDomain` (2D, which can be extruded in 3D).
- `Mesh`: a division of a domain into elements.
- `Topology`: determines the ordering and connections between elements of a mesh.
- `Space`: represents a discretized function space over some domain. Currently two main discretizations are supported: Spectral Element Discretization (both Continuous Galerkin and Discontinuous Galerkin types), a staggered Finite Difference Discretization, and the combination of these two in what we call a hybrid space.
- `Field`: on a given `Space`, we can construct a `Field`, which can represent mathematically either a scalar-valued field, a vector-valued field, or a combination of these two. A field is simply the association of a space and the values at each node in the space.
### Composable Spatial Operators
Operators can compute spatial derivative operations:

 - For performance reasons, we need to be able to "fuse" multiple operators and function applications.
 - Julia provides a tool for this: broadcasting, with a very flexible API.

We can think of operators are "pseudo-functions": can't be called directly, but act similar to functions in the context of broadcasting. They are matrix-free, in the sense that we define the action of the operator directly on a field, without explicitly assembling the matrix representing the discretized operator. ClimaCore.jl supports Spectral element operators for the horizontal direction and finite difference ones for the vertical direction.
### Other operations
- DSS, limiters, remapping:
Since in a finite element representation a given field is discretely defined across subdomains (elements), it might have discontinuous values across element boundaries. When we use a _continuous Galerkin_ (CG) spectral element discretization, we must ensure that the state variable is continuous across element boundaries. Therefore, we apply a so-called  Direct Stiffness Summation (DSS) operator to ensure this continuity by removing redundant degrees of freedom multiplicity across element boundaries/corners.

For atmospheric model applications, it may be necessary to ensure monotonocity or positivity of some quantities (e.g., moisture). For this reason, ClimaCore.jl supports a class of so-called _flux-limiters_ that take care of finding values that do not satisfy constraints and bringing these values to a closest desirable constraint.

In ESMs, for postprocessing and visualization purposes, it is often necessary to map data from a spherical grid (in spherical or Cartesian coordinates) to a latitude and longitude grid. To achieve this, ClimaCore.jl uses the external software package _TempestRemap_, a consistent and monotone remapping package for arbitrary grid geometry [@TempestRemap1;@TempestRemap2].

Remapping is following the conservative, consistent and (if required) monotonous method, detailed in [Ullrich and Taylor 2015](https://journals.ametsoc.org/view/journals/mwre/143/6/mwr-d-14-00343.1.xml). This is a linear remapping operator, which obtains the target field by multiplying the source field with a sparse matrix of source-to-target weights ($\psi_i = R_{ij} \psi_j$). The weight for cartesian domains are generated in ClimaCore, and for the equiangular cubed sphere weight generation we use TempestRemap. The sparse matrix multiply is (will be soon! TODO) parallelized. If monotonicity is not required, this method can capitalize on the high order of our CG discretization.

### ODE API compatibility [Dennis]



## Low-level API [Charlie]
ClimaCore has layered abstractions for data layouts to flexibly and efficiently read and write data of different sizes. The space and field abstraction layers, which contain local geometry and field variable values, sit ontop of this layer so that all fields leverage the flexibility of the data layouts. In addition, operators for slicing data in different ways, for example column-wise and level-wise are supported for fields. These layers help separate concerns of what variables are stored, and in which shape they exist, from the variables that users interact with. One benefit of this is that adjusting memory layout or memory access patterns can be changed internally, which can be very helpful to improve performance.

# Parallelism

## ClimaComms [Sriharsha]

## GPU

<!-- Acknowledgement of any financial support. -->
# Acknowledgements
We acknowledge contributions from several others who played a role in the evolution of this library, especially contributors and users of an eralier iteration of this effort, [ClimateMachine.jl](https://github.com/CliMA/ClimateMachine.jl) [@climate_machine_zenodo]. The development of this package was supported by the generosity of Eric and Wendy Schmidt by recommendation of the Schmidt Futures program, and by the Defense Advanced Research Projects Agency (Agreement No. HR00112290030).

# References
