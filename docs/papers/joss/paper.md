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
  - name: Dennis Yatunin
    orcid: XXXX
    affiliation: 1
  - name: Ben Mackay
    orcid: XXXX
    affiliation: 1
  - name: Akshay Sridhar
    orcid: XXXX
    affiliation: 1
  - name: Simon Danish ?
    orcid: XXXX
    affiliation: 2
  - name: Kiran Pamnany
    orcid: XXXX
    affiliation: 3
  - name: Toby Bischoff
    orcid: XXXX
    affiliation: 1
  - name: LenkaNovak
    orcid: XXXX
    affiliation: 1
  - name: Julia Sloan
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
date: 10 January 2023
bibliography: paper.bib
---

<!-- The list of authors of the software and their affiliations, using the correct format was generated from https://github.com/CliMA/ClimaCore.jl/graphs/contributors + Paul, Oswald and Tapio. The order of co-authors reflects the order of GitHub contributions by commits. Contributors with only one commit are not included at the moment. We can revise this policy if anyone does not agree with it. Also, anyone included who does not wish to be included can be removed, of course. -->

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

The Climate Modelling Alliance ([CliMA](https://clima.caltech.edu/)) is developing a new Earth System Model (ESM), entirely written in the [Julia](https://julialang.org/) programming language [@julia-paper]. The main goal of the project is to build an ESM that automatically learns from diverse data sources to produce accurate climate predictions with quantified uncertainties.

`ClimaCore.jl` is a new open-source software library that provides a suite of tools for constructing spatial discretizations, underlying many of the CliMA model components and providing the dynamical core (_dycore_) for the atmosphere and land components of the ESM. It is designed with a high-level application programming interface (API), which facilitates modularity, composition of differential operators, definition of flexible discretizations, and library reuse.

# Statement of need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

Earth system model dynamical cores are traditionally hard-coded to specific equation sets, with fixed spatial and temporal discretizations, and specific geometries, such as spherical geometries for general circulation models (GCM) or Cartesian ones for large-eddy simulations (LES) (see, for instance, the High Order Method Modeling Environment (HOMME) used by the Energy Exascale Earth System Model (E3SM) [@E3SM]).

`ClimaCore.jl` aims to be a more flexible approach, inspired by other mathematical software libraries for constructing spatial discretizations of partial differential equations (PDEs), such as PETSc [@petsc-web-page; @petsc-user-ref; @petsc-efficient], libCEED [@libceed-joss-paper; @libceed-user-manual], MFEM [@MFEMlibrary; @mfem-paper], deal.II [@dealII92], Firedrake [@firedrake], and FeniCS [@FeniCS].

However, ESMs tend to have some specific properties, some of which can leverage modern heterogenous architectures (including CPUs and GPUs) or modern ML/AI tools, that there are advantages to developing a new library

  - very skewed aspect ratio for the atmosphere component: O(100km) in the horizontal vs O(10m) in the vertical;
  - implicit-explicit (IMEX) timestepping, with only the vertical parts handled implicitly: horizontally-explicit, vertically-implicit (HEVI) schemes;
  - use of different discertizations in each dimension, for example our current atmosphere model uses a specral element discretization in the horizontal, with a staggered finite difference discretization in the verfical;
  - don't need a fully unstructured mesh: 3D meshes are constucted by extruding a 2D mesh;
  - distributed parallely only in the horizontal direction;
  - support both Cartesian and spherical geometries and vector bases: on a sphere, vector components are typical specified in spherical basis (zonal, meridonal, radial);
  - capability to run embedded regional high-resolution and global simulations.



<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->

`ClimaCore.jl` is currently being used as the basis for the atmosphere and land model components of the CliMA earth system model.



# Introduction



<!-- from README -->
`ClimaCore.jl` is a the dynamical core (_dycore_) of the atmosphere and land models, providing discretization tools to solve the governing equations of the ESM component models.

`ClimaCore.jl`'s high-level API facilitates modularity and composition of differential operators and the definition of flexible discretizations. This, in turn, is coupled with low-level APIs that support different data layouts, specialized implementations, and flexible models for threading, to better face high-performance optimization, data storage, and scalability challenges on modern HPC architectures. `ClimaCore.jl` is designed to be performance portable and can be used in a distributed setting with CPU and GPU clusters.

## Technical aims and current support
* Support both large-eddy simulation (LES) and general circulation model (GCM) configurations for the atmosphere.
* A suite of tools for constructing space discretizations.
* Horizontal spectral elements:
    - Supports both continuous Galerkin (CG) and discontinuous Galerkin (DG) spectral element discretizations.
* Flexible choice of vertical discretization (currently staggered finite differences)
* Support for different geometries (Cartesian, spherical), with governing equations discretized in terms of covariant  vectors for curvilinear, non-orthogonal systems and Cartesian vectors for Euclidean spaces.
* `Field` abstraction:
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




# Tentative TODOs
Draft of a "white paper" that briefly mentions all the nice properties of the library:  extensibility, composability, ease-of-use, library-reuse, (performance-portability?), (scalability?). This will give ClimaCore its proper DOI and citability tool.

Improve Docs:
- [x] Getting started/How to guide
- [x] Contributing guide + Code of Conduct (?)
- [ ] Examples documentation (equations set, what to expect from each example, artifacts, if included)


Improve Unit Tests:
- [ ] Unit tests: strive for best code coverage: e.g., double check that all operators are tested

Examples:
- [ ] Address memory usage and OoM issues when examples are run locally


Performance:
- [ ] Distributed computing capability (re-run latest scaling studies)
- [ ] Multi-threading capability?
- [ ] Might include initial GPU support?


<!-- Acknowledgement of any financial support. -->
# Acknowledgements
We acknowledge contributions from several others who played a role in the evolution of this library, especially contributors and users of an eralier iteration of this effort, [ClimateMachine.jl](https://github.com/CliMA/ClimateMachine.jl) [@climate_machine_zenodo]. The development of this package was supported by the generosity of Eric and Wendy Schmidt by recommendation of the Schmidt Futures program, and by the Defense Advanced Research Projects Agency (Agreement No. HR00112290030).

# References
