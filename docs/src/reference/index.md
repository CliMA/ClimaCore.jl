# API overview

ClimaCore is organized as one module per subsystem. A model is assembled from
the bottom of this table up: a domain is meshed, the mesh gets a topology, the
topology and a quadrature rule give a grid, a grid and a staggering give a
space, and fields live on spaces.

| Module         | Provides                                                                    |
|:-------------- |:--------------------------------------------------------------------------- |
| `Domains`      | The continuous domain: intervals, rectangles, the sphere                    |
| `Meshes`       | Element meshes: intervals, rectilinear planes, the equiangular cubed sphere |
| `Topologies`   | Element connectivity, including distributed (MPI) topologies                |
| `Geometry`     | Coordinates, local geometry, covariant/contravariant/Cartesian vectors      |
| `Quadratures`  | Gauss–Lobatto–Legendre and Gauss–Legendre nodes and weights                 |
| `Grids`        | Spectral-element, finite-difference, and extruded grids; the CG/DG switch   |
| `CommonGrids`  | Convenience constructors for the standard grids                             |
| `Spaces`       | Function spaces on grids; center/face staggering; DSS                       |
| `CommonSpaces` | Convenience constructors for the standard spaces                            |
| `Fields`       | `Field` and `FieldVector`, the data containers and state vectors            |
| `DataLayouts`  | The memory layouts behind fields                                            |
| `Operators`    | Spectral-element, finite-difference, and DG operators; tendency completion  |
| `Limiters`     | Quasi-monotone and mass-borrowing limiters                                  |
| `Hypsography`  | Terrain-following vertical coordinates                                      |
| `MatrixFields` | Banded matrix fields and solvers for implicit vertical problems             |
| `Remapping`    | Interpolation between spaces and to latitude–longitude grids                |
| `InputOutput`  | HDF5 checkpoint writers and readers                                         |
| `Utilities`    | Half-integer indexing, caches, and other shared helpers                     |
| `DebugOnly`    | Hooks for locating NaNs and mismatched spaces                               |

## Export conventions

Only `Domains`, `Meshes`, `Geometry`, `Quadratures`, `DataLayouts`,
`Limiters`, `MatrixFields`, `Remapping`, `CommonGrids`, and `CommonSpaces`
export names. Everything in `Spaces`, `Grids`, `Operators`, `Fields`,
`Topologies`, `Hypsography`, and `InputOutput` is called with its module
qualifier, as in `Spaces.SpectralElementSpace2D` or `Operators.GradientC2F`.
The usual import line is

```julia
import ClimaCore: Domains, Meshes, Topologies, Spaces, Fields, Geometry, Operators
```

## Naming conventions

  - Finite-difference operators name the staggering they map between: `C2F`
    from cell centers to faces, `F2C` from faces to centers.
  - Spectral-element operators take a form-type parameter: `Divergence()` is the
    strong form and `Divergence{WeakForm}()` the weak form, and likewise for
    `Gradient` and `Curl`.
  - A trailing `!` marks a function that mutates its first argument.
  - The Galerkin discretization of a spectral-element grid is `Grids.CG()` or
    `Grids.DG()`; `Spaces.discretization(space)` reads it back.
