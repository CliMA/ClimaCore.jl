module Meshes
using DocStringExtensions
export EquispacedRectangleMesh,
    rectangular_mesh,
    cube_panel_mesh,
    sphere_mesh,
    equispaced_rectangular_mesh,
    TensorProductMesh,
    Mesh2D,
    AbstractWarp,
    AbstractSphereWarp,
    EquiangularSphereWarp,
    EquidistantSphereWarp,
    ConformalSphereWarp,
    equiangular_sphere_warp,
    equiangular_sphere_unwarp,
    equidistant_sphere_warp,
    equidistant_sphere_unwarp,
    conformal_sphere_warp,
    conformal_sphere_unwarp

import ..Domains:
    Domains,
    IntervalDomain,
    RectangleDomain,
    SphereDomain,
    boundary_names,
    coordinate_type
import IntervalSets: ClosedInterval
import ..Geometry
using SparseArrays
import CubedSphere


"""
    AbstractMesh{dim}

A `Mesh` is an object which represents how we discretize a domain into elements.

It should be lightweight (i.e. exists on all MPI ranks), e.g for meshes stored
in a file, it would contain the filename.
"""
abstract type AbstractMesh{dim} end
const AbstractMesh1D = AbstractMesh{1}

"""
    AbstractMesh2D

Vertices and faces are numbered using the following convention.

```
 v4            f3           v3
   o------------------------o
   |                        |	  face    vertices
   |       <----------      |
   |                 |      |		f1 =>  v1 v2
f4 |                 |      | f2    f2 =>  v2 v3
   |                 |      |		f3 =>  v3 v4
   |                 |      |       f4 =>  v4 v1
   |       -----------      |
   |                        |
   o------------------------o
  v1           f1           v2
```
"""
const AbstractMesh2D = AbstractMesh{2}

include("common.jl")
include("interval.jl")
include("rectangle.jl")
include("cubedsphere.jl")

# deprecations
const EquispacedRectangleMesh = RectangleMesh
const equispaced_rectangular_mesh = RectangleMesh
const TensorProductMesh = RectangleMesh
EquiangularSphereWarp() = EquiangularCubedSphereMesh
EquidistantSphereWarp() = EquidistantCubedphereMesh
ConformalSphereWarp() = ConformalCubedSphereMesh
Mesh2D(
    domain::Domains.SphereDomain,
    ::Type{T},
    Ne,
) where {T <: AbstractCubedSphereMesh} = T(domain, Ne)


end # module
