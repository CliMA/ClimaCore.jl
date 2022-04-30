# API

```@meta
CurrentModule = ClimaCore
```

## DataLayouts

```@docs
DataLayouts
DataLayouts.DataF
DataLayouts.IF
DataLayouts.IJF
DataLayouts.VF
DataLayouts.IFH
DataLayouts.IJFH
DataLayouts.VIFH
DataLayouts.VIJFH
```

## Geometry

### Coordinates
```@docs
Geometry.AbstractPoint
Geometry.float_type
```

Points represent _locations_ in space, specified by coordinates in a given
coordinate system (Cartesian, spherical, etc), whereas vectors, on the other hand,
represent _displacements_ in space.

An analogy with time works well: times (also called instants or datetimes) are
_locations_ in time, while, durations are _displacements_ in time.

**Note 1**: Latitude and longitude are specified via angles (and, therefore, trigonometric functions:
`cosd`, `sind`, `acosd`, `asind`, `tand`,...) in degrees, not in radians.
Moreover, `lat` (usually denoted by ``\theta``) ``\in [-90.0, 90.0]``, and `long`
(usually denoted by ``\lambda``) ``\in [-180.0, 180.0]``.

**Note 2:**: In a `Geometry.LatLongZPoint(lat, long, z)`, `z` represents the
elevation above the surface of the sphere with radius R (implicitly accounted for in the geoemtry).

**Note 3**: There are also a set of specific Cartesian points
(`Cartesian1Point(x1)`, `Cartesian2Point(x2)`, etc). These are occasionally
useful for converting everything to a full Cartesian domain (e.g. for visualization
purposes). These are distinct from `XYZPoint` as `ZPoint` can mean different
things in different domains.

### Vectors and vector fields
[Introduction to Vectors and Vector Fields in ClimaCore.jl](intro-to-vectors.md)

## Domains

### Types
```@docs
Domains.AbstractDomain
Domains.IntervalDomain
Domains.RectangleDomain
Domains.SphereDomain
```

### Interfaces
```@docs
Domains.boundary_names
```

## Meshes
A `Mesh` is a division of a domain into elements.

### Mesh types
```@docs
Meshes.AbstractMesh
Meshes.IntervalMesh
Meshes.RectilinearMesh
Meshes.AbstractCubedSphere
Meshes.EquiangularCubedSphere
Meshes.EquidistantCubedSphere
Meshes.ConformalCubedSphere
```

### Local element map

```@docs
Meshes.LocalElementMap
Meshes.IntrinsicMap
Meshes.NormalizedBilinearMap
```

### Mesh stretching
```@docs
Meshes.Uniform
Meshes.ExponentialStretching
```

### Interfaces
```@docs
Meshes.domain
Meshes.elements
Meshes.nelements
Meshes.is_boundary_face
Meshes.boundary_face_name
Meshes.opposing_face
Meshes.coordinates
Meshes.containing_element
Meshes.reference_coordinates
Meshes.SharedVertices
Meshes.face_connectivity_matrix
Meshes.vertex_connectivity_matrix
Meshes.linearindices
```

## Topologies
A `Topology` determines the ordering and connections between elements of a mesh.
### Types
```@docs
Topologies.AbstractTopology
Topologies.IntervalTopology
Topologies.Topology2D
```

### Interfaces
```@docs
Topologies.domain
Topologies.mesh
Topologies.nlocalelems
Topologies.vertex_coordinates
Topologies.opposing_face
Topologies.interior_faces
Topologies.boundary_tags
Topologies.boundary_tag
Topologies.boundary_faces
Topologies.vertices
Topologies.local_neighboring_elements
Topologies.ghost_neighboring_elements
```

## Spaces
A `Space` represents a discretized function space over some domain.
Currently two main discretizations are supported: Spectral Element Discretization
(both Continuous Galerkin and Discontinuous Galerkin types) and a staggered
Finite Difference Discretization. Combination of these two in the horizontal/vertical
directions, respectively, is what we call a _hybrid_ space.

Sketch of a 2DX hybrid discretization:

![3D hybrid discretization in a Cartesian domain](DiscretizationSketch.png)

```@docs
Spaces
```
### Finite Difference Spaces
ClimaCore.jl supports staggered Finite Difference discretizations. Finite Differences
discretize an interval domain by approximating the function by a value at either
the center of each element (also referred to as _cell_) (`CenterFiniteDifferenceSpace`),
or the interfaces (faces in 3D, edges in 2D or points in 1D) between elements
(`FaceFiniteDifferenceSpace`).

Users should construct either the center or face space from the mesh, then construct
the other space from the original one: this internally reuses the same data structures, and avoids allocating additional memory.

### Spectral Element Spaces

[Introduction to the Finite/Spectral Element Method](intro-to-sem.md)

### Quadratures


```@docs
Spaces.Quadratures.QuadratureStyle
Spaces.Quadratures.GLL
Spaces.Quadratures.GL
Spaces.Quadratures.Uniform
Spaces.Quadratures.degrees_of_freedom
Spaces.Quadratures.polynomial_degree
Spaces.Quadratures.quadrature_points
Spaces.Quadratures.barycentric_weights
Spaces.Quadratures.interpolation_matrix
Spaces.Quadratures.differentiation_matrix
Spaces.Quadratures.orthonormal_poly
```

#### Internals

```@docs
Spaces.dss_transform
Spaces.dss_untransform
Spaces.dss_interior_faces!
Spaces.dss_local_vertices!
Spaces.dss_ghost_faces!
Spaces.dss_ghost_vertices!
```

## RecursiveApply

```@docs
RecursiveApply
RecursiveApply.tuplemap
```

## Fields


```@docs
Fields.Field
Fields.coordinate_field
Fields.local_geometry_field
Base.zeros(::Spaces.AbstractSpace)
Base.ones(::Spaces.AbstractSpace)
Base.sum(::Fields.Field)
Fields.Statistics.mean(::Fields.Field)
Fields.LinearAlgebra.norm(::Fields.Field)
Fields.set!
```

## Limiters

### Interfaces
```@docs
Limiters.quasimonotone_limiter!
```
