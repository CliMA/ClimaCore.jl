# Geometry

```@meta
CurrentModule = ClimaCore
```

Coordinates, vectors and tensors with their bases, and the local and global
geometry of a grid ([Mathematical framework](../explanation/math_framework.md),
[Hybrid grids and generalized coordinates](../explanation/geometry.md)).

## Points

Points are locations in space, given by coordinates in a coordinate system;
vectors are displacements. The distinction is that between a time and a
duration.

```@docs
Geometry.AbstractPoint
Geometry.Abstract1DPoint
Geometry.Abstract2DPoint
Geometry.Abstract3DPoint
Geometry.float_type
```

Latitude and longitude are in degrees: `lat ∈ [−90, 90]`, `long ∈ [−180, 180]`,
and the trigonometric functions to use on them are `sind`, `cosd`, `tand`, and
their inverses. In a `Geometry.LatLongZPoint(lat, long, z)`, `z` is the height
above the surface of the sphere; the radius is part of the global geometry. The
Cartesian points `Cartesian1Point`, `Cartesian2Point`, `Cartesian3Point`,
`Cartesian12Point`, and `Cartesian123Point` refer to a single global Cartesian
frame, used when everything is mapped to one frame for output or
visualization; they are distinct from `XPoint`, `XYPoint`, and `XYZPoint`,
whose meaning depends on the domain. The concrete point types are listed in the
`AbstractPoint` docstring; the coordinates of a point are read as properties
(`p.lat`, `p.z`, `p.x`, …) or with `component`.

```@docs
Geometry.component
Geometry.coordinate
Geometry.tofloat
Geometry.euclidean_distance
Geometry.great_circle_distance
```

## Vectors and tensors

```@docs
Geometry.Tensor
Geometry.Components
Geometry.CovariantVector
Geometry.ContravariantVector
Geometry.LocalVector
Geometry.:⊗
Geometry.outer
Geometry.project
Geometry.transform
Geometry.LinearAlgebra.norm(::Geometry.AbstractTensor, ::Geometry.LocalGeometry)
Geometry.LinearAlgebra.norm_sqr(::Any, ::Geometry.LocalGeometry)
```

```@docs
Geometry.CartesianVector
Geometry.CartesianPoint
```

## Local geometry

The local geometry of a node: its coordinates, the Jacobian, and the metric
terms of the coordinate map. `Fields.local_geometry_field(space)` exposes it
as a field.

```@docs
Geometry.LocalGeometry
Geometry.SurfaceGeometry
Geometry.undertype
```

## Global geometry

```@docs
Geometry.AbstractGlobalGeometry
Geometry.CartesianGlobalGeometry
Geometry.AbstractSphericalGlobalGeometry
Geometry.SphericalGlobalGeometry
Geometry.ShallowSphericalGlobalGeometry
Geometry.DeepSphericalGlobalGeometry
Geometry.radius
```
