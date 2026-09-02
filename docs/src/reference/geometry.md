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
whose meaning depends on the domain.

## Vectors and tensors

```@docs
Geometry.Tensor
Geometry.:⊗
Geometry.project
Geometry.transform
```

## Local geometry

The local geometry of a node: its coordinates, the Jacobian, and the metric
terms of the coordinate map. `Fields.local_geometry_field(space)` exposes it
as a field.

```@docs
Geometry.LocalGeometry
Geometry.SurfaceGeometry
```

## Global geometry

```@docs
Geometry.AbstractGlobalGeometry
Geometry.CartesianGlobalGeometry
Geometry.SphericalGlobalGeometry
```

## Internal helpers

```@docs
Geometry.mul_with_projection
Geometry.mul_return_type
```
