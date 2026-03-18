# Geometry

```@meta
CurrentModule = ClimaCore
```

## Global Geometry

```@docs
Geometry.AbstractGlobalGeometry
Geometry.CartesianGlobalGeometry
```

## LocalGeometry

```@docs
Geometry.LocalGeometry
```

## Internals

```@docs
Geometry.Δz_metric_component
Geometry.:⊗
```

## Coordinates

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
