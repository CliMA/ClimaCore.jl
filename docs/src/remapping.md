# Remapping to regular grids

`ClimaCore` horizontal domains are spectral elements. Points are not distributed
uniformly within an element, and elements are also not necessarily organized in
a simple way. For these reasons, remapping to regular grids becomes a
fundamental operations when inspecting the simulation output. In this section,
we describe the remappers currently available in `ClimaCore`.

Broadly speaking, we can classify remappers in two categories: non-conservative,
and conservative. Conservative remappers preserve areas (and masses) when going
from the spectral grid to Cartesian ones. Conservative remappers are non-local
operations (meaning that they require communication between different elements)
and are more expensive, so they are typically reserved to operations where
physical conservation is important (e.g., exchange between component models in a
coupled simulation). On the other hand, non-conservative remappers are local to
an element and faster to evaluate, which makes them suitable to operations like
diagnostics and plotting, where having perfect physical conservation is not as
important.

### Non-conservative remapping

Non-conservative remappers are fast and don't require communication, but they
are not as accurate as conservative remappers, especially with large elements
with sharp gradients. These remappers are optimally suited for diagnostics and
plots. 

# `Remapper`

The main remapper currently implemented utilizes a Lagrange interpolation with
the barycentric formula in [Berrut2004], equation (3.2), for the horizontal
interpolation. Vertical interpolation is linear except in the boundary elements
where it is 0th order.

## Quick start

Assuming you have a `ClimaCore` `Field` with name `field`, the simplest way to
interpolate into a uniform grid is with
```julia
julia> import ClimaCore.Remapping
julia> Remapping.interpolate(field)
```

This will return an Array with the `field` interpolated on some uniform grid
that is automatically determined based on the `Space` of definition of `field`.

To obtain such coordinates, you can call the
[`Remapping.default_target_hcoords`](@ref) and
[`Remapping.default_target_zcoords`](@ref) functions. These functions return an
Array with the coordinates over which interpolation will occur. These arrays are
of `Geometry.Point`s.

Multiple `fields` can be interpolated at the same time as long as they share the
underlying space.


[`Remapping.interpolate`](@ref) allocates new output arrays. As such, it is not
suitable for performance-critical applications. [`Remapping.interpolate!`](@ref)
performs interpolation in-place. Even better, it is to create a
[`Remapping.Remapper`](@ref) object and use it directly.

## More performance




```julia
Remapper(space, target_hcoords, target_zcoords, buffer_length = 1)
Remapper(space; target_hcoords, target_zcoords, buffer_length = 1)
Remapper(space, target_hcoords; buffer_length = 1)
Remapper(space, target_zcoords; buffer_length = 1)
```

Return a `Remapper` responsible for interpolating any `Field` defined on the
given `space` to the Cartesian product of `target_hcoords` with
`target_zcoords`.

`target_zcoords` can be `nothing` for interpolation on horizontal spaces. Similarly, `target_hcoords` can be `nothing` for interpolation on vertical spaces.

The `Remapper` is designed to not be tied to any particular `Field`. You can use the same `Remapper` for any `Field` as long as they are all defined on the same `topology`.

`Remapper` is the main argument to the `interpolate` function.

### Keyword arguments

`buffer_length` is size of the internal buffer in the Remapper to store intermediate values for interpolation. Effectively, this controls how many fields can be remapped simultaneously in `interpolate`. When more fields than `buffer_length` are passed, the remapper will batch the work in sizes of `buffer_length`.

## Interpolation

```julia
interpolate(remapper::Remapper, fields)
interpolate!(dest, remapper::Remapper, fields)
```

Interpolate the given `field`(s) as prescribed by `remapper`.

The optimal number of fields passed is the `buffer_length` of the `remapper`. If more fields are passed, the `remapper` will batch work with size up to its `buffer_length`.

This call mutates the internal (private) state of the `remapper`.


`interpolate!` writes the output to the given `dest`ination. `dest` is expected to be defined on the root process and to be `nothing` for the other processes.

**Note:** `interpolate` allocates new arrays and has some internal type-instability, `interpolate!` is non-allocating and type-stable.

When using `interpolate!`, the `dest`ination has to be the same array type as the device in use (e.g., `CuArray` for CUDA runs).

### Example

Given `field1`,`field2`, two `Field` defined on a cubed sphere.

```julia
longpts = range(-180.0, 180.0, 21)
latpts = range(-80.0, 80.0, 21)
zpts = range(0.0, 1000.0, 21)

hcoords = [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
zcoords = [Geometry.ZPoint(z) for z in zpts]

space = axes(field1)

remapper = Remapper(space, hcoords, zcoords)

int1 = interpolate(remapper, field1)
int2 = interpolate(remapper, field2)

# Or
int12 = interpolate(remapper, [field1, field2])
# With int1 = int12[1, :, :, :]
```

## Convenience Interpolation

```julia
interpolate(field::ClimaCore.Fields;
           hresolution = 180,
           resolution = 50,
           target_hcoords = default_target_hcoords(space; hresolution),
           target_zcoords = default_target_vcoords(space; vresolution)
           )
```

Interpolate the given fields on the Cartesian product of `target_hcoords` with `target_zcoords` (if not empty).

Coordinates have to be `ClimaCore.Geometry.Points`.

**Note:** do not use this method when performance is important. Instead, define a `Remapper` and call `interpolate(remapper, fields)`. Different `Field`s defined on the same `Space` can share a `Remapper`, so that interpolation can be optimized.

### Example

Given `field`, a `Field` defined on a cubed sphere.

By default, a target uniform grid is chosen (with resolution `hresolution` and `vresolution`), so remapping is simply

```julia
julia> interpolate(field, hcoords, zcoords)
```

Coordinates can be specified:

```julia
julia> longpts = range(-180.0, 180.0, 21)
julia> latpts = range(-80.0, 80.0, 21)
julia> zpts = range(0.0, 1000.0, 21)

julia> hcoords = [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
julia> zcoords = [Geometry.ZPoint(z) for z in zpts]

julia> interpolate(field, hcoords, zcoords)
```
```

### Conservative remapping with `TempestRemap`

This section hasn't been written yet. You can help by writing it.
