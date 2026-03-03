# Remapping to regular grids

`ClimaCore` horizontal domains are spectral elements. Points are not distributed
uniformly within an element, and elements are also not necessarily organized in
a simple way. For these reasons, remapping to regular grids becomes a
fundamental operations when inspecting the simulation output. In this section,
we describe the remappers currently available in `ClimaCore`.

Broadly speaking, we can classify remappers in two categories: conservative, and
non-conservative. Conservative remappers preserve areas (and masses) when
interpolating from the spectral grid to Cartesian ones. Conservative remappers
are non-local operations (meaning that they require communication between
different elements) and are more expensive, so they are typically reserved to
operations where physical conservation is important (e.g., exchange between
component models in a coupled simulation). On the other hand, non-conservative
remappers are local to an element and faster to evaluate, which makes them
suitable to operations like diagnostics and plotting, where having perfect
physical conservation is not as important.

## Non-conservative remapping

Non-conservative remappers are fast and do not require communication, but they
are not as accurate as conservative remappers, especially with large elements
with sharp gradients. These remappers are better suited for diagnostics and
plots.

The main non-conservative remapper currently implemented utilizes a Lagrange
interpolation with the barycentric formula in [Berrut2004], equation (3.2), for
the horizontal interpolation. Vertical interpolation is linear except in the
boundary elements where it is 0th order.

### Quick start

Assuming you have a `ClimaCore` `Field` with name `field`, the simplest way to
interpolate onto a uniform grid is with
```julia
julia> import ClimaCore.Remapping
julia> Remapping.interpolate(field)
```

This will return an `Array` (or a `CuArray`) with the `field` interpolated on
some uniform grid that is automatically determined based on the `Space` of the
given `field`. To obtain such coordinates, you can call the
`Remapping.default_target_hcoords` and `Remapping.default_target_zcoords`
functions. These functions return an `Array` with the coordinates over which
interpolation will occur. These arrays are of type `Geometry.Point`s.

By default, vertical interpolation is off (field evaluated on levels). Horizontal
interpolation: `SpectralElementRemapping()` (default; uses spectral element quadrature weights) or `BilinearRemapping()`:

```julia
interpolated_array = Remapping.interpolate(field; horizontal_method = Remapping.BilinearRemapping())
```

`ClimaCore.Remapping.interpolate` allocates new output arrays. As such, it is
not suitable for performance-critical applications.
`ClimaCore.Remapping.interpolate!` performs interpolation in-place. When using
the in-place version`, the `dest`ination has to have the same array type as the
device in use (e.g., `CuArray` for CUDA runs) and has to be `nothing` for
non-root processes. For performance-critical applications, it is preferable to a
`ClimaCore.Remapping.Remapper` and use it directly (see next Section).

#### Example

Given `field`, a `Field` defined on a cubed sphere.

By default, a target uniform grid is chosen (with resolution `hresolution` and
`vresolution`), so remapping is
```julia
interpolated_array = interpolate(field, hcoords, zcoords)
```
Coordinates can be specified:

```julia
longpts = range(-180.0, 180.0, 21)
latpts = range(-80.0, 80.0, 21)
zpts = range(0.0, 1000.0, 21)

hcoords = [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
zcoords = [Geometry.ZPoint(z) for z in zpts]

interpolated_array = interpolate(field, hcoords, zcoords)
# Or, to use bilinear remapping without spectral element weighting:
# interpolate(field, hcoords, zcoords; horizontal_method = Remapping.BilinearRemapping())
```
The output is defined on the Cartesian product of `hcoords` with `zcoords`.

If the default target coordinates are being used, it is possible to broadcast
`ClimaCore.Geometry.components` to extract them as a vector of tuples (and then
broadcast `getindex` to extract the respective coordinates as vectors).

This also provides the simplest way to plot a `Field`. Suppose `field` is a 2D `Field`:
```julia
using CairoMakie
heatmap(ClimaCore.Remapping.interpolate(field))
```

### Remapping methods: Bilinear vs SpectralElementRemapping

Two horizontal remapping methods are available:

- **`SpectralElementRemapping()`** (default): Uses spectral element quadrature weights for high-order polynomial interpolation. More accurate for smooth fields but can produce overshoots/undershoots near discontinuities.
- **`BilinearRemapping()`**: Uses bilinear interpolation on the 2×2 GLL cell containing each target point. More conservative (bounds-preserving) but lower-order accuracy.

Both methods can be used with `interpolate_array` or `Remapper`:

```julia
using ClimaCore.Remapping: SpectralElementRemapping, BilinearRemapping

# Use spectral remapping (default)
interpolated = Remapping.interpolate_array(field, xpts, ypts)

# Use bilinear remapping
interpolated = Remapping.interpolate_array(
    field, xpts, ypts; horizontal_method = BilinearRemapping()
)

# With Remapper
remapper = Remapper(space; target_hcoords, horizontal_method = BilinearRemapping())
```

#### Slotted-cylinder example (demo of horizontal remapping types)

```@example remap_visualization
using ClimaComms
using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Remapping, Quadratures
using CairoMakie

device = ClimaComms.CPUSingleThreaded()
nelements_horz = 6
Nq = 4
n_interp = 24

# Simple test field: disk with slot (discontinuous)
slot_radius = 0.15
slot_cx, slot_cy = 0.5, 0.5
slot_half_width = 0.025
slot_y_hi = slot_cy + slot_radius

horzdomain = Domains.RectangleDomain(
    Geometry.XPoint(0.0) .. Geometry.XPoint(1.0),
    Geometry.YPoint(0.0) .. Geometry.YPoint(1.0),
    x1periodic = true, x2periodic = true,
)

quad = Quadratures.GLL{Nq}()
horzmesh = Meshes.RectilinearMesh(horzdomain, nelements_horz, nelements_horz)
horztopology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(device), horzmesh)
space = Spaces.SpectralElementSpace2D(horztopology, quad)

coords = Fields.coordinate_field(space)
function slotted_cylinder(x, y)
    in_disk = (x - slot_cx)^2 + (y - slot_cy)^2 <= slot_radius^2
    in_slot = (abs(x - slot_cx) <= slot_half_width) && (y >= slot_cy) && (y <= slot_y_hi)
    return (in_disk && !in_slot) ? 1.0 : 0.0
end
field = @. slotted_cylinder(coords.x, coords.y)
Spaces.weighted_dss!(field)

xpts = range(Geometry.XPoint(0.0), Geometry.XPoint(1.0), length = n_interp)
ypts = range(Geometry.YPoint(0.0), Geometry.YPoint(1.0), length = n_interp)

# Compare both methods
interp_bilinear = Remapping.interpolate_array(
    field, xpts, ypts; horizontal_method = Remapping.BilinearRemapping(),
)
interp_spectral = Remapping.interpolate_array(
    field, xpts, ypts; horizontal_method = Remapping.SpectralElementRemapping(),
)

# Error (bilinear − spectral): highlights where the methods differ
err_bilinear_spectral = interp_bilinear .- interp_spectral

# Raw data at GLL nodes (source field before interpolation)
x_se = Float64[]
y_se = Float64[]
vals_se = Float64[]
Fields.byslab(space) do slabidx
    x_data = parent(Fields.slab(coords.x, slabidx))
    y_data = parent(Fields.slab(coords.y, slabidx))
    f_data = parent(Fields.slab(field, slabidx))
    for j in 1:Nq, i in 1:Nq
        push!(x_se, x_data[i, j])
        push!(y_se, y_data[i, j])
        push!(vals_se, f_data[i, j])
    end
end

x_plot = [p.x for p in xpts]
y_plot = [p.y for p in ypts]

fig = Figure(size = (1200, 700))
ax1 = Axis(fig[1, 1], title = "Bilinear", xlabel = "x", ylabel = "y")
hm1 = heatmap!(
    ax1, x_plot, y_plot, interp_bilinear';
    colorrange = (0, 1), colormap = :viridis,
    lowclip = :orange, highclip = :red,
)
Colorbar(fig[1, 2], hm1; label = "value")

ax2 = Axis(fig[1, 3], title = "Spectral", xlabel = "x", ylabel = "y")
hm2 = heatmap!(
    ax2, x_plot, y_plot, interp_spectral';
    colorrange = (0, 1), colormap = :viridis,
    lowclip = :orange, highclip = :red,
)
Colorbar(fig[1, 4], hm2; label = "value")

ax3 = Axis(fig[1, 5], title = "Error (bilinear − spectral)", xlabel = "x", ylabel = "y")
erange = extrema(err_bilinear_spectral)
hm3 = heatmap!(
    ax3, x_plot, y_plot, err_bilinear_spectral';
    colorrange = erange, colormap = :RdBu,
)
Colorbar(fig[1, 6], hm3; label = "error")

# Row 2: raw spectral element grid (exact values at GLL nodes)
# Swap (y_se, x_se) so orientation matches heatmaps (slab i,j vs display x,y convention)
ax_se = Axis(
    fig[2, 1],
    title = "Raw spectral element grid (GLL nodes)",
    xlabel = "x",
    ylabel = "y",
)
sc_se = scatter!(
    ax_se, y_se, x_se;
    color = vals_se,
    colorrange = (0, 1),
    colormap = :viridis,
    lowclip = :orange,
    highclip = :red,
    markersize = 8,
)
boundary_pos = (0:nelements_horz) ./ nelements_horz
vlines!(ax_se, boundary_pos; color = :pink, linewidth = 2)
hlines!(ax_se, boundary_pos; color = :pink, linewidth = 2)
limits!(ax_se, 0, 1, 0, 1)
Colorbar(fig[2, 2], sc_se; label = "value")

fig
```

Row 1: heatmaps use **orange** for undershoots (&lt; 0) and **red** for overshoots (&gt; 1). The spectral method produces overshoots/undershoots near the discontinuity; bilinear stays in [0, 1]. The error panel (bilinear − spectral) shows where the two methods differ. Row 2: raw field values at the GLL nodes (the source data); pink lines show element boundaries.

### The `Remapper` object

A `Remapping.Remapper` is an object that is tied to a specified `Space` and can
interpolate scalar `Field`s defined on that space onto a predefined target grid.
The grid does not have to be regular, but it has to be defined as a Cartesian
product between some horizontal and vertical coordinates (meaning, for each
horizontal point, there is a fixed column of vertical coordinates).

Let us create our first remapper, assuming we have `space` defined on the
surface of the sphere
```julia
import ClimaCore.Geometry: LatLongPoint, ZPoint
import ClimaCore.Remapping: Remapper

hcoords = [Geometry.LatLongPoint(lat, long) for long in -180.:180., lat in -90.:90.]
remapper = Remapper(space, target_hcoords)
```
This `remapper` object knows can interpolate `Field`s defined on `space` with
the same `interpolate` and `interpolate!` functions.
```julia
import ClimaCore.Fields: coordinate_field
import ClimaCore.Remapping: interpolate, interpolate!

example_field = coordinate_field(space)
interpolated_array = interpolate(remapper, example_field)

# Interpolate in place
interpolate!(interpolated_array, remapper, example_field)
```

Multiple fields defined on the same space can be interpolate at the same time
```julia
example_field2 = cosd.(example_field)
interpolated_arrays = interpolate(remapper, [example_field, example_field2])
```

When interpolating multiple fields, greater performance can be achieved by
creating the `Remapper` with a larger internal buffer to store intermediate
values for interpolation. Effectively, this controls how many fields can be
remapped simultaneously in `interpolate`. When more fields than `buffer_length`
are passed, the remapper will batch the work in sizes of `buffer_length`. The
optimal number of fields passed is the `buffer_length` of the `remapper`. If
more fields are passed, the `remapper` will batch work with size up to its
`buffer_length`.

#### Example

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

## Conservative remapping with `TempestRemap`

This section hasn't been written yet. You can help by writing it.

TODO: finish writing this section.

# Interpolating to pressure coordinates

In addition to the `Remapper`, you can also interpolate to pressure coordinates.
This can be used with the `Remapper` to remap a `ClimaCore` `Field` to a regular
grid, where the vertical is pressure.

`ClimaCore` provides the `PressureInterpolator` for efficient vertical
interpolation from height to pressure coordinates.

## Quick start

The simplest way to interpolate a field to pressure coordinates:

```julia
import ClimaCore: Remapping, Fields
using ClimaInterpolations

# Define target pressure levels (in ascending order)
pressure_levels = 100.0 .* [100.0, 250.0, 500.0, 850.0, 1000.0]

# Create the interpolator
# pressure_field is a pressure field on center space
pressure_intp = Remapping.PressureInterpolator(pressure_field, pressure_levels)

# Interpolate the field to pressure coordinates
field_on_pressure_space = Remapping.interpolate_pressure(field, pressure_intp)

# Get the pressure field and space with pressure as the vertical
p_field = Remapping.pfull_field(pressure_intp)
p_space = Remapping.pressure_space(pressure_intp)

# If the pressure field changes, then you need to call update!
Remapping.update!(pressure_intp)

# This mutates both field_on_pressure_space and the interpolation done
# in-place
field_on_pressure_space =
    Remapping.interpolate_pressure!(field_on_pressure_space, field, pressure_intp)
```

The result `field_on_pressure_space` is defined on a new space where the
vertical coordinate is pressure rather than height.

## How it works

The `PressureInterpolator` performs the following steps:

1. **Ensure monotonicity**: Applies a cumulative minimum along each column to
   ensure pressure decreases monotonically with height.
2. **Vertical interpolation**: Interpolates field values to the specified
   pressure coordinates using linear interpolation with constant boundary
   conditions.

!!! warning "Pressure-height relationship"
    The implementation assumes pressure decreases monotonically with height. If
    the interpolated field appears unrealistic, check for instabilities or
    inversions in your pressure field.

!!! note "Boundary conditions"
    By default, vertical interpolation uses constant boundary conditions at the
    top and bottom of the atmosphere. Interpolated values at pressure levels
    outside the model's vertical range may be inaccurate.

## Space and staggering requirements

!!! note "Space compatibility"
    The pressure field and the field being interpolated must be defined on the
    same space with the same vertical staggering (`CellCenter` or `CellFace`).
    The pressure field must use `CellCenter` staggering.

The `PressureInterpolator` works with:
- `ExtrudedFiniteDifferenceSpace` - 3D spaces (e.g., cubed sphere with vertical
  levels)
- `FiniteDifferenceSpace` - 1D column spaces

Interpolating fields on center and face spaces are supported, but the pressure
field itself must always be on a center space.
