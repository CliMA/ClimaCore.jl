import ClimaInterpolations

"""
    PressureInterpolator

Interpolate fields from a space whose vertical coordinate is height `z` to a space whose
vertical coordinate is pressure.

Construct one with `PressureInterpolator(pfull_field, pressure_levels)` and apply it with
[`interpolate_pressure`](@ref) or [`interpolate_pressure!`](@ref). After the pressure field
changes, call [`update!`](@ref) before interpolating again.

Interpolation proceeds in two steps:

 1. Apply a column-wise cumulative minimum to the pressure field, so that pressure is
    monotone in each column.
 2. Interpolate linearly in the monotone pressure to the target `pressure_levels`.

# Fields

  - `pfull_field`: pressure on a center space whose vertical coordinate is height.
  - `scratch_center_pressure_field`: column-wise cumulative minimum of `pfull_field`.
  - `scratch_face_pressure_field`: `scratch_center_pressure_field` interpolated to faces.
  - `pressure_space`: the space of `pfull_field` with pressure as the vertical coordinate.
  - `pressure_levels`: the target pressure levels, in decreasing order, shared by all
    columns.
  - `extrapolate`: `ClimaInterpolations.Interpolation1D.Extrapolate1D` rule for pressure
    levels outside a column's pressure range.

!!! warning "No validation of the pressure-height relationship"

    The implementation assumes that pressure decreases monotonically with height. Where
    the cumulative minimum flattens the pressure profile, the interpolated field is
    unreliable; check for instabilities or inversions in the pressure field.

!!! note "Boundary conditions"

    By default, values at pressure levels outside a column's pressure range are
    extrapolated as constants (`Flat()`) and may be inaccurate.

!!! note "Center space"

    `pfull_field` must be defined on a center space.
"""
struct PressureInterpolator{
    CENTER <: Fields.Field,
    FACE <: Fields.Field,
    SPACE <: Spaces.AbstractSpace,
    LEVELS,
    EXTRAPOLATE <: ClimaInterpolations.Interpolation1D.Extrapolate1D,
}
    pfull_field::CENTER
    scratch_center_pressure_field::CENTER
    scratch_face_pressure_field::FACE
    pressure_space::SPACE
    pressure_levels::LEVELS
    extrapolate::EXTRAPOLATE
end

"""
    construct_pressure_space(::Type{FT}, space, pressure_levels)

Return a space like `space` but with pressure as the vertical coordinate.

The vertical grid is a `FiniteDifferenceGrid` of `PPoint{FT}`s with faces at
`pressure_levels`, and the staggering is `CellFace`. For an
`ExtrudedFiniteDifferenceSpace`, the horizontal grid and global geometry are kept and the
hypsography is `Flat`; for an `AbstractFiniteDifferenceSpace`, the result is a
`FiniteDifferenceSpace`.
"""
function construct_pressure_space(
    ::Type{FT},
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    pressure_levels,
) where {FT}
    device = ClimaComms.device(space)
    pfull_grid = construct_pfull_grid(FT, pressure_levels, device)
    # Since fields constructed from the pressure space is a container for
    # values, the hypsography does not need to be the same as the hypsography of
    # the space passed in
    grid = Grids.ExtrudedFiniteDifferenceGrid(
        space.grid.horizontal_grid,
        pfull_grid,
        Grids.Flat(),
        space.grid.global_geometry,
    )
    pressure_space = Spaces.ExtrudedFiniteDifferenceSpace(
        grid,
        Spaces.CellFace(),
    )
    return pressure_space
end

function construct_pressure_space(
    ::Type{FT},
    space::Spaces.AbstractFiniteDifferenceSpace,
    pressure_levels,
) where {FT}
    device = ClimaComms.device(space)
    pfull_grid = construct_pfull_grid(FT, pressure_levels, device)
    pressure_space = Spaces.FiniteDifferenceSpace(pfull_grid, Spaces.CellFace())
    return pressure_space
end

"""
    construct_pfull_grid(::Type{FT}, pressure_levels, device)

Return a `Grids.FiniteDifferenceGrid` whose faces are `PPoint{FT}`s at `pressure_levels`,
on a `SingletonCommsContext` for `device`.

`pressure_levels` must be increasing. The boundary at the minimum pressure is named `:top`
and the boundary at the maximum pressure `:bottom`.
"""
function construct_pfull_grid(::Type{FT}, pressure_levels, device) where {FT}
    pfull_boundary_names = (:top, :bottom)
    # This needs to be increasing because of the Remapping object does not
    # work when pressures are decreasing
    pfull_domain = Domains.IntervalDomain(
        Geometry.PPoint{FT}(minimum(pressure_levels)),
        Geometry.PPoint{FT}(maximum(pressure_levels));
        boundary_names = pfull_boundary_names,
    )
    pfull_mesh = Meshes.IntervalMesh(pfull_domain, Geometry.PPoint.(pressure_levels))
    pfull_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        pfull_mesh,
    )
    pfull_grid = Grids.FiniteDifferenceGrid(pfull_topology)
    return pfull_grid
end

"""
    PressureInterpolator(
        pfull_field::Fields.Field,
        pressure_levels;
        extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
    )

Construct a `PressureInterpolator` from `pfull_field`, the pressure on a center space, and
`pressure_levels`, the vector of pressure levels to interpolate to.

`pressure_levels` must be sorted, ascending or descending; they are converted to the
element type of `pfull_field`. `extrapolate` sets the treatment of levels outside a
column's pressure range; the default `Flat()` extrapolates constants.
"""
function PressureInterpolator(
    pfull_field::Fields.Field,
    pressure_levels;
    extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
)
    if issorted(pressure_levels, rev = true)
        pressure_levels = sort(pressure_levels)
    end
    issorted(pressure_levels) || error("Pressure levels are not sorted")
    FT = eltype(pfull_field)
    pressure_levels = FT.(pressure_levels)

    space = axes(pfull_field)
    pressure_space = construct_pressure_space(FT, space, pressure_levels)
    return PressureInterpolator(
        pfull_field,
        pressure_space;
        extrapolate,
    )
end

"""
    PressureInterpolator(
        pfull_field::Fields.Field,
        pressure_space;
        extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
    )

Construct a `PressureInterpolator` from `pfull_field`, the pressure on a center space, and
`pressure_space`, a space whose vertical coordinate is `PPoint` (as built by
`construct_pressure_space`). The face coordinates of `pressure_space` are the target
pressure levels.

`extrapolate` sets the treatment of levels outside a column's pressure range; the default
`Flat()` extrapolates constants.
"""
function PressureInterpolator(
    pfull_field::Fields.Field,
    pressure_space::Union{
        Spaces.AbstractFiniteDifferenceSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    };
    extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
)
    axes(pfull_field).staggering isa Grids.CellCenter || error("The staggering of the
    pressure field must be cell center")
    vertical_domain =
        Spaces.vertical_topology(pressure_space) |> Topologies.mesh |> Meshes.domain
    vertical_domain.coord_max isa Geometry.PPoint ||
        error("Vertical domain of space must have PPoint")
    typeofarray = ClimaComms.array_type(pfull_field)
    scratch_center_pressure_field = copy(pfull_field)
    intp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    scratch_face_pressure_field = intp_c2f.(pfull_field)
    pfull_mesh = pressure_space |> Spaces.vertical_topology |> Topologies.mesh
    pressure_levels = [point.p for point in Iterators.reverse(pfull_mesh.faces)]
    issorted(pressure_levels, rev = true) || error("Pressure levels are not sorted")
    pressure_levels = typeofarray(pressure_levels)
    _update!(pfull_field, scratch_center_pressure_field, scratch_face_pressure_field)
    return PressureInterpolator(
        pfull_field,
        scratch_center_pressure_field,
        scratch_face_pressure_field,
        pressure_space,
        pressure_levels,
        extrapolate,
    )
end

"""
    pfull_field(pfull_intp::PressureInterpolator)

Return the pressure field on the center space stored in `pfull_intp`.
"""
pfull_field(pfull_intp::PressureInterpolator) = pfull_intp.pfull_field

"""
    pressure_space(pfull_intp::PressureInterpolator)

Return the space of `pfull_intp` whose vertical coordinates are `PPoint`s.
"""
pressure_space(pfull_intp::PressureInterpolator) = pfull_intp.pressure_space

"""
    update!(pfull_intp::PressureInterpolator)

Recompute the monotone scratch pressure fields of `pfull_intp` from its pressure field.

Call this once after the pressure field changes and before interpolating again.
"""
function update!(pfull_intp::PressureInterpolator)
    (; pfull_field, scratch_center_pressure_field, scratch_face_pressure_field) = pfull_intp
    _update!(pfull_field, scratch_center_pressure_field, scratch_face_pressure_field)
    return nothing
end

"""
    _update!(pfull_field, scratch_center_pressure_field, scratch_face_pressure_field)

Fill `scratch_center_pressure_field` with the column-wise cumulative minimum (from the
bottom up) of `pfull_field`, and `scratch_face_pressure_field` with its interpolation to
faces, extrapolating at the top and bottom.

Called from [`update!`](@ref) and the `PressureInterpolator` constructor.
"""
function _update!(
    pfull_field::Fields.Field,
    scratch_center_pressure_field,
    scratch_face_pressure_field,
)
    pfull_array = Fields.field2array(pfull_field)
    scratch_pfull_array = Fields.field2array(scratch_center_pressure_field)
    # Pressure is decreasing for increasing z
    accumulate!(min, scratch_pfull_array, pfull_array, dims = 1)
    intp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. scratch_face_pressure_field = intp_c2f.(scratch_center_pressure_field)
    return nothing
end

"""
    interpolate_pressure(field::Fields.Field, pfull_intp::PressureInterpolator)

Interpolate `field` vertically onto the pressure space of `pfull_intp` and return the
result as a new `Field`.

See [`interpolate_pressure!`](@ref) for the in-place version.
"""
function interpolate_pressure(
    field::Fields.Field,
    pfull_intp::PressureInterpolator,
)
    (; pfull_field) = pfull_intp
    dest = fill(one(eltype(pfull_field)), pfull_intp.pressure_space)
    interpolate_pressure!(dest, field, pfull_intp)
    return dest
end

"""
    interpolate_pressure!(
        dest::Fields.Field,
        field::Fields.Field,
        pfull_intp::PressureInterpolator,
    )

Interpolate `field` vertically onto `dest`, a `Field` on the pressure space of
`pfull_intp`, and return `nothing`.

`field` may live on a center or a face space; the matching scratch pressure field of
`pfull_intp` serves as the source coordinate. Interpolation is linear in pressure, with the
extrapolation rule of `pfull_intp` outside a column's pressure range.
"""
function interpolate_pressure!(
    dest::Fields.Field,
    field::Fields.Field,
    pfull_intp::PressureInterpolator,
)
    (;
        scratch_center_pressure_field,
        scratch_face_pressure_field,
        pressure_levels,
        extrapolate,
    ) =
        pfull_intp
    scratch_pfull_array = if axes(field).staggering isa Grids.CellCenter
        Fields.field2array(scratch_center_pressure_field)
    else
        Fields.field2array(scratch_face_pressure_field)
    end
    field_array = Fields.field2array(field)
    dest_array = Fields.field2array(dest)
    # Note that interpolate1d! still works even if there are repeated values in
    # the columns of pfull_array
    ClimaInterpolations.Interpolation1D.interpolate1d!(
        dest_array,
        scratch_pfull_array,
        pressure_levels,
        field_array,
        ClimaInterpolations.Interpolation1D.Linear(),
        extrapolate,
        reverse = true,
    )
    reverse!(dest_array, dims = 1)
    return nothing
end
