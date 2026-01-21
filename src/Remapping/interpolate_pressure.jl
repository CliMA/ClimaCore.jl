import ClimaInterpolations

"""
    PressureInterpolator

An interpolator for interpolating ClimaCore fields defined on a space whose
vertical coordinate is z to a space whose vertical coordinate is pressure.

Interpolating to pressure coordinates is done by:
1. Applying column-wise cumulative minimum to the pressure field which ensures
   monotonicity.
2. Interpolating to the specified pressure levels using the monotonic pressure
   field.

!!! warning "No validation of pressure-height relationship"
    The implementation assumes pressure decreases monotonically with height. If
    the interpolated field appears unrealistic, check for instabilities or
    inversions in the pressure field.

!!! note "Boundary conditions for vertical interpolation"
    By default, vertical interpolation uses constant boundary conditions at the
    top and bottom of the atmosphere. Interpolated values at pressure levels
    outside the model's vertical range may be inaccurate.

!!! note "Center space"
    The pressure field must be defined on a center space.
"""
struct PressureInterpolator{
    CENTER <: Fields.Field,
    FACE <: Fields.Field,
    SPACE <: Spaces.AbstractSpace,
    LEVELS,
}
    """A ClimaCore.Field representing pressure on center space. This field is
    defined on a space with height (z) as the vertical coordinate, not
    pressure."""
    pfull_field::CENTER

    """A scratch pressure field that stores the result of applying accumulate
    with min."""
    scratch_center_pressure_field::CENTER

    """A scratch face pressure field that is interpolated from
    scratch_center_pressure_field"""
    scratch_face_pressure_field::FACE

    """A space that is identical to the space that pfull_field is defined on,
    but the vertical coordinate is pressure"""
    pressure_space::SPACE

    """A 1D vector of pressure coordinates to interpolate onto for every
    column"""
    pressure_levels::LEVELS
end

"""
    construct_pressure_space(
        ::Type{FT},
        space::Union{
            Spaces.ExtrudedFiniteDifferenceSpace,
            Spaces.AbstractFiniteDifferenceSpace,
        },
        pressure_levels,
    )

Construct a new space for pressure-coordinate interpolation.

Given an input `space`, creates a new space where:
- The vertical coordinate type is `PPoint` (pressure) instead of `ZPoint`
  (height)
- The vertical staggering is `CellFace`
- All horizontal components (e.g. topology, quadrature, grid if they exist) are
  preserved from the input space
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
    construct_pfull_grid(::Type{FT}, pressure_levels, device) where {FT}

Construct a `Grids.FiniteDifferenceGrid` consisting of `PPoints`.
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
    PressureInterpolator(pfull_field::Fields.Field, pressure_levels)

Construct a `PressureInterpolator` from `pfull_field`, a pressure field defined
on a center space and `pressure_levels`, a vector of pressure levels to
interpolate to.

The pressure levels must be in ascending or descending order.
"""
function PressureInterpolator(pfull_field::Fields.Field, pressure_levels)
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
        pressure_space,
    )
end

"""
    PressureInterpolator(
        pfull_field::Fields.Field,
        pressure_space::Union{
            Spaces.AbstractFiniteDifferenceSpace,
            Spaces.ExtrudedFiniteDifferenceSpace,
        },
    )

Construct a `PressureInterpolator` from `pfull_field`, a pressure field, and
`pressure_space`, a space with pressure as the vertical coordinate.
"""
function PressureInterpolator(
    pfull_field::Fields.Field,
    pressure_space::Union{
        Spaces.AbstractFiniteDifferenceSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
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
    )
end

"""
    pfull_field(pfull_intp::PressureInterpolator)

Get the pressure field defined on the center space from `pfull_intp`.
"""
pfull_field(pfull_intp::PressureInterpolator) = pfull_intp.pfull_field

"""
    pressure_space(pfull_intp::PressureInterpolator)

Return the space where the points along the vertical are `PPoint`s.
"""
pressure_space(pfull_intp::PressureInterpolator) = pfull_intp.pressure_space

"""
    update!(pfull_intp::PressureInterpolator)

Update `pfull_intp` after the pressure field has been modified.

This should only be called once whenever the pressure field changes before
performing new interpolations.
"""
function update!(pfull_intp::PressureInterpolator)
    (; pfull_field, scratch_center_pressure_field, scratch_face_pressure_field) = pfull_intp
    _update!(pfull_field, scratch_center_pressure_field, scratch_face_pressure_field)
    return nothing
end

"""
    _update!(
        pfull_field::Fields.Field,
        scratch_center_pressure_field,
        scratch_face_pressure_field,
    )

A helper function to update the pressure fields, so that the pressures for each
column are monotonic.
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
    interpolate_pressure(
        field::Fields.Field,
        pfull_intp::PressureInterpolator;
        extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
    )

Vertically interpolate field onto a space identical to that of field, but with
pressure as the vertical coordinate and return the interpolated field.
"""
function interpolate_pressure(
    field::Fields.Field,
    pfull_intp::PressureInterpolator;
    extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
)
    (; pfull_field) = pfull_intp
    dest = fill(one(eltype(pfull_field)), pfull_intp.pressure_space)
    Remapping.interpolate_pressure!(dest, field, pfull_intp; extrapolate)
    return dest
end

"""
    interpolate_pressure!(
        dest::Fields.Field,
        field::Fields.Field,
        pfull_intp::PressureInterpolator;
        extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
    )

Vertically interpolate `field` onto `dest` and return `nothing`.

The vertical coordinate of the space of `dest` must be in pressure.
"""
function interpolate_pressure!(
    dest::Fields.Field,
    field::Fields.Field,
    pfull_intp::PressureInterpolator;
    extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
)
    (; scratch_center_pressure_field, scratch_face_pressure_field, pressure_levels) =
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
