"""
    PressureInterpolator

A interpolator for interpolating ClimaCore fields defined on a space whose
vertical direction is z to a space whose vertical direction is pressure.

Using this struct requires loading ClimaInterpolations.jl.

The approach for computing the pressure field is by:
1. Ensures monotonicity by applying column-wise cumulative maximum to the
   pressure field.
2. Performs vertical interpolation to the specified pressure levels using the
   monotonic pressure field.

!!! warning "No validation of pressure-height relationship"
    The implementation assumes pressure decreases monotonically with height. If
    the interpolated field appears unrealistic, check for instabilities or
    inversions in the pressure field.

!!! note "Boundary conditions for vertical interpolation"
    Vertical interpolation uses flat boundary conditions at the top and bottom
    of the atmosphere. Interpolated values at pressure levels outside the
    model's vertical range may be inaccurate.

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
    with max."""
    scratch_center_pressure_field::CENTER

    """A scratch face pressure field that is interpolated from
    scratch_center_pressure_field"""
    scratch_face_pressure_field::FACE

    """A space that is identical to the space that pfull_field is defined on,
    but the vertical direction is pressure"""
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
    grid = Grids.ExtrudedFiniteDifferenceGrid(
        space.grid.horizontal_grid,
        pfull_grid,
        space.grid.hypsography,
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
on a center space and `pressure_levels`, a vector of pressure levels to interpolate
to.

This mutates `pfull_field`.
"""
function PressureInterpolator(pfull_field::Fields.Field, pressure_levels)
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
`pressure_space`, a space with pressure as the vertical direction.
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
    scratch_phalf_field = intp_c2f.(pfull_field)
    pfull_mesh = pressure_space |> Spaces.vertical_topology |> Topologies.mesh
    pressure_levels = [point.p for point in pfull_mesh.faces]
    issorted(pressure_levels) || error("Pressure levels are not sorted")
    pressure_levels = typeofarray(pressure_levels)
    pfull_field = copy(pfull_field)
    _update!(pfull_field, scratch_center_pressure_field, scratch_phalf_field)
    return PressureInterpolator(
        pfull_field,
        scratch_center_pressure_field,
        scratch_phalf_field,
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
        scratch_phalf_field,
    )

A helper function to update the pressure fields, so that the pressures for each
column are monotonic.
"""
function _update!(
    pfull_field::Fields.Field,
    scratch_center_pressure_field,
    scratch_phalf_field,
)
    pfull_array = Fields.field2array(pfull_field)
    scratch_pfull_array = Fields.field2array(scratch_center_pressure_field)
    # Pressure is decreasing for increasing z
    reverse!(pfull_array, dims = 1)
    accumulate!(max, scratch_pfull_array, pfull_array, dims = 1)
    intp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. scratch_phalf_field = intp_c2f.(scratch_center_pressure_field)
    return nothing
end

function interpolate_pressure! end

function interpolate_pressure!! end

extension_fns = [
    :ClimaInterpolations => [
        :interpolate_pressure!,
        :interpolate_pressure!!,
    ],
]

"""
    is_pkg_loaded(pkg::Symbol)

Check if `pkg` is loaded or not.
"""
function is_pkg_loaded(pkg::Symbol)
    return any(k -> Symbol(k.name) == pkg, keys(Base.loaded_modules))
end

function __init__()
    # Register error hint if a package is not loaded
    if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(
            MethodError,
        ) do io, exc, _argtypes, _kwargs
            for (pkg, fns) in extension_fns
                if Symbol(exc.f) in fns && !is_pkg_loaded(pkg)
                    print(
                        io,
                        "\nImport ClimaInterpolations to enable `$(exc.f)`.";
                    )
                end
            end
        end
    end
end
