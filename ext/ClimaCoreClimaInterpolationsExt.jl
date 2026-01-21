module ClimaCoreClimaInterpolationsExt

import ClimaCore: Fields, Grids, Remapping
import ClimaCore.Remapping: PressureInterpolator
import ClimaCore.Fields: field2array
import ClimaInterpolations

"""
    interpolate_pressure!(field::Fields.Field, pfull_intp::PressureInterpolator)

Vertically interpolate field onto a space identical to that of field, but with
pressure as the vertical coordinate and returns the interpolated field.

This mutates `field`.
"""
function Remapping.interpolate_pressure!(
    field::Fields.Field,
    pfull_intp::PressureInterpolator,
)
    (; pfull_field) = pfull_intp
    dest = fill(one(eltype(pfull_field)), pfull_intp.pressure_space)
    Remapping.interpolate_pressure!!(dest, field, pfull_intp)
    return dest
end

"""
    interpolate_pressure!!(
        dest::Fields.Field,
        field::Fields.Field,
        pfull_intp::PressureInterpolator,
    )

Vertically interpolate `field` onto `dest` and return `nothing`.

The vertical direction of the space of `dest` must be in pressure coordinates.

This mutates both `dest` and `field`.
"""
function Remapping.interpolate_pressure!!(
    dest::Fields.Field,
    field::Fields.Field, # TODO: Dispatch off of the staggering of this
    pfull_intp::PressureInterpolator,
)
    (; scratch_center_pressure_field, scratch_face_pressure_field, pressure_levels) =
        pfull_intp
    scratch_pfull_array = if axes(field).staggering isa Grids.CellCenter
        field2array(scratch_center_pressure_field)
    else
        field2array(scratch_face_pressure_field)
    end
    field_array = field2array(field)
    dest = field2array(dest)
    reverse!(field_array, dims = 1)
    # Note that interpolate1d! still works even if there are repeated values in
    # the columns of pfull_array
    ClimaInterpolations.Interpolation1D.interpolate1d!(
        dest,
        scratch_pfull_array,
        pressure_levels,
        field_array,
        ClimaInterpolations.Interpolation1D.Linear(),
        ClimaInterpolations.Interpolation1D.Flat(),
    )
    return nothing
end

end
