
import ClimaCore:
    InputOutput,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    column

using Interpolations
using Plots
using Downloads
using Test


# plotting packages:
# - Plots.jl (has multiple backends)
# - Makie.jl (GPU-accelerated plotting with 3D support)
"""
Takes in two Fields from the same dataset (one pressure and one variable, e.g. temperature) 
and a vec of pressure values. Each pressure value is interpolated to a coordinate and to 
the variable in order to obtain the variable at each pressure value for each coordinate in the Space.
"""
function interp_vertical(
    data_field::Fields.Field,
    pressure_field::Fields.Field,
    pressure_levels::Vector = [
        1000,
        925,
        850,
        700,
        600,
        500,
        300,
        250,
        200,
        150,
        100,
    ],
    fill_value = NaN,
)
    # Preprocess pressure_levels
    FT = eltype(data_field)
    pressure_levels = reverse(sort(map(x -> FT(x), pressure_levels)))
    # Construct output space
    data_space = axes(data_field)
    horizontal_space = Spaces.horizontal_space(data_space)
    p_min, p_max = extrema(pressure_levels)
    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(p_min),
        Geometry.ZPoint{FT}(p_max);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, Geometry.ZPoint.(pressure_levels))
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
    pressure_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horizontal_space, vert_face_space)

    out_field = Fields.Field(FT, pressure_space)
    data = parent(out_field)
    fill!(data, fill_value)

    Fields.bycolumn(axes(data_field)) do colidx
        interp_column!(
            out_field[colidx],
            data_field[colidx],
            pressure_field[colidx],
            pressure_levels,
        )
    end
    return out_field
end

function interp_column!(
    out_col::Fields.Field,
    data_col::Fields.Field,
    pressure_col::Fields.Field,
    pressure_levels::Vector,
)
    # Setup data for loop
    out_col_values = Fields.field_values(out_col)
    data_col_values = Fields.field_values(data_col)
    z_vec = vec(parent(Fields.coordinate_field(axes(pressure_col)).z))
    pressure_vec = vec(parent(pressure_col))
    # Assume pressure is sorted monotonically decreasing, so reverse:
    itp = interpolate(
        reverse(pressure_vec),
        reverse(z_vec),
        FiniteDifferenceMonotonicInterpolation(),
    )
    z_i = 1
    for (p_i, p_level) in enumerate(pressure_levels)
        (min_p, max_p) = extrema(pressure_col)
        if p_level >= min_p && p_level <= max_p
            z_level = itp(p_level)
            while z_vec[z_i] < z_level && z_i < length(z_vec)
                z_i += 1
            end
            if z_i > 1
                data_below = data_col_values[z_i - 1]
                data_above = data_col_values[z_i]
                z_below = z_vec[z_i - 1]
                z_above = z_vec[z_i]
                fractional_dist = (z_above - z_level) / (z_above - z_below)
                out_col_values[p_i] =
                    (data_above - data_below) * fractional_dist + data_below
            else
                # Issue: sometimes interpolates to slightly differing values
                # @assert z_vec[z_i] == z_level
                # Must be minimal height
                out_col_values[p_i] = data_col_values[z_i]
            end
            # println(p_level, "--> p interpolated to z --> ", z_level)
            # println(z_vec[z_i], "--> z-data to p --> ", Fields.field_values(pressure_col)[z_i])
        end
    end
end

FT = Float32
filename = "day600.0.hdf5"
if !isfile(filename)
    Downloads.download(
        "https://caltech.box.com/shared/static/uy2l2prwzb4mik49aajhodhym5fb7d7l.hdf5",
        filename,
    )
end
# read temperature data
reader = InputOutput.HDF5Reader(filename)
data_field = InputOutput.read_field(reader, "diagnostics/temperature")
pressure_field = InputOutput.read_field(reader, "diagnostics/pressure")


# Testing:
# check small output levels - what does this mean? 
# edge cases
# TODO: Test single-element list
# TODO: # interpolation at level edge - that it’s the same

@testset "Z -> Pressure Space Interpolation" begin


    # Test out-of-bounds 
    allNaN(field::Fields.Field) = all(isnan, parent(field))

    pressure_levels = [1, 2, 3]
    out_field = interp_vertical(data_field, pressure_field, pressure_levels)
    @test allNaN(out_field)

    pressure_levels = [10^12, 10^11]
    out_field = interp_vertical(data_field, pressure_field, pressure_levels)
    @test allNaN(out_field)

    pressure_levels = [1, 6000, 10^12]
    out_values =
        parent(interp_vertical(data_field, pressure_field, pressure_levels))
    # Only value from 6000 should be filled
    @test all(
        i % 3 == 2 ? !isnan(out_values[i]) : isnan(out_values[i]) for
        i in 1:length(out_values)
    )


end
