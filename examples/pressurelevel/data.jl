
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
    out_vec = Fields.field_values(out_col)
    data_vec = Fields.field_values(data_col)
    z_vec = vec(parent(Fields.coordinate_field(axes(pressure_col)).z))
    pressure_vec = [parent(pressure_col)[i] for i in 1:length(parent(pressure_col))]
    for (p_i, p_level) in enumerate(pressure_levels)
        (min_p, max_p) = extrema(pressure_col)
        if min_p <= p_level <= max_p
            j = searchsortedfirst(pressure_vec, p_level; rev=true)
            if j > 1
                frac_dist = (pressure_vec[j] - p_level) / (pressure_vec[j] - pressure_vec[j-1])
                out_vec[p_i] = (data_vec[j] - data_vec[j-1]) * frac_dist + data_vec[j]
            else
                # Must be minimal height
                out_vec[p_i] = data_vec[j]
            end
            # println(p_level, " @index ", p_i, " --> ", pressure_vec[j], " --> ", out_vec[p_i])
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

    pressure_levels = [1, 10^12, 6000]
    out_values =
        parent(interp_vertical(data_field, pressure_field, pressure_levels))
    # Only value from 6000 should be filled
    @test all(
        i % 3 == 2 ? !isnan(out_values[i]) : isnan(out_values[i]) for
        i in 1:length(out_values)
    )

    # Test repeating values
    pressure_levels = [4000, 4000]
    out_values =
        parent(interp_vertical(data_field, pressure_field, pressure_levels))
    @test all(
        i % 2 == 0 ? out_values[i]==out_values[i-1] : true for
        i in 1:length(out_values)
    )


end

# # Test bottom and top extremes
# pressure_levels = [p for p in extrema(pressure_field[Fields.ColumnIndex((1,1),1)])]
# pcol = pressure_field[Fields.ColumnIndex((1,1),1)]
# interp_vertical(data_field, pcol, pressure_levels)


# # Test all pressure levels in the column
# pcol = pressure_field[Fields.ColumnIndex((1,1),1)]
# pcol_values = parent(pcol)
# pressure_levels = [pcol_values[i] for i in 1:length(pcol_values)]
# interp_vertical(data_field, pcol, pressure_levels)