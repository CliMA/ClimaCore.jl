
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

FT = Float32

filename = "day600.0.hdf5"
if !isfile(filename)
    Downloads.download("https://caltech.box.com/shared/static/uy2l2prwzb4mik49aajhodhym5fb7d7l.hdf5", filename)
end
# read temperature data
reader = InputOutput.HDF5Reader(filename)

data_field = InputOutput.read_field(reader, "diagnostics/temperature")
pressure_field = InputOutput.read_field(reader, "diagnostics/pressure")
pressure_levels = reverse(sort(FT[2000, 4000, 6000, 8000, 8001]))


# plotting packages:
# - Plots.jl (has multiple backends)
# - Makie.jl (GPU-accelerated plotting with 3D support)

function interp_vertical(
    data_field::Fields.Field,
    pressure_field::Fields.Field,
    pressure_levels::Vector=FT[1000, 925, 850, 700, 600, 500, 300, 250, 200, 150, 100],
    fill_value=NaN
)
    # Sort pressure_levels monotonically decreasing:
    pressure_levels = reverse(sort(pressure_levels))
    # Construct output space
    data_space = axes(data_field)
    horizontal_space = Spaces.horizontal_space(data_space)
    z_min, z_max = extrema(pressure_levels)
    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(z_min),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, Geometry.ZPoint.(pressure_levels))
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
    pressure_space = Spaces.ExtrudedFiniteDifferenceSpace(horizontal_space, vert_face_space)
    out_field = Fields.Field(FT, pressure_space)
    data = parent(out_field)
    fill!(data, fill_value)

    Fields.bycolumn(axes(data_field)) do colidx
        println(colidx)
        interp_column!(out_field[colidx], data_field[colidx], pressure_field[colidx], pressure_levels)
    end

    return out_field
end

# TODO: Test out-of-bounds indexing, write unit tests

function interp_column!(out_col::Fields.Field, data_col::Fields.Field, pressure_col::Fields.Field, pressure_levels::Vector)
    # Setup data for loop
    out_col_values = Fields.field_values(out_col)
    data_col_values = Fields.field_values(data_col)
    z_vec = vec(parent(Fields.coordinate_field(axes(data_col)).z))
    pressure_vec = vec(parent(pressure_col))
    itp = interpolate(reverse(pressure_vec), reverse(z_vec), FiniteDifferenceMonotonicInterpolation())
    z_levels = itp.(pressure_levels)
    z_vec_ind = 1
    # println(z_levels)
    # println(pressure_levels)
    for (z_level_index, z_level) in enumerate(z_levels)
        while z_vec[z_vec_ind] < z_level && z_vec_ind < length(z_vec)
            # println(z_vec[z_vec_ind], " ", z_level)
            z_vec_ind += 1
        end
        data_below =  data_col_values[z_vec_ind-1]
        data_above = data_col_values[z_vec_ind]
        z_below = z_vec[z_vec_ind-1]
        z_above = z_vec[z_vec_ind]
        z_fractional_distance =  (z_above - z_level) / (z_above - z_below)
        out_col_values[z_level_index] = (data_above - data_below) * z_fractional_distance + data_below
        # println(z_vec[z_vec_ind-1], " < ",z_level , " < ", z_vec[z_vec_ind])
        # println(data_above, " > ",out_col_values[z_level_index], " > ", data_below)
    end
end

out_field = interp_vertical(data_field, pressure_field, pressure_levels)
