using Downloads

filename = "day600.0.hdf5"
if !isfile(filename)
    Downloads.download("https://caltech.box.com/shared/static/uy2l2prwzb4mik49aajhodhym5fb7d7l.hdf5", filename)
end

using ClimaCore
# read temperature data
reader = ClimaCore.InputOutput.HDF5Reader(filename)


temperature = ClimaCore.InputOutput.read_field(reader, "diagnostics/temperature")
temperature_col = ClimaCore.column(temperature, 1,1,1)
temperature_vec = vec(parent(temperature_col))

pressure = ClimaCore.InputOutput.read_field(reader, "diagnostics/pressure")
pressure_col = ClimaCore.column(pressure, 1,1,1)
pressure_vec = vec(parent(pressure_col))

z_vec = vec(parent(ClimaCore.Fields.coordinate_field(axes(temperature_col)).z))

# given a vector of pressures (what range?) find the temperatures at those levels
# - initially pick some conventient values (say 20)
# - ask around: what range of pressures, do they want log-scaling?
# - handle out-of-range value: have an option to "fill"
# - should we find z first, then interpolate?


# plotting packages:
# - Plots.jl (has multiple backends)
# - Makie.jl (GPU-accelerated plotting with 3D support)