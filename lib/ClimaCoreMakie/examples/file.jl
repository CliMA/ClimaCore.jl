using Downloads

filename = "day600.0.hdf5"
if !isfile(filename)
    Downloads.download("https://caltech.box.com/shared/static/uy2l2prwzb4mik49aajhodhym5fb7d7l.hdf5", filename)
end

using ClimaCore
# read temperature data
reader = ClimaCore.InputOutput.HDF5Reader(filename)
T = ClimaCore.InputOutput.read_field(reader, "diagnostics/temperature")

using Makie, GLMakie, ClimaCoreMakie

n = ClimaCore.Spaces.nlevels(axes(T))

# plot the lowest level (surface)
plot(ClimaCore.level(T,1))

# plot the mid level
plot(ClimaCore.level(T,n÷2))

# plot the highest level
plot(ClimaCore.level(T,n))
