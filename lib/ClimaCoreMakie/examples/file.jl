using Downloads

filename = "day600.0.hdf5"
if !isfile(filename)
    Downloads.download(
        "https://caltech.box.com/shared/static/uy2l2prwzb4mik49aajhodhym5fb7d7l.hdf5",
        filename,
    )
end

using ClimaCore
# read temperature data
reader = ClimaCore.InputOutput.HDF5Reader(filename)
T = ClimaCore.InputOutput.read_field(reader, "diagnostics/temperature")

using Makie, GLMakie, ClimaCoreMakie

n = ClimaCore.Spaces.nlevels(axes(T))

begin
    fig = Figure()
    level_s = Slider(fig[2, 1], range = 1:n)
    # plot level at slider
    level_field = map(level_s.value) do n
        ClimaCore.level(T, n)
    end
    plot(fig[1, 1], level_field, axis = (show_axis = false,))
    fig
end
