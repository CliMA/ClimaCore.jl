using Downloads
using Makie, GLMakie, ClimaCoreMakie
using ClimaCore
using Makie: GLTriangleFace

filename = "day600.0.hdf5"
if !isfile(filename)
    Downloads.download("https://caltech.box.com/shared/static/uy2l2prwzb4mik49aajhodhym5fb7d7l.hdf5", filename)
end

# read temperature data
reader = ClimaCore.InputOutput.HDF5Reader(filename)
T = ClimaCore.InputOutput.read_field(reader, "diagnostics/temperature")

begin
    n = ClimaCore.Spaces.nlevels(axes(T))
    fig = Figure()
    level_s = Slider(fig[2, 1], range = 1:n)
    # plot level at slider
    level_field = map(level_s.value) do n
        ClimaCore.level(T, n)
    end
    plot(fig[1, 1], level_field, axis=(show_axis=false,))
    fig
end

begin
    fig = Figure()
    ax = GeoAxis(fig[1, 1])
    ax2 = LScene(fig[1, 2]; show_axis=false)
    colorrange = extrema(hcat(vec.(parent.(ClimaCore.level.((T,), 1:n)))...))
    level_s = Colorbar(fig[2, :], colorrange=colorrange, colormap=:balance, vertical = false)
    level_s = Slider(fig[3, :], range = 1:n)

    field = ClimaCore.level(T, n)
    space = axes(field)
    a, b, c = ClimaCore.Spaces.triangulate(space)
    triangles = GLTriangleFace.(a, b, c)
    cf = ClimaCore.Fields.coordinate_field(space)
    long, lat = vec.(parent.((cf.long, cf.lat)))
    vertices = Point2f.(long, lat)
    # plot level at slider
    level_field = map(level_s.value) do n
        ClimaCore.level(T, n)
    end

    scalars = map(level_field) do field
        Float32.(vec(parent(field)))
    end
    mesh!(ax, vertices, triangles; color=scalars, shading=false, colormap=:balance, colorrange=colorrange)
    plot!(ax2, level_field, colorrange=colorrange)

    fig
end
