using Downloads
using Makie, GLMakie, ClimaCoreMakie
using ClimaCore
using Makie: GLTriangleFace
using GeoMakie

filename = "day600.0.hdf5"
if !isfile(filename)
    Downloads.download(
        "https://caltech.box.com/shared/static/uy2l2prwzb4mik49aajhodhym5fb7d7l.hdf5",
        filename,
    )
end

# read temperature data
reader = ClimaCore.InputOutput.HDF5Reader(filename)
diagnostics = ClimaCore.InputOutput.read_field(reader, "diagnostics")

begin
    T = diagnostics.temperature
    n = ClimaCore.Spaces.nlevels(axes(T))
    fig = Figure()
    level_s = Slider(fig[2, 1], range = 1:n)
    # plot level at slider
    level_field = map(level_s.value) do n
        ClimaCore.level(T, n)
    end
    plot(fig[1, 1], level_field, axis = (show_axis = false,))
    fig
end

sproj = [
    "+proj=weren",
    "+proj=crast",
    "+proj=wink1",
    "+proj=wink2",
    "+proj=wintri",
    "+proj=poly",
    "+proj=putp1",
    "+proj=putp2",
    "+proj=putp3",
    "+proj=putp3p",
    "+proj=putp4p",
    "+proj=putp5",
    "+proj=putp5p",
    "+proj=putp6",
    "+proj=putp6p",
    "+proj=qua_aut",
    "+proj=robin",
    "+proj=rouss",
    "+proj=rpoly",
    "+proj=sinu",
]

begin
    fig = Figure(resolution = (1600, 800))
    toggles = fig[1, 1] = GridLayout()
    plots = fig[2, 1] = GridLayout()
    colgap!(plots, 0)
    projection = Observable(sproj[1])
    fields = collect(string.(propertynames(diagnostics)))

    toggles[1, 1] = Label(fig, "earth overlay:", tellwidth = true)
    toggles[1, 2] = overlay_toggle = Toggle(fig, active = true)
    toggles[1, 3] =
        field_selector = Menu(fig, options = fields, default = "temperature")
    toggles[1, 4] =
        dest_proj = Menu(fig, options = sproj, default = projection[])

    colorrange = Observable((0.0, 1.0))

    toggles[2, :] =
        level_s = Colorbar(
            fig,
            colorrange = colorrange,
            colormap = :balance,
            vertical = false,
        )

    plots[1, 1] = ax = GeoAxis(fig; dest = projection)
    earth_overlay =
        poly!(ax, GeoMakie.land(), color = (:white, 0.2), transparency = true)
    translate!(earth_overlay, 0, 0, 10)
    connect!(earth_overlay.visible, overlay_toggle.active)

    plots[1, 2] = ax2 = LScene(fig; show_axis = false)

    colsize!(plots, 1, Relative(3 / 4))


    on(dest_proj.selection) do proj
        projection[] = proj
    end

    field_observable = Observable{Any}(diagnostics.temperature)
    on(field_selector.selection; update = true) do fieldname
        # Don't block menu when updating
        @async begin
            field_observable[] = getproperty(diagnostics, Symbol(fieldname))
        end
    end

    on(field_observable; update = true) do field
        @async begin
            colorrange[] =
                extrema(hcat(vec.(parent.(ClimaCore.level.((field,), 1:n)))...))
        end
    end

    field = field_observable[]
    n = ClimaCore.Spaces.nlevels(axes(field))

    toggles[3, :] = height_slider = Slider(fig, range = 1:n)

    field_slice = ClimaCore.level(T, 1)
    space = axes(field_slice)

    a, b, c = ClimaCore.Spaces.triangulate(space)
    triangles = GLTriangleFace.(a, b, c)
    cf = ClimaCore.Fields.coordinate_field(space)
    long, lat = vec.(parent.((cf.long, cf.lat)))
    vertices = Point2f.(long, lat)

    # plot level at slider
    field_slice_observable = Observable{Any}()
    map!(
        ClimaCore.level,
        field_slice_observable,
        field_observable,
        height_slider.value,
    )

    scalars = map(field_slice_observable) do field_slice
        Float32.(vec(parent(field_slice)))
    end

    mesh!(
        ax,
        vertices,
        triangles;
        color = scalars,
        shading = false,
        colormap = :balance,
        colorrange = colorrange,
    )
    plot!(ax2, field_slice_observable, colorrange = colorrange)

    fig
end
