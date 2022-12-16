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

# Downloading and unpacking time series needs to happen manually from:
# https://caltech.box.com/shared/static/a65jnfdfpwqt9ka9nszvoedcg412c6q1.gz

time_series = map(readdir(joinpath(pwd(), "time-series"); join = true)) do file
    reader = ClimaCore.InputOutput.HDF5Reader(file)
    return ClimaCore.InputOutput.read_field(reader, "diagnostics")
end
field = time_series[1].temperature
nlevels = ClimaCore.Spaces.nlevels(axes(field))
# Calculate a global color range per field type, so that the colorrange stays static when moving time/level.
color_range_per_field = Dict(
    map(propertynames(time_series[1])) do name
        vecs = [
            extrema(
                hcat(
                    vec.(
                        parent.(
                            ClimaCore.level.(
                                (getproperty(slice, name),),
                                1:nlevels,
                            )
                        )
                    )...,
                ),
            ) for slice in time_series
        ]
        name => reduce(
            ((amin, amax), (bmin, bmax)) ->
                (min(amin, bmin), max(amax, bmax)),
            vecs,
        )
    end,
)

begin
    fig = Figure(resolution = (1600, 800))
    ####
    ### Setup the grid layout for the menu and different plots
    ####

    # One slot for menus + toggles + colorbar
    toggles = fig[1, 1] = GridLayout()
    plots = fig[2, 1] = GridLayout()
    colgap!(plots, 0) # leave no space between plots

    # Define a few observables we need to create entries, which get connected to the widgets later
    projection = Observable(sproj[1])
    fields = collect(string.(propertynames(time_series[1])))
    # Two sliders for setting Time + Label
    toggles[1, :] =
        sliders = SliderGrid(
            fig,
            (label = "Time", range = 1:length(time_series)),
            (label = "Level", range = 1:nlevels),
        )
    time_slider, level_slider = sliders.sliders
    # Need a sub gridlayout, to get the columns aligned with the sliders
    menu_items = toggles[2, :] = GridLayout()
    # Use labels to label toggle + menu
    menu_items[1, 1] = Label(fig, "earth overlay:", tellwidth = true)
    menu_items[1, 2] = overlay_toggle = Toggle(fig, active = true)

    menu_items[1, 3] = Label(fig, "field:", tellwidth = true)
    menu_items[1, 4] =
        field_selector = Menu(fig, options = fields, default = "temperature")

    menu_items[1, 5] = Label(fig, "projection:", tellwidth = true)
    menu_items[1, 6] =
        dest_proj = Menu(fig, options = sproj, default = projection[])
    # Have a global color range we
    colorrange = Observable((0.0, 1.0))
    toggles[3, :] =
        level_s = Colorbar(
            fig,
            colorrange = colorrange,
            colormap = :balance,
            vertical = false,
        )

    # Create a geo axis to handle the projection
    # Note, that Makie still needs some work in the event propagation of menu + friends,
    # so when selecting e.g. a field, it may happen that it changes the axis zoom level
    # use ctrl + double mouse click to reset it.
    plots[1, 1] = ax = GeoAxis(fig; dest = projection)

    # Disable all interactions with the axis for now, since they're buggy :(
    foreach(name -> deregister_interaction!(ax, name), keys(interactions(ax)))

    # Create a toggable land overlay
    earth_overlay =
        poly!(ax, GeoMakie.land(), color = (:white, 0.2), transparency = true)
    translate!(earth_overlay, 0, 0, 10)
    connect!(earth_overlay.visible, overlay_toggle.active)
    # create a 3D plot without an axis for mapping the simulation on the sphere
    plots[1, 2] = ax2 = LScene(fig; show_axis = false)

    colsize!(plots, 1, Relative(3 / 4)) # Give the GeoAxis more space

    # Now, connect all the observables to the widgets
    on(dest_proj.selection) do proj
        projection[] = proj
    end

    field_observable = Observable{Any}()
    #=
    map!(f, new_observable, observable) works like:
    new_observable = map(f, observable)
    which works like:
    # run everytime observable updates
    on(observable) do new_value
        new_observable[] = new_value
    end
    =#
    map!(
        field_observable,
        field_selector.selection,
        time_slider.value,
    ) do fieldname, idx
        return getproperty(time_series[idx], Symbol(fieldname))
    end
    # update=true, runs f immediately, otherwise f only runs when the input observable is triggered the first time
    on(field_selector.selection; update = true) do field_name
        # select the correct color range from the globally calcualted color ranges
        colorrange[] = color_range_per_field[Symbol(field_name)]
        return
    end

    # manually create the mesh for the 3D Sphere plot:

    field_slice = ClimaCore.level(field, 1)
    space = axes(field_slice)
    a, b, c = ClimaCore.Spaces.triangulate(space)
    triangles = GLTriangleFace.(a, b, c)
    cf = ClimaCore.Fields.coordinate_field(space)
    long, lat = vec.(parent.((cf.long, cf.lat)))
    vertices = Point2f.(long, lat)

    # plot level at slider, needs to be any since the  type changes (also the reason why we use map! instead of map)
    field_slice_observable = Observable{Any}()
    map!(
        ClimaCore.level,
        field_slice_observable,
        field_observable,
        level_slider.value,
    )

    # extract the scalar field
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
