import UnicodePlots
import RecipesBase
import TriplotBase

function UnicodePlots.heatmap(
    field::Fields.SpectralElementField2D;
    width = 80,
    height = 40,
    kwargs...,
)
    if !(eltype(field) <: Real)
        error("Can only plot heatmaps of scalar fields")
    end

    space = axes(field)
    mesh = space.topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2

    Nu = max(div(width, n1), div(height, n2))
    M = Operators.matrix_interpolate(field, Nu)

    m1, m2 = size(M')
    domain = Meshes.domain(mesh)
    x1min = Geometry.component(domain.x1x2min, 1)
    x2min = Geometry.component(domain.x1x2min, 2)
    x1max = Geometry.component(domain.x1x2max, 1)
    x2max = Geometry.component(domain.x1x2max, 2)

    coord_field = Fields.coordinate_field(space)
    coord_symbols = propertynames(coord_field)

    UnicodePlots.heatmap(
        M',
        xlabel = "$(coord_symbols[1])",
        ylabel = "$(coord_symbols[2])",
        xoffset = x1min,
        xfact = (x1max - x1min) / (m1 - 1),
        yoffset = x2min,
        yfact = (x2max - x2min) / (m2 - 1),
        width = width,
        height = height,
        kwargs...,
    )
end

function UnicodePlots.lineplot(
    field::Fields.FiniteDifferenceField;
    name::Union{Nothing, Symbol} = nothing,
    width = 80,
    height = 40,
    kwargs...,
)
    if name !== nothing
        field = getproperty(field, name)
    else
        name = :x
    end
    space = axes(field)

    xlabel = repr(name) * " value"
    xdata = Array(parent(field))[:, 1]

    ydata = Array(parent(Spaces.coordinates_data(space)))[:, 1]

    coord_field = Fields.coordinate_field(space)
    coord_symbols = propertynames(coord_field)

    ylabel = if field isa Spaces.FaceFiniteDifferenceSpace
        "$(coord_symbols[1]) faces"
    else
        "$(coord_symbols[1]) centers"
    end
    # fix the ylim to the column space (domain)
    ylim = extrema(ydata)
    UnicodePlots.lineplot(
        xdata,
        ydata;
        xlabel = xlabel,
        ylabel = ylabel,
        ylim = [ylim[1], ylim[2]],
        width = width,
        height = height,
        kwargs...,
    )
end

RecipesBase.@recipe function f(field::Fields.FiniteDifferenceField)
    # unwrap the data to plot
    space = axes(field)
    coord_field = Fields.coordinate_field(space)

    xdata = parent(field)[:, 1]
    ydata = parent(Spaces.coordinates_data(space))[:, 1]

    coord_symbols = propertynames(coord_field)

    # set the plot attributes
    xguide --> "value"
    yguide --> (
        if field isa Spaces.FaceFiniteDifferenceSpace
            "$(coord_symbols[1]) faces"
        else
            "$(coord_symbols[1]) centers"
        end
    )

    # fix the ylim to the column space (domain)
    ylims --> extrema(ydata)
    linewidth --> 2
    legend --> false

    (xdata, ydata)
end

RecipesBase.@recipe function f(space::Spaces.SpectralElementSpace2D;)
    quad = Spaces.quadrature_style(space)
    quad_name = Base.typename(typeof(quad)).name
    dof = Spaces.Quadratures.degrees_of_freedom(quad)

    topology = Spaces.topology(space)
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2

    coord_field = Fields.coordinate_field(space)
    x1coord = vec(parent(coord_field)[:, :, 1, :])
    x2coord = vec(parent(coord_field)[:, :, 2, :])

    coord_symbols = propertynames(coord_field)

    seriestype := :scatter
    title --> "$n1 × $n2 $quad_name{$dof} element space"
    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    marker --> :cross
    markersize --> 1
    seriescolor --> :blue
    legend --> false

    (x1coord, x2coord)
end




RecipesBase.@recipe function f(space::Spaces.ExtrudedFiniteDifferenceSpace)
    coord_field = Fields.coordinate_field(space)
    data = Fields.field_values(coord_field)
    Ni, Nj, _, Nv, Nh = size(data)

    #TODO: assumes VIFH layout
    @assert Nj == 1 "plotting only defined for 1D extruded fields"

    hspace = space.horizontal_space

    quad = Spaces.quadrature_style(hspace)
    quad_name = Base.typename(typeof(quad)).name
    dof = Spaces.Quadratures.degrees_of_freedom(quad)

    coord_symbols = propertynames(coord_field)
    hcoord = vec(parent(coord_field)[:, :, 1, :])
    vcoord = vec(parent(coord_field)[:, :, 2, :])

    stagger = space.staggering isa Spaces.CellCenter ? :center : :face

    seriestype := :scatter
    title --> "$Nh $quad_name{$dof} element × $Nv level $stagger space"
    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    marker --> :cross
    markersize --> 1
    seriescolor --> :blue
    legend --> false

    (hcoord, vcoord)
end


RecipesBase.@recipe function f(
    field::Fields.SpectralElementField2D;
    interpolate = 10,
)
    @assert interpolate ≥ 1 "number of element quadrature points for uniform interpolation must be ≥ 1"

    # compute the interpolated data to plot
    space = axes(field)
    topology = Spaces.topology(space)
    mesh = topology.mesh

    Nu = interpolate
    coord_field = Fields.coordinate_field(space)

    M_coords = Operators.matrix_interpolate(coord_field, Nu)
    M = Operators.matrix_interpolate(field, Nu)

    domain = Meshes.domain(mesh)
    x1min = Geometry.component(domain.x1x2min, 1)
    x2min = Geometry.component(domain.x1x2min, 2)
    x1max = Geometry.component(domain.x1x2max, 1)
    x2max = Geometry.component(domain.x1x2max, 2)

    # our interpolated field is transposed
    x1coord = [Geometry.component(pt, 1) for pt in M_coords[:, 1]]
    x2coord = [Geometry.component(pt, 2) for pt in M_coords[1, :]]

    coord_symbols = propertynames(coord_field)

    # set the plot attributes
    seriestype := :heatmap

    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    seriescolor --> :balance

    (x1coord, x2coord, M')
end

RecipesBase.@recipe function f(
    field::Fields.CubedSphereSpectralElementField2D;
    interpolate = 10,
)
    @assert interpolate ≥ 1 "number of element quadrature points for uniform interpolation must be ≥ 1"

    space = axes(field)
    topology = Spaces.topology(space)
    mesh = topology.mesh

    nelem = Topologies.nlocalelems(topology)
    panel_size = isqrt(div(nelem, 6))

    quad_from = Spaces.quadrature_style(space)
    quad_to = Spaces.Quadratures.Uniform{interpolate}()
    Imat = Spaces.Quadratures.interpolation_matrix(Float64, quad_to, quad_from)

    dof_in = Spaces.Quadratures.degrees_of_freedom(quad_from)
    dof = interpolate

    pannel_range(i) =
        ((panel_size * dof) * (i - 1) + 1):((panel_size * dof) * i)

    # construct a matrix to fill in the rotated / flipped pannel data
    unfolded_panels =
        fill(NaN, ((panel_size * dof) * 3, (panel_size * dof) * 4))

    # temporary pannels as we have to rotate / flip some and not all operators are in place
    # TODO: inefficient memory wise, but good enough for now
    panels = [fill(NaN, (panel_size * dof, panel_size * dof)) for _ in 1:6]

    interpolated_data =
        DataLayouts.IJFH{Float64, interpolate}(Array{Float64}, nelem)
    Operators.tensor_product!(
        interpolated_data,
        Fields.field_values(field),
        Imat,
    )

    # element index ordering defined by a specific layout
    eidx = 1
    for panel_idx in 1:6
        panel_data = panels[panel_idx]
        # elements are ordered along fastest axis defined in sphere box mesh
        for ex2 in 1:panel_size, ex1 in 1:panel_size
            # compute the nodal extent index range for this element
            x1_nodal_range = (dof * (ex1 - 1) + 1):(dof * ex1)
            x2_nodal_range = (dof * (ex2 - 1) + 1):(dof * ex2)
            # transpose the data as our plotting axis order is
            # reverse nodal element order (x1 axis varies fastest)
            data_element = permutedims(parent(interpolated_data)[:, :, 1, eidx])
            panel_data[x2_nodal_range, x1_nodal_range] = data_element
            eidx += 1
        end
    end

    unfolded_panels[pannel_range(1), pannel_range(2)] =
        reverse(panels[5], dims = 1)
    unfolded_panels[pannel_range(2), pannel_range(1)] =
        reverse(panels[4], dims = 2)
    unfolded_panels[pannel_range(2), pannel_range(2)] = transpose(panels[1])
    unfolded_panels[pannel_range(2), pannel_range(3)] = transpose(panels[2])
    unfolded_panels[pannel_range(2), pannel_range(4)] =
        reverse(panels[6], dims = 2)
    unfolded_panels[pannel_range(3), pannel_range(2)] =
        reverse(panels[3], dims = 2)

    quad_from_name = Base.typename(typeof(quad_from)).name
    # set the plot attributes
    seriestype := :heatmap
    title --> "$nelem $quad_from_name{$dof_in} element space"
    xguide --> "panel x1"
    yguide --> "panel x2"
    seriescolor --> :balance

    (unfolded_panels)
end

function triangulate(Ni, Nv, Nh)
    L = LinearIndices((1:Nv, 1:Ni))
    I = vec([
        (t == 1 ? L[v, i] : L[v + 1, i]) + Nv * Ni * (h - 1) for t in 1:2,
        v in 1:(Nv - 1), i in 1:(Ni - 1), h in 1:Nh
    ])
    J = vec([
        (t == 1 ? L[v + 1, i] : L[v + 1, i + 1]) + Nv * Ni * (h - 1) for
        t in 1:2, v in 1:(Nv - 1), i in 1:(Ni - 1), h in 1:Nh
    ])
    K = vec([
        (t == 1 ? L[v, i + 1] : L[v, i + 1]) + Nv * Ni * (h - 1) for
        t in 1:2, v in 1:(Nv - 1), i in 1:(Ni - 1), h in 1:Nh
    ])
    return hcat(I, J, K)'
end

function triangulate(field::Fields.ExtrudedFiniteDifferenceField)
    field_data = Fields.field_values(field)
    Ni, Nj, _, Nv, Nh = size(field_data)
    @assert Nj == 1 "triangulation only defined for 1D extruded fields"
    return triangulate(Ni, Nv, Nh)
end

RecipesBase.@recipe function f(
    field::Fields.ExtrudedFiniteDifferenceField;
    hinterpolate = 0,
    ncolors = 256,
)
    data = Fields.field_values(field)
    Ni, Nj, _, Nv, Nh = size(data)

    space = axes(field)
    #TODO: assumes VIFH layout
    @assert Nj == 1 "plotting only defined for 1D extruded fields"

    coord_symbols = propertynames(Fields.coordinate_field(space))
    hcoord_field = getproperty(Fields.coordinate_field(space), 1)
    vcoord_field = getproperty(Fields.coordinate_field(space), 2)

    if hinterpolate ≥ 1
        Nu = hinterpolate
        uquad = Spaces.Quadratures.ClosedUniform{Nu}()
        M_hcoord = Operators.matrix_interpolate(hcoord_field, uquad)
        M_vcoord = Operators.matrix_interpolate(vcoord_field, uquad)
        M_data = Operators.matrix_interpolate(field, uquad)

        hcoord_data = vec(M_hcoord)
        vcoord_data = vec(M_vcoord)
        data = vec(M_data)
        triangles = triangulate(Nu, Nv, Nh)
    else
        hcoord_data = vec(parent(hcoord_field))
        vcoord_data = vec(parent(vcoord_field))
        data = vec(parent(data))
        triangles = triangulate(Ni, Nv, Nh)
    end
    cmap = range(extrema(data)..., length = ncolors)

    z = TriplotBase.tripcolor(
        hcoord_data,
        vcoord_data,
        data,
        triangles,
        cmap;
        bg = NaN,
        px = length(hcoord_data),
        py = length(vcoord_data),
    )

    # set the plot attributes
    seriestype := :heatmap

    xguide --> "$(coord_symbols[1])"
    yguide --> "$(coord_symbols[2])"
    seriescolor --> :balance

    # some plots backends need coordinates in sorted order
    (sort(hcoord_data), sort(vcoord_data), z')
end


function play(
    timesteps::Vector;
    name::Union{Nothing, Symbol} = nothing,
    fps::Real = 5,
    width = 80,
    height = 40,
    paused = true,
)
    #TODO: compute extrema over timetep fields
    io_w, io_h = width, height
    nframes = length(timesteps)

    setraw!(io, raw) =
        ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), io.handle, raw)
    ansi_moveup(n::Int) = string("\e[", n, "A")
    ansi_movecol1 = "\e[1G"
    ansi_enablecursor = "\e[?25h"
    ansi_disablecursor = "\e[?25l"

    frame = lastframe = 1
    finished = false
    first_print = true
    actual_fps = 0

    # column y limits are consistent based on the space
    # compute the extrema of over field values for all timesteps

    #= TODO: need to implement field iteration
    field_limits = [extrema(timesteps[1])...]
    if length(timesteps) > 1
        for t in timesteps[2:end]
            tlim = extrema(t)
     if tlim[1] < xlim[1]
    field_limits[1] = tlim[1]
     end
     if tlim[2] > xlim[2]
    field_limits[2] = tlim[2]
     end
    end
    end
    =#

    keytask = @async begin
        try
            setraw!(stdin, true)
            while !finished
                keyin = read(stdin, Char)
                if UInt8(keyin) == 27
                    keyin = read(stdin, Char)
                    if UInt8(keyin) == 91
                        keyin = read(stdin, Char)
                        # left & up arrows
                        if UInt8(keyin) in [68, 65]
                            frame = frame <= 1 ? 1 : frame - 1
                        end
                        # right & down arrows
                        if UInt8(keyin) in [67, 66]
                            frame = frame >= nframes ? nframes : frame + 1
                        end
                    end
                end
                # seek to some pct of simulation time (1 start, 0 end)
                if keyin in '0':'9'
                    k = parse(Int, keyin)
                    pct = (k == 0 ? 10 : k) / 10.0
                    frame = round(Int, pct * nframes)
                end
                # p or spacebar to pause
                if keyin in ['p', ' ']
                    paused = !paused
                end
                # esc or q to quite
                if keyin in ['\x03', 'q']
                    finished = true
                end
            end
        catch
        finally
            setraw!(stdin, false)
        end
    end
    try
        print(ansi_disablecursor)
        setraw!(stdin, true)
        while !finished
            tim = Timer(1 / fps)
            t = @elapsed begin
                if first_print || lastframe != frame
                    field = timesteps[frame]
                    field_plot = if field isa Fields.FiniteDifferenceField
                        UnicodePlots.lineplot(
                            field,
                            title = "Column t=$(frame-1)",
                            width = io_w,
                            height = io_h,
                            #xlim = field_limits,
                        )
                    elseif field isa Fields.SpectralElementField2D
                        UnicodePlots.heatmap(field, width = io_w, height = io_h)
                    else
                        error("unknown field type: $(summary(field))")
                    end
                    # TODO: dig through structure to just compute the number of lines directly
                    nlines = countlines(IOBuffer(string(field_plot)))
                    str = sprint(; context = (:color => true)) do ios
                        println(ios, field_plot)
                        if paused
                            println(ios, "Frame: $frame/$nframes", " "^15)
                        else
                            println(
                                ios,
                                "Frame: $frame/$nframes FPS: $(round(actual_fps, digits=1))",
                                " "^5,
                            )
                        end
                    end
                    first_print ? print(str) :
                    print(ansi_moveup(nlines + 1), ansi_movecol1, str)
                end
                first_print = false
                lastframe = frame
                if !paused
                    if frame == nframes
                        frame = 1
                    else
                        frame += 1
                    end
                end
                wait(tim)
            end
            actual_fps = 1 / t
        end
    catch e
        isa(e, InterruptException) || rethrow()
    finally
        print(ansi_enablecursor)
        finished = true
        @async Base.throwto(keytask, InterruptException())
        wait(keytask)
    end
    return
end
