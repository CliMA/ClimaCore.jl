import UnicodePlots
import RecipesBase

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
    domain = mesh.domain

    Nu = max(div(width, n1), div(height, n2))
    M = Operators.matrix_interpolate(field, Nu)
    m1, m2 = size(M)

    UnicodePlots.heatmap(
        M',
        xlabel = "x1",
        ylabel = "x2",
        xoffset = domain.x1min,
        xscale = (domain.x1max - domain.x1min) / (m1 - 1),
        yoffset = domain.x2min,
        yscale = (domain.x2max - domain.x2min) / (m2 - 1),
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
    ylabel = if field isa Spaces.FaceFiniteDifferenceSpace
        ":y faces"
    else
        ":y centers"
    end
    # fix the ylim to the column space (domain)
    ylim = extrema(ydata)
    UnicodePlots.lineplot(
        xdata,
        ydata;
        title = "Column",
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
    xdata = parent(field)[:, 1]
    ydata = parent(Spaces.coordinates_data(space))[:, 1]

    # set the plot attributes
    title --> "Column"
    xguide --> ":x value"
    yguide --> (
        if field isa Spaces.FaceFiniteDifferenceSpace
            ":y faces"
        else
            ":y centers"
        end
    )

    # fix the ylim to the column space (domain)
    ylims --> extrema(ydata)
    linewidth --> 2
    legend --> false

    (xdata, ydata)
end

RecipesBase.@recipe function f(field::Fields.SpectralElementField2D)
    # compute the interpolated data to plot
    space = axes(field)
    mesh = space.topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    domain = mesh.domain

    Nu = 10
    M = Operators.matrix_interpolate(field, Nu)
    r1 = range(domain.x1min, domain.x1max, length = n1 * Nu + 1)
    r1 = r1[1:(end - 1)] .+ step(r1) ./ 2
    r2 = range(domain.x2min, domain.x2max, length = n2 * Nu + 1)
    r2 = r2[1:(end - 1)] .+ step(r2) ./ 2

    # set the plot attributes
    seriestype := :heatmap

    xguide --> "x1"
    yguide --> "x2"
    seriescolor --> :balance

    (r1, r2, M')
end

RecipesBase.@recipe function f(field::Fields.ExtrudedFiniteDifferenceField)
    data = Fields.field_values(field)
    Ni, _, _, Nv, Nh = size(data)
    space = axes(field)
    hcoord = vec(parent(Fields.coordinate_field(space).x)[1, :, 1, :])
    vcoord = vec(parent(Fields.coordinate_field(space).z)[:, 1, 1, 1])

    # assumes VIFH layout
    # set the plot attributes
    seriestype := :heatmap

    xguide --> "x"
    yguide --> "z"
    seriescolor --> :balance

    (hcoord, vcoord, reshape(parent(data), (Nv, Ni * Nh)))
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
