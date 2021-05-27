import UnicodePlots, Plots

function UnicodePlots.heatmap(
    field::Fields.SpectralElementField2D;
    width = 80,
    height = 40,
    kwargs...,
)
    if !(eltype(field) <: Real)
        error("Can only plot heatmaps of scalar fields")
    end

    space = Fields.space(field)
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
    width = 80,
    height = 40,
    kwargs...,
)
    space = Fields.space(field)
    x = parent(field)[Spaces.interior_face_range(space), 1]
    # TODO: use scaled domains using coordinates
    UnicodePlots.lineplot(
        x;
        xlabel = "column h",
        ylabel = "column y",
        width = width,
        height = height,
        kwargs...,
    )
end


function Plots.heatmap(field::Fields.SpectralElementField2D; kwargs...)
    space = Fields.space(field)
    mesh = fieldmesh.topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    domain = mesh.domain

    Nu = 10
    M = Operators.matrix_interpolate(field, Nu)
    m1, m2 = size(M)


    r1 = range(domain.x1min, domain.x1max, length = n1 * Nu + 1)
    r1 = r1[1:(end - 1)] .+ step(r1) ./ 2
    r2 = range(domain.x2min, domain.x2max, length = n2 * Nu + 1)
    r2 = r2[1:(end - 1)] .+ step(r2) ./ 2

    Plots.heatmap(r1, r2, M'; xlabel = "x1", ylabel = "x2", kwargs...)
end
