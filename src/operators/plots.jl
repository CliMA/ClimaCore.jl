import UnicodePlots, Plots

function UnicodePlots.heatmap(field::Field; width = 80, height = 40, kwargs...)
    if !(eltype(field) <: Real)
        error("Can only plot heatmaps of scalar fields")
    end

    fieldmesh = Fields.mesh(field)
    discretization = fieldmesh.topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    domain = discretization.domain

    Nu = max(div(width, n1), div(height, n2))
    M = matrix_interpolate(field, Nu)
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



function Plots.heatmap(field::Field; kwargs...)
    fieldmesh = Fields.mesh(field)
    discretization = fieldmesh.topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    domain = discretization.domain

    Nu = 10
    M = matrix_interpolate(field, Nu)
    m1, m2 = size(M)


    r1 = range(domain.x1min, domain.x1max, length = n1 * Nu + 1)
    r1 = r1[1:(end - 1)] .+ step(r1) ./ 2
    r2 = range(domain.x2min, domain.x2max, length = n2 * Nu + 1)
    r2 = r2[1:(end - 1)] .+ step(r2) ./ 2

    Plots.heatmap(r1, r2, M'; xlabel = "x1", ylabel = "x2", kwargs...)
end
