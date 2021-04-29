import UnicodePlots

function UnicodePlots.heatmap(field::Field; width=80, height=40, kwargs...)
    fieldmesh = mesh(field)
    discretization = fieldmesh.topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    domain = discretization.domain

    Nu = max(div(width,n1), div(height,n2))
    M = matrix_interpolate(field, Nu)
    m1,m2 = size(M)

    UnicodePlots.heatmap(M',
       xlabel="x1", ylabel="x2",
       xoffset=domain.x1min,
       xscale=(domain.x1max-domain.x1min)/(m1-1),
       yoffset=domain.x2min,
       yscale=(domain.x2max-domain.x2min)/(m2-1),
       width=width, height=height,
       kwargs...
    )
end