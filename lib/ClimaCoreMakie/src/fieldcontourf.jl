
"""
    fieldcontourf(field::Field)

Plots a 2D filled contour plot of a field.

## Attributes

These are inherited from [`Makie.tricontourf`](https://docs.makie.org/stable/examples/plotting_functions/tricontourf/):

- `levels = 10` can be either an `Int` which results in n bands delimited by n+1 equally spaced levels, or it can be an `AbstractVector{<:Real}` that lists n consecutive edges from low to high, which result in n-1 bands.
- `mode = :normal` sets the way in which a vector of levels is interpreted, if it's set to `:relative`, each number is interpreted as a fraction between the minimum and maximum values of `zs`. For example, `levels = 0.1:0.1:1.0` would exclude the lower 10% of data.
- `extendlow = nothing`. This sets the color of an optional additional band from `minimum(zs)` to the lowest value in `levels`. If it's `:auto`, the lower end of the colormap is picked and the remaining colors are shifted accordingly. If it's any color representation, this color is used. If it's `nothing`, no band is added.
- `extendhigh = nothing`. This sets the color of an optional additional band from the highest value of `levels` to `maximum(zs)`. If it's `:auto`, the high end of the colormap is picked and the remaining colors are shifted accordingly. If it's any color representation, this color is used. If it's `nothing`, no band is added.
- `color` sets the color of the plot. It can be given as a named color `Symbol` or a `Colors.Colorant`. Transparency can be included either directly as an alpha value in the `Colorant` or as an additional float in a tuple `(color, alpha)`. The color can also be set for each scattered marker by passing a `Vector` of colors or be used to index the `colormap` by passing a `Real` number or `Vector{<: Real}`.
- `colormap::Union{Symbol, Vector{<:Colorant}} = :viridis` sets the colormap from which the band colors are sampled.

"""
@recipe(FieldContourf, field) do scene
    attrs = Makie.Attributes(; coords = nothing)
    return merge(attrs, Makie.default_theme(scene, Makie.Tricontourf))
end

function Makie.plot!(plot::FieldContourf)

    field = plot.field
    # Only update vertices if axes updates
    space = lift(axes, field; ignore_equal_values = true)
    vertices = lift(plot_vertices, space, plot.coords)
    plot.triangulation = lift(plot_triangles_matrix, space)

    xs = lift(vertices) do vertices
        map(x -> x[1], vertices)
    end
    ys = lift(vertices) do vertices
        map(x -> x[2], vertices)
    end
    zs = map(vec ∘ parent, field)

    Makie.tricontourf!(
        plot,
        xs,
        ys,
        zs;
        ((k, v) for (k, v) in pairs(plot.attributes) if k !== :coords)...,
    )
end

Makie.Colorbar(fig_or_scene, plot::FieldContourf; kwargs...) =
    Makie.Colorbar(fig_or_scene, plot.plots[1]; kwargs...)
