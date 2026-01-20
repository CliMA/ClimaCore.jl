
"""
    fieldheatmap(field)

Plots a heatmap of a field.

## Attributes

Inherited from [`Makie.mesh`](https://docs.makie.org/stable/examples/plotting_functions/mesh/index.html#mesh).

- `colormap::Union{Symbol, Vector{<:Colorant}} = :viridis`` sets the colormap that is sampled for numeric colors.

- `colorrange::Tuple{<:Real, <:Real}` sets the values representing the start and end points of colormap.

- `nan_color::Union{Symbol, <:Colorant} = RGBAf(0,0,0,0)` sets a replacement color for `color = NaN`.

- `lowclip::Union{Automatic, Symbol, <:Colorant} = automatic` sets a color for any value below the colorrange.

- `highclip::Union{Automatic, Symbol, <:Colorant} = automatic` sets a color for any value above the colorrange.

"""
@recipe(FieldHeatmap, field) do scene
    attrs = Makie.Attributes(; coords = nothing, shading = Makie.NoShading)
    if isnothing(scene)
        return attrs
    end
    return merge(attrs, Makie.default_theme(scene, Makie.Mesh))
end

Makie.plottype(::ClimaCore.Fields.ExtrudedFiniteDifferenceField) =
    FieldHeatmap{<:Tuple{ClimaCore.Fields.ExtrudedFiniteDifferenceField}}
Makie.plottype(::ClimaCore.Fields.SpectralElementField2D) =
    FieldHeatmap{<:Tuple{ClimaCore.Fields.SpectralElementField2D}}

function Makie.plot!(plot::FieldHeatmap)
    field = plot.field

    # Only update vertices if axes updates
    space = lift(axes, field; ignore_equal_values = true)

    vertices = lift(plot_vertices, space, plot.coords)
    triangles = lift(plot_triangles, space)

    plot.color = map(vec ∘ parent, field)

    Makie.mesh!(
        plot,
        plot.attributes,
        vertices,
        triangles;
    )
end

Makie.Colorbar(fig_or_scene, plot::FieldHeatmap; kwargs...) =
    Makie.Colorbar(fig_or_scene, plot.plots[1]; kwargs...)
