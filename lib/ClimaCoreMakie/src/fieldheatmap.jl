
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
@recipe FieldHeatmap (field,) begin
    coords = nothing
    shading = Makie.NoShading
    Makie.filter_attributes(
        Makie.documented_attributes(Makie.Mesh);
        exclude = (:coords, :shading),
    )...
end

Makie.plottype(::ClimaCore.Fields.ExtrudedFiniteDifferenceField) =
    FieldHeatmap{<:Tuple{ClimaCore.Fields.ExtrudedFiniteDifferenceField}}
Makie.plottype(::ClimaCore.Fields.SpectralElementField2D) =
    FieldHeatmap{<:Tuple{ClimaCore.Fields.SpectralElementField2D}}

function Makie.plot!(plot::FieldHeatmap)

    # Only update vertices if axes updates
    input_nodes = [:field, :coords]
    output_nodes = [:vertices, :triangles, :new_color]
    map!(plot.attributes, input_nodes, output_nodes) do f, coords
        space = axes(f)
        vertices = plot_vertices(space, coords)
        triangles = plot_triangles(space)
        new_color = vec(parent(f))
        return (vertices, triangles, new_color)
    end
    Makie.mesh!(
        plot,
        plot.attributes,
        plot.vertices,
        plot.triangles;
        color = plot.new_color,
    )
end

Makie.Colorbar(fig_or_scene, plot::FieldHeatmap; kwargs...) =
    Makie.Colorbar(fig_or_scene, plot.plots[1]; kwargs...)
