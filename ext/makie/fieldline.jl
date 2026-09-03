"""
    fieldline(field)

Plots a line plot of a 1D field.

## Attributes

Inherited from [`Makie.lines`](https://docs.makie.org/stable/examples/plotting_functions/lines/index.html#lines).

  - `color` sets the color of the line.

  - `linewidth` sets the width of the line.

  - `linestyle` sets the style of the line (e.g., `:solid`, `:dash`, `:dot`).

See the [Makie.lines documentation](https://docs.makie.org/stable/examples/plotting_functions/lines/index.html#lines) for a complete list of attributes.
"""
@recipe FieldLine (field,) begin
    Makie.documented_attributes(Makie.Lines)...
end

Makie.plottype(::ClimaCore.Fields.SpectralElementField1D) =
    FieldLine{<:Tuple{ClimaCore.Fields.SpectralElementField1D}}

function Makie.plot!(
    plot::FieldLine{<:Tuple{ClimaCore.Fields.SpectralElementField1D}},
)

    input_nodes = [:field]
    output_nodes = [:positions]
    map!(plot.attributes, input_nodes, output_nodes) do f
        space = axes(f)
        # parent arrays are laid out as (V, I, J, F, H)
        vertices = parent(ClimaCore.Spaces.coordinates_data(space))
        (_, Nq, _, _, Nh) = size(vertices)
        vals = parent(f)
        positions = [
            i <= Nq ?
            Makie.Point2f(vertices[1, i, 1, 1, h], vals[1, i, 1, 1, h]) :
            Makie.Point2f(NaN, NaN) for i in 1:(Nq + 1), h in 1:Nh
        ]
        return (vec(positions),)
    end


    Makie.lines!(plot, plot.positions)
end
