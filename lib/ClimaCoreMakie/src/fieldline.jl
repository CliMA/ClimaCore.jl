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
        vertices = parent(ClimaCore.Spaces.coordinates_data(space))
        Nq, _, Nh = size(vertices)
        vals = parent(f)
        positions = [
            i <= Nq ? Makie.Point2f(vertices[i, 1, h], vals[i, 1, h]) :
            Makie.Point2f(NaN, NaN) for i in 1:(Nq + 1), h in 1:Nh
        ]
        return (vec(positions),)
    end


    Makie.lines!(plot, plot.positions)
end
