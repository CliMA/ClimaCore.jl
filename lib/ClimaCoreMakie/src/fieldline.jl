@recipe(FieldLine, field) do scene
    attrs = Makie.Attributes()
    if isnothing(scene)
        return attrs
    end
    return merge(attrs, Makie.default_theme(scene, Makie.Lines))
end

Makie.plottype(::ClimaCore.Fields.SpectralElementField1D) =
    FieldLine{<:Tuple{ClimaCore.Fields.SpectralElementField1D}}

function Makie.plot!(
    plot::FieldLine{<:Tuple{ClimaCore.Fields.SpectralElementField1D}},
)

    field = plot.field
    # Only update vertices if space updates
    space = lift(axes, field; ignore_equal_values = true)

    vertices = parent(ClimaCore.Spaces.coordinates_data(space[]))

    # insert NaNs to create gaps between elements
    Nq, _, Nh = size(vertices[])
    vals = parent(field[])
    positions = [
        i <= Nq ? Makie.Point2f(vertices[i, 1, h], vals[i, 1, h]) :
        Makie.Point2f(NaN, NaN) for i in 1:(Nq + 1), h in 1:Nh
    ]

    Makie.lines!(plot, vec(positions))
end
