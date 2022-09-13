module ClimaCoreMakie

import Makie: Makie, @recipe, lift, GLTriangleFace, Point3f
import ClimaCore

@recipe(Viz, field) do scene
    return Makie.Attributes(;
        colormap = :balance,
        shading = false
    )
end

Makie.plottype(::ClimaCore.Fields.SpectralElementField2D) =
    Viz{<:Tuple{ClimaCore.Fields.SpectralElementField2D}}

function Makie.plot!(
    plot::Viz{<:Tuple{ClimaCore.Fields.SpectralElementField2D}},
)
    # Use ignore_equal_values to not trigger an update when nothing changes
    space = lift(plot.field; ignore_equal_values=true) do field
        if !(eltype(field) <: Union{Float64, Float32})
            error("plotting only implemented for F64, F32 scalar fields")
        end
        return ClimaCore.axes(field)
    end

    # Only update vertices if global_geometry updates (is this the correct value to check for changes?)
    global_geometry = lift(x-> x.global_geometry, space; ignore_equal_values=true)
    vertices = lift(global_geometry) do gg
        cartesian = ClimaCore.Geometry.Cartesian123Point.(
            ClimaCore.Fields.coordinate_field(space[]),
            Ref(gg),
        )
        # Use the OpenGL native Point3f type to avoid conversions
        return Point3f.(
            vec(parent(cartesian.x1)),
            vec(parent(cartesian.x2)),
            vec(parent(cartesian.x3))
        )
    end

    # Triangles stay static if the size doesn't change, so we lift and only trigger updates if size changes:
    space_size = lift(x-> size(x.local_geometry), space; ignore_equal_values=true)

    triangles = lift(space_size) do (Ni, Nj, _, _, Nh)
        a, b, c = ClimaCore.Spaces.triangles(Ni, Nj, Nh)
        # Use the native GPU types to avoid copies/conversions
        return GLTriangleFace.(a, b, c)
    end

    colors = lift(x-> Float32.(vec(parent(x))), plot.field)

    Makie.mesh!(
        plot,
        vertices,
        triangles,
        color = colors,
        colormap = plot.colormap,
        shading = plot.shading,
    )
end

end # module
