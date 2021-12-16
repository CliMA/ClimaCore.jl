module ClimaCoreMakie

import Makie: Makie, @recipe
import ClimaCore

@recipe(Viz, object) do scene
    Makie.Attributes(; colormap = :balance)
end

function level_mesh(field::ClimaCore.Fields.SpectralElementField2D, level = 1)
    space = axes(field)
    Ni, Nj, _, _, Nh = size(space.local_geometry)
    @assert Ni == Nj

    cart_coords =
        ClimaCore.Geometry.Cartesian123Point.(
            ClimaCore.Fields.coordinate_field(space),
            Ref(space.global_geometry),
        )

    triangle_count = Nh * (Nj - 1) * (Ni - 1) * 2
    FaceType = Makie.GeometryBasics.GLTriangleFace
    PointType = Makie.GeometryBasics.Point3f

    faces = Array{FaceType}(undef, triangle_count)
    vertices = Array{PointType}(undef, triangle_count * 3)

    x1_data = ClimaCore.Fields.todata(cart_coords.x1)
    x2_data = ClimaCore.Fields.todata(cart_coords.x2)
    x3_data = ClimaCore.Fields.todata(cart_coords.x3)

    L = LinearIndices((1:Ni, 1:Nj))
    idx = 0
    for h in 1:Nh
        x1_slab = ClimaCore.slab(x1_data, level, h)
        x2_slab = ClimaCore.slab(x2_data, level, h)
        x3_slab = ClimaCore.slab(x3_data, level, h)
        for j in 1:(Nj - 1), i in 1:(Ni - 1), t in 1:2
            if t == 1
                # quad triangle 1
                I = L[i, j] + Ni * Nj * (h - 1)
                J = L[i + 1, j] + Ni * Nj * (h - 1)
                K = L[i, j + 1] + Ni * Nj * (h - 1)
                faces[idx + 1] = Makie.GeometryBasics.GLTriangleFace(I, J, K)
                vertices[idx * 3 + 1] = Makie.GeometryBasics.Point3f(
                    x1_slab[i, j],
                    x2_slab[i, j],
                    x3_slab[i, j],
                )
                vertices[idx * 3 + 2] = Makie.GeometryBasics.Point3f(
                    x1_slab[i + 1, j],
                    x2_slab[i + 1, j],
                    x3_slab[i + 1, j],
                )
                vertices[idx * 3 + 3] = Makie.GeometryBasics.Point3f(
                    x1_slab[i, j + 1],
                    x2_slab[i, j + 1],
                    x3_slab[i, j + 1],
                )
            else
                # quad triangle 2
                I = L[i + 1, j] + Ni * Nj * (h - 1)
                J = L[i + 1, j + 1] + Ni * Nj * (h - 1)
                K = L[i, j + 1] + Ni * Nj * (h - 1)
                faces[idx + 1] = Makie.GeometryBasics.GLTriangleFace(I, J, K)
                vertices[idx * 3 + 1] = Makie.GeometryBasics.Point3f(
                    x1_slab[i + 1, j],
                    x2_slab[i + 1, j],
                    x3_slab[i + 1, j],
                )
                vertices[idx * 3 + 2] = Makie.GeometryBasics.Point3f(
                    x1_slab[i + 1, j + 1],
                    x2_slab[i + 1, j + 1],
                    x3_slab[i + 1, j + 1],
                )
                vertices[idx * 3 + 3] = Makie.GeometryBasics.Point3f(
                    x1_slab[i, j + 1],
                    x2_slab[i, j + 1],
                    x3_slab[i, j + 1],
                )
            end
            idx += 1
        end
    end
    return Makie.GeometryBasics.Mesh(vertices, faces)
end

Makie.plottype(::ClimaCore.Fields.SpectralElementField2D) =
    Viz{<:Tuple{ClimaCore.Fields.SpectralElementField2D}}

function Makie.plot!(
    plot::Viz{<:Tuple{ClimaCore.Fields.SpectralElementField2D}},
)
    # retrieve the field to plot
    field = plot[:object][]

    if !(eltype(field) <: Union{Float64, Float32})
        error("plotting only implemented for F64, F32 scalar fields")
    end
    space = ClimaCore.axes(field)

    cart_coords =
        ClimaCore.Geometry.Cartesian123Point.(
            ClimaCore.Fields.coordinate_field(space),
            Ref(space.global_geometry),
        )
    vertices = hcat(
        vec(parent(cart_coords.x1)),
        vec(parent(cart_coords.x2)),
        vec(parent(cart_coords.x3)),
    )
    triangles = hcat(ClimaCore.Spaces.triangulate(space)...)

    colors = vec(parent(field))
    Makie.mesh!(
        plot,
        vertices,
        triangles,
        color = colors,
        colormap = plot[:colormap],
        shading = false,
    )
end

end # module
