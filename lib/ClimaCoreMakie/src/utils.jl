"""
    plot_vertices(space::AbstractSpace[, coords])

Compute the vertices of `space`` for plotting, returning a vector of `Point2f`
or `Point3f`.

`coords` can be one of the following:
- `field` containing the coordinates of the vertices (default is to use `Fields.coordinate_field(space)`)
- a tuple of scalar fields, one for each coordinate
- a type which 
"""

function plot_vertices(space::ClimaCore.Spaces.AbstractSpace, ::Nothing)
    plot_vertices(space, ClimaCore.Fields.coordinate_field(space))
end

function plot_vertices(
    space::ClimaCore.Spaces.AbstractSpace,
    ::Type{ClimaCore.Geometry.Cartesian123Point},
)
    plot_vertices(
        space,
        ClimaCore.Geometry.Cartesian123Point.(
            ClimaCore.Fields.coordinate_field(space),
            Ref(Spaces.global_geometry(space)),
        ),
    )
end
function plot_vertices(
    space::ClimaCore.Spaces.AbstractSpace,
    field::ClimaCore.Fields.Field,
)
    T = eltype(field)
    if T <: ClimaCore.Geometry.LatLongPoint ||
       T <: ClimaCore.Geometry.LatLongZPoint
        return plot_vertices(space, (field.long, field.lat))
    end
    n = fieldcount(T)
    coords = ntuple(i -> getproperty(field, i), n)
    plot_vertices(space, coords)
end

function plot_vertices(
    space::ClimaCore.Spaces.AbstractSpace,
    coords::NTuple{N},
) where {N}
    Makie.Pointf{N}.(map(vec ∘ parent, coords)...)
end


function plot_vertices(
    space::ClimaCore.Spaces.CenterExtrudedFiniteDifferenceSpace,
    ::Nothing,
)
    coords = ClimaCore.Spaces.coordinates_data(space)
    vertices =
        Makie.Point2f.(
            parent(getproperty(coords, 1)),
            parent(getproperty(coords, 2)),
        )

    # we extend the extrema to the boundaries
    face_space = ClimaCore.Spaces.FaceExtrudedFiniteDifferenceSpace(space)
    face_coords = ClimaCore.Spaces.coordinates_data(face_space)
    nf = ClimaCore.Spaces.nlevels(face_space)
    bottom_coords = ClimaCore.level(face_coords, 1)
    vertices[1, :, :, :] =
        Makie.Point2f.(
            parent(getproperty(bottom_coords, 1)),
            parent(getproperty(bottom_coords, 2)),
        )
    top_coords = ClimaCore.level(face_coords, nf)
    vertices[end, :, :, :] =
        Makie.Point2f.(
            parent(getproperty(top_coords, 1)),
            parent(getproperty(top_coords, 2)),
        )
    return vec(vertices)
end



"""
    plot_triangles(space::AbstractSpace)

Return a triangulation of `space`, as a vector of `GLTriangleFace`s.
"""
function plot_triangles(space::ClimaCore.Spaces.SpectralElementSpace2D)
    (Ni, Nj, _, _, Nh) = size(ClimaCore.Spaces.local_geometry_data(space))
    a, b, c = ClimaCore.Spaces.triangles(Ni, Nj, Nh)
    return GLTriangleFace.(a, b, c)
end
function plot_triangles(space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace)
    (Ni, _, _, Nv, Nh) = size(ClimaCore.Spaces.local_geometry_data(space))
    a, b, c = ClimaCore.Spaces.triangles(Nv, Ni, Nh)
    return GLTriangleFace.(a, b, c)
end

"""
    plot_triangles_matrix(space::AbstractSpace)

Return a triangulation of `space`, as an `3 x n` `Matrix{Int}`
"""
function plot_triangles_matrix(space::ClimaCore.Spaces.SpectralElementSpace2D)
    (Ni, Nj, _, _, Nh) = size(ClimaCore.Spaces.local_geometry_data(space))
    a, b, c = ClimaCore.Spaces.triangles(Ni, Nj, Nh)
    return vcat(a', b', c')
end
function plot_triangles_matrix(
    space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace,
)
    (Ni, _, _, Nv, Nh) = size(ClimaCore.Spaces.local_geometry_data(space))
    a, b, c = ClimaCore.Spaces.triangles(Nv, Ni, Nh)
    return vcat(a', b', c')
end
