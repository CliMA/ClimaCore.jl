"""
    TensorProductMesh(domain::RectangleDomain, n1::Integer, n2::Integer, coordinates::Vector{<:Abstract2DPoint})

A tensor-product `AbstractMesh` of `domain` with `n1` elements in dimension 1, and `n2`
in dimension 2. `coordinates` is an optional argument that can be used to specify the coordinates vector. If `coordinates = nothing`, mesh vertices are defaulted to be equispaced.
"""
struct TensorProductMesh{
    FT,
    CT <: Geometry.Abstract2DPoint{FT},
    RD <: RectangleDomain{CT},
} <: AbstractMesh{FT}
    domain::RD
    n1::Int64 # number of elements in x1 direction
    n2::Int64 # number of elements in x2 direction
    faces::Vector{Tuple{Int64, Int64, Int64, Int64, Bool}}
    coordinates::Vector{CT}
end

function TensorProductMesh(
    domain::RectangleDomain,
    n1,
    n2,
    coordinates = nothing,
)

    nelem = n1 * n2
    x1periodic = isnothing(domain.x1boundary)
    x2periodic = isnothing(domain.x2boundary)
    faces = Vector{Tuple{Int64, Int64, Int64, Int64, Bool}}(undef, nelem * 4)

    # Store all mesh faces as (elem, face, opelem, opface, reversed)
    # so that we can determine face pairs via the map (elem, face) to neighbouring (opelem, opface, reversed)
    for e in 1:nelem
        z2s, z1s = fldmod(e - 1, n1)

        # Face 1
        z1 = z1s - 1
        z2 = z2s
        if z1 < 0 && !x1periodic
            faces[(e - 1) * 4 + 1] = (e, 1, 0, 1, false)
        else
            if z1 < 0
                z1 += n1
            end
            opface = 2
            opelem = z2 * n1 + z1 + 1
            faces[(e - 1) * 4 + 1] = (e, 1, opelem, opface, false)
        end

        # Face 2
        z1 = z1s + 1
        if z1 == n1 && !x1periodic
            faces[(e - 1) * 4 + 2] = (e, 2, 0, 2, false)
        else
            if z1 == n1
                z1 -= n1
            end
            opface = 1
            opelem = z2 * n1 + z1 + 1
            faces[(e - 1) * 4 + 2] = (e, 2, opelem, opface, false)
        end

        # Face 3
        z1 = z1s
        z2 = z2s - 1
        if z2 < 0 && !x2periodic
            faces[(e - 1) * 4 + 3] = (e, 3, 0, 3, false)
        else
            if z2 < 0
                z2 += n2
            end
            opface = 4
            opelem = z2 * n1 + z1 + 1
            faces[(e - 1) * 4 + 3] = (e, 3, opelem, opface, false)
        end

        # Face 4
        z2 = z2s + 1
        if z2 == n2 && !x2periodic
            faces[(e - 1) * 4 + 4] = (e, 4, 0, 4, false)
        else
            if z2 == n2
                z2 -= n2
            end
            opface = 3
            opelem = z2 * n1 + z1 + 1
            faces[(e - 1) * 4 + 4] = (e, 4, opelem, opface, false)
        end
    end

    if isnothing(coordinates)
        CT = Domains.coordinate_type(domain)
        coordinates = Vector{CT}(undef, (n1 + 1) * (n2 + 1))
        x1min = Geometry.component(domain.x1x2min, 1)
        x2min = Geometry.component(domain.x1x2min, 2)
        x1max = Geometry.component(domain.x1x2max, 1)
        x2max = Geometry.component(domain.x1x2max, 2)
        # Default equispaced vertex coordinates, if the user has not specified their locations
        range1 = range(x1min, x1max; length = n1 + 1)
        range2 = range(x2min, x2max; length = n2 + 1)
        # Coordinates array, row-major storage
        for i in 1:(n1 + 1)
            for j in 1:(n2 + 1)
                coordinates[(i - 1) * (n2 + 1) + j] = CT(range1[i], range2[j])
            end
        end
    end
    TensorProductMesh(domain, n1, n2, faces, coordinates)
end

function Base.show(io::IO, mesh::TensorProductMesh)
    print(io, mesh.n1, "×", mesh.n2, " TensorProductMesh of ")
    print(io, mesh.domain)
end
