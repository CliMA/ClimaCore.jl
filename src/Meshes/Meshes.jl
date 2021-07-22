module Meshes

using Base: Bool
export EquispacedRectangleMesh
export TensorProductMesh

import ..Domains: IntervalDomain, RectangleDomain, SphereDomain
import ..Geometry: Cartesian2DPoint

"""
    AbstractMesh

A `Mesh` is an object which represents how we discretize a domain into elements.

It should be lightweight (i.e. exists on all MPI ranks), e.g for meshes stored
in a file, it would contain the filename.
"""
abstract type AbstractMesh{FT} end

Base.eltype(::AbstractMesh{FT}) where {FT} = FT

warp_mesh(mesh::AbstractMesh) = mesh

struct IntervalMesh{FT, I <: IntervalDomain, V <: AbstractVector, B} <:
       AbstractMesh{FT}
    domain::I
    faces::V
    boundaries::B
end

IntervalMesh{FT}(domain::I, faces::V, boundaries::B) where {FT, I, V, B} =
    IntervalMesh{FT, I, V, B}(domain, faces, boundaries)

abstract type Stretching end

struct Uniform <: Stretching end

function IntervalMesh(domain::IntervalDomain{FT}, ::Uniform; nelems) where {FT}
    faces = range(domain.x3min, domain.x3max; length = nelems + 1)
    boundaries = NamedTuple{domain.x3boundary}((5, 6))
    IntervalMesh{FT}(domain, faces, boundaries)
end

# 3.1.2 in the design docs
"""
    ExponentialStretching(H)

Apply exponential stretching to the  domain. `H` is the scale height (a typical atmospheric scale height `H ≈ 7.5e3`km).
"""
struct ExponentialStretching{FT} <: Stretching
    H::FT
end

function IntervalMesh(
    domain::IntervalDomain{FT},
    stretch::ExponentialStretching;
    nelems,
) where {FT}
    R = domain.x3max - domain.x3min
    h = stretch.H / R
    η(ζ) = -h * log1p(-(1 - exp(-1 / h)) * ζ)
    faces = [
        domain.x3min + R * η(ζ) for
        ζ in range(FT(0), FT(1); length = nelems + 1)
    ]
    boundaries = NamedTuple{domain.x3boundary}((5, 6))
    IntervalMesh{FT, typeof(domain), typeof(faces), typeof(boundaries)}(
        domain,
        faces,
        boundaries,
    )
end

IntervalMesh(domain::IntervalDomain; nelems) =
    IntervalMesh(domain, Uniform(); nelems)

function Base.show(io::IO, mesh::IntervalMesh)
    nelements = length(mesh.faces) - 1
    print(io, nelements, " IntervalMesh of ")
    print(io, mesh.domain)
end


struct EquispacedLineMesh{FT, ID <: IntervalDomain{FT}, R} <: AbstractMesh{FT}
    domain::ID
    n1::Int64 # number of elements in x1 direction
    n2::Int64 # always 1
    range1::R
    range2::R # always 1:1
end

function EquispacedLineMesh(domain::IntervalDomain, n1)
    range1 = range(domain.x3min, domain.x3max; length = n1 + 1)
    range2 = range(
        one(domain.x3min),
        one(domain.x3max) + one(domain.x3max);
        length = 2,
    )
    return EquispacedLineMesh(domain, n1, one(n1), range1, range2)
end

function Base.show(io::IO, mesh::EquispacedLineMesh)
    print(io, "(", mesh.n1, " × ", " ) EquispacedLineMesh of ")
    print(io, mesh.domain)
end

"""
    EquispacedRectangleMesh(domain::RectangleDomain, n1::Integer, n2::Integer)

A regular `AbstractMesh` of `domain` with `n1` elements in dimension 1, and `n2`
in dimension 2.
"""
struct EquispacedRectangleMesh{FT, RD <: RectangleDomain{FT}, R} <:
       AbstractMesh{FT}
    domain::RD
    n1::Int64 # number of elements in x1 direction
    n2::Int64 # number of elements in x2 direction
    range1::R
    range2::R
end

function EquispacedRectangleMesh(domain::RectangleDomain, n1, n2)
    range1 = range(domain.x1min, domain.x1max; length = n1 + 1)
    range2 = range(domain.x2min, domain.x2max; length = n2 + 1)
    EquispacedRectangleMesh(domain, n1, n2, range1, range2)
end

function Base.show(io::IO, mesh::EquispacedRectangleMesh)
    print(io, mesh.n1, "×", mesh.n2, " EquispacedRectangleMesh of ")
    print(io, mesh.domain)
end

struct EquiangularCubedSphereMesh{FT} <: AbstractMesh{FT}
    domain::SphereDomain{FT}
    n::Int64
end

"""
    TensorProductMesh(domain::RectangleDomain, n::Integer)

A tensor-product `AbstractMesh` of `domain` with `n1` elements in dimension 1, and `n2`
in dimension 2.
"""
struct TensorProductMesh{FT, RD <: RectangleDomain{FT}} <: AbstractMesh{FT}
    domain::RD
    n1::Int64 # number of elements in x1 direction
    n2::Int64 # number of elements in x2 direction
    faces::Vector{Tuple{Int64, Int64, Int64, Int64, Bool}}
    coordinates::Vector{Cartesian2DPoint{FT}}
end

function TensorProductMesh(
    domain::RectangleDomain{FT},
    n1,
    n2,
    coordinates = nothing,
) where {FT}

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

        coordinates = Vector{Cartesian2DPoint{FT}}(undef, (n1 + 1) * (n2 + 1))
        # Default equispaced vertex coordinates, if the user has not specified their locations
        range1 = range(domain.x1min, domain.x1max; length = n1 + 1)
        range2 = range(domain.x2min, domain.x2max; length = n2 + 1)

        # Coordinates array, row-major storage
        for i in 1:(n1 + 1)
            for j in 1:(n2 + 1)
                coordinates[(i - 1) * (n2 + 1) + j] =
                    Cartesian2DPoint(range1[i], range2[j])
            end
        end
    end
    TensorProductMesh(domain, n1, n2, faces, coordinates)
end

function Base.show(io::IO, mesh::TensorProductMesh)
    print(io, mesh.n1, "×", mesh.n2, " TensorProductMesh of ")
    print(io, mesh.domain)
end

end # module
