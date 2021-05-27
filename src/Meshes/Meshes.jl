module Meshes

export EquispacedRectangleMesh

import ..Domains: RectangleDomain, SphereDomain

"""
    AbstractMesh

A `Mesh` is an object which represents how we discretize a domain into elements.

It should be lightweight (i.e. exists on all MPI ranks), e.g for meshes stored
in a file, it would contain the filename.
"""
abstract type AbstractMesh end


"""
    EquispacedRectangleMesh(domain::RectangleDomain, n1::Integer, n2::Integer)

A regular `AbstractMesh` of `domain` with `n1` elements in dimension 1, and `n2`
in dimension 2.
"""
struct EquispacedRectangleMesh{FT, RD <: RectangleDomain{FT}, R} <: AbstractMesh
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
Base.eltype(::EquispacedRectangleMesh{FT}) where {FT} = FT
function Base.show(io::IO, disc::EquispacedRectangleMesh)
    print(io, disc.n1, "×", disc.n2, " EquispacedRectangleMesh of ")
    print(io, disc.domain)
end




struct EquiangularCubedSphereMesh{FT} <: AbstractMesh
    domain::SphereDomain{FT}
    n::Int64
    # 6*n^2 elements
end
end # module
