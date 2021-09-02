module Meshes
using DocStringExtensions
export EquispacedRectangleMesh, rectangular_mesh, cube_panel_mesh

import ..Domains: IntervalDomain, RectangleDomain, SphereDomain

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
    Mesh2D{I,IA2D,FT,FTA2D} <: AbstractMesh{FT}

Conformal mesh for a 2D manifold. The manifold can be 
embedded in a higher dimensional space.

                        Quadrilateral

                v3            f4           v4
                  o------------------------o
                  |                        |		  face    vertices
                  |                        |             
                  |                        |		   f1 =>  v1 v3 
               f1 |                        | f2        f2 =>  v2 v4
                  |                        |		   f3 =>  v1 v2
                  |                        |           f4 =>  v3 v4
                  |                        |
                  |                        |
                  o------------------------o
                 v1           f3           v2

z-order numbering convention for 2D quadtrees

Reference:

p4est: SCALABLE ALGORITHMS FOR PARALLEL ADAPTIVE
MESH REFINEMENT ON FORESTS OF OCTREES∗
CARSTEN BURSTEDDE†, LUCAS C. WILCOX‡ , AND OMAR GHATTAS§
SIAM J. Sci. Comput. Vol. 33, No. 3, pp. 1103-1133

https://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Mesh2D{I,IA1D,IA2D,FT,FTA2D} <: AbstractMesh{FT}
    "# of unique nodes in the mesh"
    nverts::I
    "# of unique faces in the mesh"
    nfaces::I
    "# of elements in the mesh"
    nelems::I
    "# of zones in the mesh"
    nbndry::I
    "x₁, x₂, ... coordinates of nodes `(nverts, dim)`, dim can be greater than 2 for 2D manifolds embedded in higher dimensional space"
    coordinates::FTA2D
    "face node numbers `(nfaces, 2)`"
    face_verts::IA2D
    "boundary elems for each face `(nfaces, 2)`"
    face_neighbors::IA2D
    "face zones for each face `(nfaces, 1)`"
    face_bndry::IA1D
    "node numbers for each elem `(nelems, 4)`"
    elem_verts::IA2D
    "face numbers for each elem `(nelems, 4)`"
    elem_faces::IA2D
end

function Mesh2D(
    nverts,
    nfaces,
    nelems,
    nbndry,
    coordinates,
    face_verts,
    face_neighbors,
    face_bndry,
    elem_verts,
    elem_faces,
)

    return Mesh2D{
        eltype(nverts),
        typeof(face_bndry),
        typeof(face_verts),
        eltype(coordinates),
        typeof(coordinates),
    }(
        nverts,
        nfaces,
        nelems,
        nbndry,
        coordinates,
        face_verts,
        face_neighbors,
        face_bndry,
        elem_verts,
        elem_faces,
    )
end

include("BoxMesh.jl")
include("CubedSphereMesh.jl")

end # module
