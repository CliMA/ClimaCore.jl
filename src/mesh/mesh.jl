"""
    Meshes

- domain
- topology
- coordinates
- metric terms (inverse partial derivatives)
- quadrature rules and weights

## References / notes
 - [ceed](https://ceed.exascaleproject.org/ceed-code/)
 - [QA](https://github.com/CliMA/ClimateMachine.jl/blob/ans/sphere/test/Numerics/DGMethods/compressible_navier_stokes_equations/sphere/sphere_helper_functions.jl)

"""
module Meshes


import ..Geometry
import ..DataLayouts, ..Domains, ..Topologies
import ..slab
using StaticArrays, ForwardDiff, LinearAlgebra, UnPack

abstract type AbstractMesh end

include("quadrature.jl")
import .Quadratures

include("dss.jl")

# we need to be able figure out the coordinate of each node
# - element vertex location (come from the topology)
# - curvature: i.e. how do we interpolate within an element
#   - e.g. flat, spherical, "warp" function
#   - bilinear for flat, equiangular for spherical
#      - domain establishes the coordinate system used (cartesian of spherical)

struct Mesh2D{T, Q, C, G, IS, BS} <: AbstractMesh
    topology::T
    quadrature_style::Q
    coordinates::C
    local_geometry::G
    internal_surface_geometry::IS
    boundary_surface_geometries::BS
end

Topologies.nlocalelems(mesh::AbstractMesh) =
    Topologies.nlocalelems(mesh.topology)

undertype(::Type{Geometry.LocalGeometry{FT, M}}) where {FT, M} = FT
undertype(mesh::AbstractMesh) = undertype(eltype(mesh.local_geometry))

function Base.show(io::IO, mesh::Mesh2D)
    println(io, "Mesh2D:")
    println(io, "  topology: ", mesh.topology)
    println(io, "  quadrature: ", mesh.quadrature_style)
end

"""
    Mesh2D(topology, quadrature_style)

Construct a `Mesh2D` instance given a `topology` and `quadrature`.
"""
function Mesh2D(topology, quadrature_style)
    CT = Domains.coordinate_type(topology)
    FT = eltype(CT)
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    coordinates = DataLayouts.IJFH{CT, Nq}(Array{FT}, nelements)
    LG = Geometry.LocalGeometry{FT, SMatrix{2, 2, FT, 4}}

    local_geometry = DataLayouts.IJFH{LG, Nq}(Array{FT}, nelements)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for elem in 1:nelements
        coordinate_slab = slab(coordinates, elem)
        local_geometry_slab = slab(local_geometry, elem)
        for i in 1:Nq, j in 1:Nq
            # this hard-codes a bunch of assumptions, and will unnecesarily duplicate data
            # e.g. where all metric terms are uniform over the space
            # alternatively: move local_geometry to a different object entirely, to support overintegration
            # (where the integration is of different order)
            ξ = SVector(quad_points[i], quad_points[j])
            x = Geometry.interpolate(
                Topologies.vertex_coordinates(topology, elem),
                ξ[1],
                ξ[2],
            )
            ∂x∂ξ = ForwardDiff.jacobian(ξ) do ξ
                local x
                x = Geometry.interpolate(
                    Topologies.vertex_coordinates(topology, elem),
                    ξ[1],
                    ξ[2],
                )
                SVector(x.x1, x.x2)
            end
            J = det(∂x∂ξ)
            ∂ξ∂x = inv(∂x∂ξ)
            WJ = J * quad_weights[i] * quad_weights[j]

            coordinate_slab[i, j] = x
            # store WJ in invM slot
            local_geometry_slab[i, j] = Geometry.LocalGeometry(J, WJ, WJ, ∂ξ∂x)
        end
    end
    # compute invM from WJ:
    # M = dss(WJ)
    horizontal_dss!(local_geometry.invM, topology, Nq)
    # invM = inv.(M)
    local_geometry.invM .= inv.(local_geometry.invM)

    SG = Geometry.SurfaceGeometry{FT, Geometry.Cartesian12Vector{FT}}
    interior_faces = Topologies.interior_faces(topology)

    internal_surface_geometry =
        DataLayouts.IFH{SG, Nq}(Array{FT}, length(interior_faces))
    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(interior_faces)
        internal_surface_geometry_slab = slab(internal_surface_geometry, iface)

        local_geometry_slab⁻ = slab(local_geometry, elem⁻)
        local_geometry_slab⁺ = slab(local_geometry, elem⁺)

        for q in 1:Nq
            sgeom⁻ = compute_surface_geometry(
                local_geometry_slab⁻,
                quad_weights,
                face⁻,
                q,
                false,
            )
            sgeom⁺ = compute_surface_geometry(
                local_geometry_slab⁻,
                quad_weights,
                face⁺,
                q,
                false,
            )

            @assert sgeom⁻.sWJ ≈ sgeom⁺.sWJ
            @assert sgeom⁻.normal ≈ -sgeom⁺.normal

            internal_surface_geometry_slab[q] = sgeom⁻
        end
    end

    boundary_surface_geometries =
        map(Topologies.boundaries(topology)) do boundarytag
            boundary_faces = Topologies.boundary_faces(topology, boundarytag)
            boundary_surface_geometry =
                DataLayouts.IFH{SG, Nq}(Array{FT}, length(boundary_faces))
            for (iface, (elem, face)) in enumerate(boundary_faces)
                boundary_surface_geometry_slab =
                    slab(boundary_surface_geometry, iface)
                local_geometry_slab = slab(local_geometry, elem)
                for q in 1:Nq
                    boundary_surface_geometry_slab[q] =
                        compute_surface_geometry(
                            local_geometry_slab,
                            quad_weights,
                            face,
                            q,
                            false,
                        )
                end
            end
            boundary_surface_geometry
        end

    return Mesh2D(
        topology,
        quadrature_style,
        coordinates,
        local_geometry,
        internal_surface_geometry,
        boundary_surface_geometries,
    )
end


function compute_surface_geometry(
    local_geometry_slab,
    quad_weights,
    face,
    q,
    reversed = false,
)
    Nq = length(quad_weights)
    @assert size(local_geometry_slab) == (Nq, Nq)
    i, j = Topologies.face_node_index(face, Nq, q, reversed)

    local_geometry = local_geometry_slab[i, j]
    @unpack J, ∂ξ∂x = local_geometry

    # surface mass matrix
    n = if face == 1
        -J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 2
        J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 3
        -J * ∂ξ∂x[2, :] * quad_weights[i]
    elseif face == 4
        J * ∂ξ∂x[2, :] * quad_weights[i]
    end
    sWJ = norm(n)
    n = n / sWJ
    return Geometry.SurfaceGeometry(sWJ, Geometry.Cartesian12Vector(n...))
end



function variational_solve!(data, mesh::AbstractMesh)
    data .= mesh.local_geometry.invM .⊠ data
end



struct MeshSlab{Q, C, G} <: AbstractMesh
    quadrature_style::Q
    coordinates::C
    local_geometry::G
end
function slab(mesh::Mesh2D, h)
    MeshSlab(
        mesh.quadrature_style,
        slab(mesh.coordinates, h),
        slab(mesh.local_geometry, h),
    )
end


#=
struct Mesh3D{T,C} <: AbstractMesh
    topology::T
    vertical_discretization::C
end
=#
# some notion of a local geometry at each node, we need
#  - coordinates
#  - mechanism to convert vectors to contravariant (for divergences) and from covariant (for gradients)
#  - Jacobian determinant + inverse (Riemannian volume form on reference element)

#=

struct LocalGeometry{FT}
    x::SVector{3, FT}
    ∂ξ∂x::SMatrix{3, 3, FT, 9}
    J::FT
end
=#

include("column_mesh.jl")
include("hybrid_mesh.jl")

end # module
