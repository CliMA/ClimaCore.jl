"""
    Meshes

- domain
- topologyc
- coordinates
- metric terms (inverse partial derivatives)
- quadrature rules and weights

https://ceed.exascaleproject.org/ceed-code/

QA: https://github.com/CliMA/ClimateMachine.jl/blob/ans/sphere/test/Numerics/DGMethods/compressible_navier_stokes_equations/sphere/sphere_helper_functions.jl

"""
module Meshes

include("quadrature.jl")

import ..DataLayouts, ..Domains, ..Topologies
import ..slab
import .Quadratures
using StaticArrays, ForwardDiff, LinearAlgebra

abstract type AbstractMesh end

# we need to be able figure out the coordinate of each node
# - element vertex location (come from the topology)
# - curvature: i.e. how do we interpolate within an element
#   - e.g. flat, spherical, "warp" function
#   - bilinear for flat, equiangular for spherical
#      - domain establishes the coordinate system used (cartesian of spherical)

"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{FT, M}
    "Jacobian determinant of the transformation `ξ` to `x`"
    J::FT
    "Metric: `J` multiplied by the quadrature weights"
    M::FT
    "inverse Metric terms: `1/M`"
    invM::FT
    "Partial derivatives of the map from `x` to `ξ`"
    ∂ξ∂x::M
end



struct Mesh2D{T, Q, C, G} <: AbstractMesh
    topology::T
    quadrature_style::Q
    coordinates::C
    local_geometry::G
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

    LG = LocalGeometry{FT, SMatrix{2, 2, FT, 4}}

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
            x = Domains.interpolate(
                Topologies.vertex_coordinates(topology, elem),
                ξ[1],
                ξ[2],
            )
            ∂x∂ξ = ForwardDiff.jacobian(ξ) do ξ
                Domains.interpolate(
                    Topologies.vertex_coordinates(topology, elem),
                    ξ[1],
                    ξ[2],
                )
            end
            J = det(∂x∂ξ)
            ∂ξ∂x = inv(∂x∂ξ)
            M = J * quad_weights[i] * quad_weights[j]
            invM = 1 / M

            coordinate_slab[i, j] = x
            local_geometry_slab[i, j] = LocalGeometry(J, M, invM, ∂ξ∂x)
        end
    end
    return Mesh2D(topology, quadrature_style, coordinates, local_geometry)
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
end # module
