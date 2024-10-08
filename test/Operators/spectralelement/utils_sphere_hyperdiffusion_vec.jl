using Test
using ClimaComms
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJHF
import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures
using StaticArrays, IntervalSets, LinearAlgebra

include("sphere_sphericalharmonics.jl")

function ∇⁴(u)
    scurl = Operators.Curl()
    sdiv = Operators.Divergence()
    wcurl = Operators.WeakCurl()
    wgrad = Operators.WeakGradient()

    χ = Spaces.weighted_dss!(
        @. wgrad(sdiv(u)) - Geometry.Covariant12Vector(
            wcurl(Geometry.Covariant3Vector(scurl(u))),
        )
    )

    ∇⁴ = Spaces.weighted_dss!(
        @. wgrad(sdiv(χ)) - Geometry.Covariant12Vector(
            wcurl(Geometry.Covariant3Vector(scurl(χ))),
        )
    )

    return ∇⁴
end
