#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Geometry", "contravariant3_projection.jl"))
=#
using Test
import Random
using StaticArrays: SMatrix, @SMatrix, SVector
using LinearAlgebra: det, I

import ClimaCore.Geometry
const G = Geometry

# Build a LocalGeometry from a full 3×3 ∂x∂ξ (Orthonormal ← Covariant).
function make_lg(M::SMatrix{3, 3, FT}) where {FT}
    coord = G.LatLongZPoint(FT(0), FT(0), FT(0))
    ∂x∂ξ = G.Tensor(
        M,
        (
            G.Components{G.Orthonormal, (1, 2, 3)}(),
            G.Components{G.Covariant, (1, 2, 3)}(),
        ),
    )
    return G.LocalGeometry(coord, det(M), det(M), ∂x∂ξ)
end

# Reference for the vertical physical→contravariant projection: the generic path
# materializes the full inverse `lg.∂ξ∂x` and extracts entry (3,3).
ref_contravariant3(lg, w) = parent(lg.∂ξ∂x)[3, 3] * w

@testset "fast contravariant3 projection of a WVector" begin
    Random.seed!(42)

    # The specialized `project(Contravariant3Axis, WVector, lg)` must be selected.
    lg0 = make_lg(SMatrix{3, 3, Float64}([2 0.1 0.5; 0.2 2 0.3; 0.7 0.4 3]))
    m = @which G.project(G.Contravariant3Axis(), G.WVector(1.0), lg0)
    @test occursin("conversions.jl", String(m.file))

    for FT in (Float64, Float32)
        rtol = FT === Float64 ? 1e-12 : 1e-4
        for _ in 1:2000
            # well-conditioned metrics, incl. terrain-following coupling (∂xʰ/∂ξ³ ≠ 0)
            M = SMatrix{3, 3, FT}(3I) + SMatrix{3, 3, FT}(randn(FT, 3, 3))
            lg = make_lg(M)
            w = randn(FT)
            fast = G.project(G.Contravariant3Axis(), G.WVector(w), lg)[1]
            @test fast ≈ ref_contravariant3(lg, w) rtol = rtol
            # equivalently, exactly the (3,3) entry of inv(∂x∂ξ) times w
            @test fast ≈ w * inv(M)[3, 3] rtol = rtol
        end

        # `contravariant3` / `Jcontravariant3` scalar extractors use the same path
        M = SMatrix{3, 3, FT}([2 FT(0.1) FT(0.5); FT(0.2) 2 FT(0.3); FT(0.7) FT(0.4) 3])
        lg = make_lg(M)
        @test G.contravariant3(G.WVector(FT(1.5)), lg) ≈ FT(1.5) * inv(M)[3, 3]
        @test G.Jcontravariant3(G.WVector(FT(1.5)), lg) ≈ lg.J * FT(1.5) * inv(M)[3, 3]

        # type stability
        @test (@inferred G.project(G.Contravariant3Axis(), G.WVector(FT(1)), lg)) isa
              G.Contravariant3Vector{FT}
    end

    # The specialization must NOT change other projections:
    lg = make_lg(SMatrix{3, 3, Float64}([2 0.1 0.5; 0.2 2 0.3; 0.7 0.4 3]))
    M = parent(getfield(lg, :∂x∂ξ))
    # a non-vertical orthonormal (UVW) source still uses the full metric (row 3 of ∂ξ∂x)
    uvw = G.UVWVector(0.3, -0.7, 1.1)
    @test G.project(G.Contravariant3Axis(), uvw, lg)[1] ≈
          (inv(M) * SVector(0.3, -0.7, 1.1))[3]
    # a contravariant source needs no metric
    @test G.project(G.Contravariant3Axis(), G.Contravariant123Vector(1.0, 2.0, 3.0), lg)[1] == 3.0
end
