using Test
using ClimateMachineCore.Geometry
using LinearAlgebra, UnPack, StaticArrays

@testset "Vectors" begin
    wᵢ = Geometry.Covariant12Vector(1.0, 2.0)
    vʲ = Geometry.Contravariant12Vector(3.0, 4.0)

    @test wᵢ[1] == 1.0
    @test wᵢ[2] == 2.0
    @test vʲ[1] == 3.0
    @test vʲ[2] == 4.0

    @test wᵢ .* 3 === Geometry.Covariant12Vector(3.0, 6.0)
    @test wᵢ .+ wᵢ === Geometry.Covariant12Vector(2.0, 4.0)


    uᵏ = Geometry.Contravariant12Vector(1.0, 2.0)
    @test_throws DimensionMismatch wᵢ .* uᵏ

    T = uᵏ ⊗ uᵏ
    @test uᵏ ⊗ uᵏ isa Geometry.Tensor{
        Geometry.Contravariant12Vector{Float64},
        Geometry.Contravariant12Vector{Float64},
    }

    @test T * wᵢ === Geometry.Contravariant12Vector(5.0, 10.0)
    @test_throws DimensionMismatch T * uᵏ


end

@testset "Sample flux calculation" begin
    state = (ρ = 2.0, ρu = Cartesian12Vector(1.0, 2.0), ρθ = 0.5)

    function flux(state, g)
        @unpack ρ, ρu, ρθ = state

        u = ρu ./ ρ

        return (ρ = ρu, ρu = (ρu ⊗ u) + (g * ρ^2 / 2) * I, ρθ = ρθ .* u)
    end

    @test flux(state, 10.0) === (
        ρ = Cartesian12Vector(1.0, 2.0),
        ρu = Tensor{Cartesian12Vector{Float64}, Cartesian12Vector{Float64}}(
            SMatrix{2, 2}(0.5 + 20.0, 1.0, 1.0, 2.0 + 20.0),
        ),
        ρθ = Cartesian12Vector(0.25, 0.5),
    )
end
