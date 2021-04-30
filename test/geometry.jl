using Test
using ClimateMachineCore.Geometry

@testset "Vectors" begin
    wᵢ = Geometry.Covariant12Vector(1.0,2.0)
    vʲ = Geometry.Contravariant12Vector(3.0,4.0)

    @test wᵢ[1] == 1.0
    @test wᵢ[2] == 2.0
    @test vʲ[1] == 3.0
    @test vʲ[2] == 4.0

    @test wᵢ .* 3 === Geometry.Covariant12Vector(3.0,6.0)
    @test wᵢ .+ wᵢ === Geometry.Covariant12Vector(2.0,4.0)


    uᵏ = Geometry.Contravariant12Vector(1.0,2.0)
    @test_throws DimensionMismatch wᵢ .* uᵏ

    T = uᵏ ⊗ uᵏ
    @test uᵏ ⊗ uᵏ isa Geometry.Tensor{Geometry.Contravariant12Vector{Float64}, Geometry.Contravariant12Vector{Float64}}

    @test T * wᵢ === Geometry.Contravariant12Vector(5.0, 10.0)
    @test_throws DimensionMismatch T * uᵏ


end