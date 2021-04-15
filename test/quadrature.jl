using Test
import ClimateMachineCore.Meshes.Quadratures

@testset "test LGL quadrature" begin
    quad = Quadratures.GLL{3}()
    @test quad isa Quadratures.QuadratureStyle
    @test Quadratures.polynomial_degree(quad) == 2
    @test Quadratures.degrees_of_freedom(quad) == 3
    points, weights = Quadratures.quadrature_points(Float64, quad)

    @test eltype(points) === eltype(weights) === Float64
    @test length(points) == length(weights) == 3
end

@testset "test GL quadrature" begin
    quad = Quadratures.GL{4}()
    @test quad isa Quadratures.QuadratureStyle
    @test Quadratures.polynomial_degree(quad) == 3
    @test Quadratures.degrees_of_freedom(quad) == 4

    points, weights = Quadratures.quadrature_points(Float32, quad)
    @test eltype(points) === eltype(weights) === Float32
    @test length(points) == length(weights) == 4
end
