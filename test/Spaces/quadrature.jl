using Test
using LinearAlgebra, StaticArrays
import ClimaCore.Spaces: Quadratures


f(x) = x^3 + 2x^2 + 3x + 4
fd(x) = 3x^2 + 4x + 3

@testset "test LGL quadrature" begin
    quad = Quadratures.GLL{3}()
    @test quad isa Quadratures.QuadratureStyle
    @test Quadratures.polynomial_degree(quad) == 2
    @test Quadratures.degrees_of_freedom(quad) == 3

    points, weights = Quadratures.quadrature_points(Float64, quad)

    @test eltype(points) === eltype(weights) === Float64
    @test length(points) == length(weights) == 3
    @test dot(f.(points), weights) ≈ 28 / 3
end

@testset "test GL quadrature" begin
    quad = Quadratures.GL{4}()
    @test quad isa Quadratures.QuadratureStyle
    @test Quadratures.polynomial_degree(quad) == 3
    @test Quadratures.degrees_of_freedom(quad) == 4

    points, weights = Quadratures.quadrature_points(Float32, quad)

    @test eltype(points) === eltype(weights) === Float32
    @test length(points) == length(weights) == 4
    @test dot(f.(points), weights) ≈ 28 / 3
end

@testset "differentiation_matrix" begin
    quad = Quadratures.GL{4}()
    points, weights = Quadratures.quadrature_points(Float64, quad)

    D = Quadratures.differentiation_matrix(Float64, quad)
    @test D isa SMatrix
    @test size(D) == (4, 4)

    @test D * f.(points) ≈ fd.(points)

    quad = Quadratures.GLL{4}()
    points, weights = Quadratures.quadrature_points(Float64, quad)

    D = Quadratures.differentiation_matrix(Float64, quad)
    @test D isa SMatrix
    @test size(D) == (4, 4)

    @test D * f.(points) ≈ fd.(points)
end

@testset "interpolation matrix" begin
    quad1 = Quadratures.GL{4}()
    quad2 = Quadratures.GLL{5}()
    points1, weights1 = Quadratures.quadrature_points(Float64, quad1)
    points2, weights2 = Quadratures.quadrature_points(Float64, quad2)
    I = Quadratures.interpolation_matrix(Float64, quad2, quad1)
    @test size(I) == (5, 4)

    @test I * f.(points1) ≈ f.(points2)

end
