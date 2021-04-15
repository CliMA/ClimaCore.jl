using Tests
import ClimatemachineCore.Mesh.Quadrature

@testset "test LGL quadrature" begin
    quad = Quadrature.LGL{4}
    @test quad isa Quadrature.QuadratureStyle
    @test Quadrature.polynomial_degree(quad) == 3
    @test Quadrature.degrees_of_freedom(quad) == 4
    @test length(Quadrature.quadrature_points(quad)) == 4
end

@testset "test GL quadrature" begin
    quad = Quadrature.GL{4}
    @test quad isa Quadrature.QuadratureStyle
    @test Quadrature.polynomial_degree(quad) == 3
    @test Quadrature.degrees_of_freedom(quad) == 4
    @test length(Quadrature.quadrature_points(quad)) == 4
end

