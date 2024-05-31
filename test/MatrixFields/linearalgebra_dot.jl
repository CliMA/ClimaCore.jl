# LinearAlgebra exports ⋅ and it is easy to get confused with MatrixFields.⋅.
# This test checks that we have a mechanism in place to inform the user when
# they are using the wrong ⋅.

import ClimaCore: Domains, Geometry, Meshes, Spaces, MatrixFields
import ClimaCore.MatrixFields: @name

import LinearAlgebra: ⋅, I

@testset "LinearAlgebra dot" begin
    n_elem_z = 2

    domain = Domains.IntervalDomain(Geometry.ZPoint(0.0), Geometry.ZPoint(1.0), boundary_names = (:bottom, :top))
    mesh = Meshes.IntervalMesh(domain, nelems = 2)
    space = Spaces.FaceFiniteDifferenceSpace(mesh)

    diverg = Operators.DivergenceC2F(; bottom = Operators.SetDivergence(0.0), top = Operators.SetDivergence(0.0))
    grad = Operators.GradientF2C()

    diverg_matrix = MatrixFields.operator_matrix(diverg)
    grad_matrix = MatrixFields.operator_matrix(grad)

    name = @name(u)
    jacobian = MatrixFields.FieldMatrix(
        (@name(u), @name(u)) => similar(zeros(space), ClimaCore.MatrixFields.TridiagonalMatrixRow{Float64}),
    )

    @test_throws ErrorException @. jacobian[name, name] = diverg_matrix() ⋅ grad_matrix() - (I,)
end
