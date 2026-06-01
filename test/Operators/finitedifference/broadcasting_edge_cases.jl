# This file contains tests for edge cases in broadcasting behavior of finite difference operators,
# particularly in the context of GPU compilation.

using ClimaCore: Geometry, Operators, MatrixFields
import ClimaCore
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;
using Test
using ClimaComms
import ClimaCore.MatrixFields: ⋅
import LinearAlgebra: I
ClimaComms.@import_required_backends

@testset "Combined stencil and poinstwise with types in broadcasted args" begin
    FT = Float32
    horizontal_layout_type = ClimaCore.DataLayouts.IJFH
    helem = 32
    Nq = 2
    # very low resolution does not use eager eval on gpu for now
    for z_elems in (10, 20)
        cspace = TU.CenterExtrudedFiniteDifferenceSpace(
            FT;
            zelem = z_elems,
            helem,
            Nq,
            horizontal_layout_type,
        )
        fspace = ClimaCore.Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
        divf2c_op = Operators.DivergenceF2C()
        divf2c_matrix = MatrixFields.operator_matrix(divf2c_op)
        full_bidiag_matrix_scratch = fill(
            zero(MatrixFields.BidiagonalMatrixRow{Geometry.Covariant3Vector{FT}}),
            fspace,
        )
        dtγ = FT(1)
        out = @. FT(-1) * float(dtγ) * (divf2c_matrix() ⋅ full_bidiag_matrix_scratch) - (I,)
        expected_result =
            fill(MatrixFields.TridiagonalMatrixRow(0.0f0, -1.0f0, 0.0f0), cspace)
        @test out == expected_result
    end
end
