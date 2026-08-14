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
    VIJH = ClimaCore.DataLayouts.VIJFH
    helem = 32
    Nq = 2
    # Low resolution does not use eager eval on gpu for now
    for z_elems in (10, 20)
        cspace = TU.CenterExtrudedFiniteDifferenceSpace(
            FT;
            zelem = z_elems,
            helem,
            Nq,
            VIJH,
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

# A level field has no vertical dimension, and a column field has no horizontal
# dimensions; both hold a single value along the dimensions they are missing,
# and are broadcast across them. On GPUs the spaces of a broadcast's arguments
# are replaced by placeholders before the launch and rebuilt inside the kernel,
# so such an argument is easy to index as though it spanned the whole extruded
# space, which reads outside of its data.
@testset "Reduced-dimension arguments of a finite difference stencil" begin
    FT = Float64
    helem = 4
    Nq = 2
    # 10 z elements use the generic stencil kernel on GPUs, 20 use the eager one
    for z_elems in (10, 20)
        fspace =
            TU.FaceExtrudedFiniteDifferenceSpace(FT; zelem = z_elems, helem, Nq)
        grad = Operators.GradientF2C()
        coords = ClimaCore.Fields.coordinate_field(fspace)
        z = coords.z
        ∇z = @. grad(z)

        # a level field is constant in z, so it cannot change a vertical gradient
        lat = ClimaCore.Fields.level(coords, 1).lat
        @test parent(@. grad(z + lat)) ≈ parent(∇z)

        # a column field of z is equal to z in every column
        z_column = ClimaCore.Fields.column(z, 1, 1, 1)
        @test parent(@. grad(z + z_column)) ≈ 2 .* parent(∇z)
    end
end
