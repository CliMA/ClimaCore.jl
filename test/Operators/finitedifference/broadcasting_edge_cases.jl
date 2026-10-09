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
        out = @. FT(-1) * float(dtγ) * (divf2c_matrix() * full_bidiag_matrix_scratch) - (I,)
        expected_result =
            fill(MatrixFields.TridiagonalMatrixRow(0.0f0, -1.0f0, 0.0f0), cspace)
        @test out == expected_result
    end
end

# A level field has no vertical dimension, and a column field has no horizontal
# dimensions; both hold a single value along the dimensions they are missing,
# and are broadcast across them.
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

# Base flattens the second operand of .&& or .|| into a closure that captures
# it, so a pointwise operand is flattened without capturing its fields instead,
# and an operator broadcast is kept as an operand; an operator broadcast in the
# first operand is always an operand.
@testset "Short-circuiting broadcasts of finite difference operators" begin
    FT = Float64
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 10, helem = 4, Nq = 2)
    fspace = ClimaCore.Spaces.face_space(cspace)
    interp = Operators.InterpolateF2C()
    c = ClimaCore.Fields.coordinate_field(cspace).z
    f = ClimaCore.Fields.coordinate_field(fspace).z
    op_bool = @. interp(f) > 1 / 2
    c_bool = @. c < 3 / 4
    @test parent(@. (interp(f) > 1 / 2) && (c < 3 / 4)) == parent(op_bool .& c_bool)
    @test parent(@. (c < 3 / 4) && (interp(f) > 1 / 2)) == parent(c_bool .& op_bool)
    @test parent(@. (interp(f) > 1 / 2) || c_bool) == parent(op_bool .| c_bool)
    @test parent(@. c_bool || (interp(f) > 1 / 2)) == parent(c_bool .| op_bool)

    # A pointwise second operand is only evaluated where the first operand does
    # not determine the result (sqrt(-c) throws a DomainError wherever c > 0).
    @test !any(parent(@. (c < 0) && (sqrt(-c) > 0)))
    @test all(parent(@. (c >= 0) || (sqrt(-c) > 0)))
end

@testset "Stencil nested in the argument of a Dirichlet operator" begin
    FT = Float32
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 10, helem = 2, Nq = 4)
    fspace = TU.FaceExtrudedFiniteDifferenceSpace(FT; zelem = 10, helem = 2, Nq = 4)
    c = ClimaCore.Fields.coordinate_field(cspace).z .+ 1
    f = ClimaCore.Fields.coordinate_field(fspace).z .+ 1
    fu³ = map(x -> Geometry.Contravariant3Vector(one(x)), f)
    upwind = Operators.UpwindBiasedProductC2F(;
        bottom = Operators.SetValue(FT(0)),
        top = Operators.SetValue(FT(0)),
    )
    interp = Operators.InterpolateF2C()
    # The argument of the Dirichlet operator contains a stencil, so its level
    # adjacent to each boundary cannot be read lazily.
    nested = upwind.(fu³, c .* interp.(f .* f))
    unnested = upwind.(fu³, c .* Base.materialize(interp.(f .* f)))
    @test parent(nested) == parent(unnested)
    div = Operators.DivergenceF2C(;
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    @test parent(div.(nested)) == parent(div.(upwind.(fu³, c .* interp.(f .* f))))
end
