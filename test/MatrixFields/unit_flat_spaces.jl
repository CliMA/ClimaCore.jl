import ClimaCore
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

include("matrix_field_test_utils.jl")
import ClimaCore.MatrixFields: @name

@testset "Matrix Fields with Spectral Element and Point Spaces" begin
    get_j_field(space, FT) = fill(MatrixFields.DiagonalMatrixRow(FT(1)), space)

    implicit_vars = (@name(tmp.v1), @name(tmp.v2))
    for FT in (Float32, Float64)
        comms_ctx = ClimaComms.SingletonCommsContext(comms_device)
        ps = TU.PointSpace(FT; context = comms_ctx)
        ses = TU.SpectralElementSpace2D(FT; context = comms_ctx)
        v1 = Fields.zeros(ps)
        v2 = Fields.zeros(ses)
        Y = Fields.FieldVector(; :tmp => (; :v1 => v1, :v2 => v2))
        implicit_blocks = MatrixFields.unrolled_map(
            var ->
                (var, var) =>
                    get_j_field(axes(MatrixFields.get_field(Y, var)), FT),
            implicit_vars,
        )
        matrix = MatrixFields.FieldMatrix(implicit_blocks...)
        alg = MatrixFields.BlockDiagonalSolve()
        solver = MatrixFields.FieldMatrixSolver(alg, matrix, Y)
        b1 = random_field(FT, ps)
        b2 = random_field(FT, ses)
        x = similar(Y)
        b = Fields.FieldVector(; :tmp => (; :v1 => b1, :v2 => b2))
        MatrixFields.field_matrix_solve!(solver, x, matrix, b)
        @test x == b
    end
end

@testset "DiagonalMatrixRow multiplication on Spectral Element and Point Spaces" begin
    for FT in (Float32, Float64)
        comms_ctx = ClimaComms.SingletonCommsContext(comms_device)
        ps = TU.PointSpace(FT; context = comms_ctx)
        ses = TU.SpectralElementSpace2D(FT; context = comms_ctx)
        for space in (ps, ses)
            J = random_field(DiagonalMatrixRow{FT}, space)
            b = random_field(FT, space)
            J_b = @. J * b
            @test J_b isa Fields.Field
            @test axes(J_b) === space
            @test J_b == @. J.entries.:1 * b
        end
    end
end
