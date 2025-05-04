#=
julia --project=.buildkite
using Revise; include("test/Operators/finitedifference/benchmark_fd_ops_shared_memory_matrix_fields.jl")
=#
include("utils_fd_ops_shared_memory.jl")
using ClimaComms
using LinearAlgebra
using BenchmarkTools
ClimaComms.@import_required_backends
using ClimaCore.CommonSpaces
using ClimaCore: Operators, Fields, MatrixFields, Geometry, Spaces
using ClimaCore.Utilities: half

covariant3_unit_vector(lg) =
    Geometry.Covariant3Vector(
        1 / Geometry._norm(Geometry.Covariant3Vector(1), lg)
    )

#! format: off
function bench_kernels!(L, K, C)
    space = axes(K)
    ᶠspace = Spaces.face_space(space)
    levels = Spaces.nlevels(ᶠspace)
    ᶠlg_N = Fields.level(
        Fields.local_geometry_field(ᶠspace),
        levels - half,
    )
    topfluxBC = @. covariant3_unit_vector(ᶠlg_N) * 0
    topBC_op = Operators.SetBoundaryOperator(
        top = Operators.SetValue(topfluxBC),
        bottom = Operators.SetValue(Geometry.Covariant3Vector(0)),
    )
    interpc2f_op = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    divf2c_op = Operators.DivergenceF2C()
    divf2c_matrix = MatrixFields.operator_matrix(divf2c_op)
    gradc2f_op = Operators.GradientC2F(
        top = Operators.SetGradient(Geometry.WVector(0)),
        bottom = Operators.SetGradient(Geometry.WVector(0)),
    )
    gradc2f_matrix = MatrixFields.operator_matrix(gradc2f_op)
    args = (L, K, C, gradc2f_matrix, divf2c_matrix, interpc2f_op, topBC_op)
    @benchmark CUDA.@sync kernel!($args...)
end

function kernel!(L, K, C, gradc2f_matrix, divf2c_matrix, interpc2f_op, topBC_op)
    @. L = (
        divf2c_matrix() * (
            MatrixFields.DiagonalMatrixRow(interpc2f_op(K)) *
            gradc2f_matrix() * MatrixFields.DiagonalMatrixRow(C) +
            MatrixFields.LowerDiagonalMatrixRow(
                topBC_op(Geometry.Covariant3Vector(zero(interpc2f_op(K)))),
            )
        )
    ) - (I,)
end

let FT = Float64
    ᶜspace =
        get_space_extruded(ClimaComms.device(), FT; z_elem = 10, h_elem = 30);
    K = Fields.Field(Float64, ᶜspace);
    C = Fields.Field(Float64, ᶜspace);
    L = Fields.Field(MatrixFields.TridiagonalMatrixRow{FT}, ᶜspace);
    fill!(K, 1);
    fill!(C, 1);
    bench_kernels!(L, K, C)
end
