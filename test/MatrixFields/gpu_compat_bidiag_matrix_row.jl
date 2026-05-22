#=
using Revise; include(joinpath("test", "MatrixFields", "gpu_compat_bidiag_matrix_row.jl"))
=#
import ClimaCore
import ClimaComms
ClimaComms.@import_required_backends
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

import ClimaCore: Spaces, Geometry, Operators, Fields, MatrixFields
using LinearAlgebra: Adjoint
import StaticArrays: SArray
import ClimaCore.Geometry: AxisTensor, CovariantAxis, ContravariantAxis
using ClimaCore.MatrixFields:
    BandMatrixRow,
    DiagonalMatrixRow,
    BidiagonalMatrixRow,
    TridiagonalMatrixRow,
    MultiplyColumnwiseBandMatrixField
const C3 = Geometry.Covariant3Vector
const CT3 = Geometry.Contravariant3Vector
GFT = Float64
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(0)),
    top = Operators.SetGradient(C3(0)),
)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)

device = ClimaComms.device()
context = ClimaComms.context(device)
cspace =
    TU.CenterExtrudedFiniteDifferenceSpace(GFT; zelem = 25, helem = 10, context)
fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
@info "device = $device"

∂ᶠu₃ʲ_err_∂ᶠu₃ʲ_type = BandMatrixRow{
    -1,
    3,
    AxisTensor{
        GFT,
        2,
        Tuple{CovariantAxis{(3,)}, ContravariantAxis{(3,)}},
        SArray{Tuple{1, 1}, GFT, 2, 1},
    },
}

f = (;
    ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ = Fields.Field(∂ᶠu₃ʲ_err_∂ᶠu₃ʲ_type, fspace),
    ᶠtridiagonal_matrix_c3 = Fields.Field(
        TridiagonalMatrixRow{C3{GFT}},
        fspace,
    ),
    ᶠu₃ = Fields.Field(C3{GFT}, fspace),
    adj_u₃ = Fields.Field(DiagonalMatrixRow{Adjoint{GFT, CT3{GFT}}}, fspace),
)
c = (;
    ᶜu₃ʲ = Fields.Field(C3{GFT}, cspace),
    bdmr_l = Fields.Field(BidiagonalMatrixRow{GFT}, cspace),
    bdmr_r = Fields.Field(BidiagonalMatrixRow{GFT}, cspace),
    bdmr = Fields.Field(BidiagonalMatrixRow{GFT}, cspace),
)

const ᶜleft_bias = Operators.LeftBiasedF2C()
const ᶜright_bias = Operators.RightBiasedF2C()
const ᶜleft_bias_matrix = MatrixFields.operator_matrix(ᶜleft_bias)
const ᶜright_bias_matrix = MatrixFields.operator_matrix(ᶜright_bias)

one_C3xACT3(::Type{_FT}) where {_FT} = C3(_FT(1)) * CT3(_FT(1))'
get_I_u₃(::Type{_FT}) where {_FT} = DiagonalMatrixRow(one_C3xACT3(_FT))

function foo(c, f)
    (; ᶠtridiagonal_matrix_c3, ᶠu₃, ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ, adj_u₃) = f
    (; ᶜu₃ʲ, bdmr_l, bdmr_r, bdmr) = c
    space = axes(ᶠtridiagonal_matrix_c3)
    FT = Spaces.undertype(space)
    I_u₃ = get_I_u₃(FT)
    dtγ = FT(1)
    to_bidiagonal_row = Base.Fix1(convert, BidiagonalMatrixRow{FT})

    @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
        dtγ * ᶠtridiagonal_matrix_c3 * DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) -
        (I_u₃,)

    @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ = dtγ * ᶠtridiagonal_matrix_c3 * adj_u₃ - (I_u₃,)

    # Fails on gpu
    @. ᶠtridiagonal_matrix_c3 =
        -(ᶠgradᵥ_matrix()) * ifelse(
            ᶜu₃ʲ.components.data.:1 > 0,
            to_bidiagonal_row(ᶜleft_bias_matrix()),
            to_bidiagonal_row(ᶜright_bias_matrix()),
        )

    # However, this can be decomposed into simpler broadcast
    # expressions that will run on gpus:
    @. bdmr_l = to_bidiagonal_row(ᶜleft_bias_matrix())
    @. bdmr_r = to_bidiagonal_row(ᶜright_bias_matrix())
    @. bdmr = ifelse(ᶜu₃ʲ.components.data.:1 > 0, bdmr_l, bdmr_r)
    @. ᶠtridiagonal_matrix_c3 = -(ᶠgradᵥ_matrix()) * bdmr

    return nothing
end

foo(c, f)
