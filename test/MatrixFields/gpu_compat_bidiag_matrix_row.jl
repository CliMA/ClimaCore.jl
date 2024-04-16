#=
using Revise; include(joinpath("test", "MatrixFields", "gpu_compat_bidiag_matrix_row.jl"))
=#
import ClimaCore
import ClimaComms
if !(@isdefined(TU))
    include(
        joinpath(
            pkgdir(ClimaCore),
            "test",
            "TestUtilities",
            "TestUtilities.jl",
        ),
    )
end
import .TestUtilities as TU

import ClimaCore: Spaces, Geometry, Operators, Fields, MatrixFields
using ClimaCore.MatrixFields:
    BidiagonalMatrixRow,
    TridiagonalMatrixRow,
    MultiplyColumnwiseBandMatrixField,
    ⋅
const C3 = Geometry.Covariant3Vector
FT = Float64
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(0)),
    top = Operators.SetGradient(C3(0)),
)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)

device = ClimaComms.device()
context = ClimaComms.context(device)
cspace =
    TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 25, helem = 10, context)
fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
@info "device = $device"

f = (;
    ᶠtridiagonal_matrix_c3 = Fields.Field(TridiagonalMatrixRow{C3{FT}}, fspace),
)

const ᶜleft_bias = Operators.LeftBiasedF2C()
const ᶜright_bias = Operators.RightBiasedF2C()
const ᶜleft_bias_matrix = MatrixFields.operator_matrix(ᶜleft_bias)
const ᶜright_bias_matrix = MatrixFields.operator_matrix(ᶜright_bias)

conv(::Type{_FT}, ᶜbias_matrix) where {_FT} =
    convert(BidiagonalMatrixRow{_FT}, ᶜbias_matrix)
function foo(f)
    (; ᶠtridiagonal_matrix_c3) = f
    space = axes(ᶠtridiagonal_matrix_c3)
    FT = Spaces.undertype(space)
    @. ᶠtridiagonal_matrix_c3 = ᶠgradᵥ_matrix() ⋅ conv(FT, ᶜleft_bias_matrix())
    return nothing
end

foo(f)
