import ClimaCore
import ClimaComms
ClimaComms.@import_required_backends
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

import ClimaCore: Spaces, Geometry, Operators, Fields, MatrixFields
import ClimaCore.Utilities: ConvertTo
import StaticArrays: SArray, SMatrix
import ClimaCore.Geometry: AbstractTensor, Tensor, Components, Covariant, Contravariant
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
    typeof(C3(GFT(0)) * CT3(GFT(0))'),
}

f = (;
    ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ = Fields.Field(∂ᶠu₃ʲ_err_∂ᶠu₃ʲ_type, fspace),
    ᶠtridiagonal_matrix_c3 = Fields.Field(
        TridiagonalMatrixRow{C3{GFT}},
        fspace,
    ),
    ᶠu₃ = Fields.Field(C3{GFT}, fspace),
    adj_u₃ = Fields.Field(DiagonalMatrixRow{typeof(CT3(GFT(0))')}, fspace),
)
c = (;
    ᶜu₃ʲ = Fields.Field(C3{GFT}, cspace),
    bdmr_l = Fields.Field(BidiagonalMatrixRow{GFT}, cspace),
    bdmr_r = Fields.Field(BidiagonalMatrixRow{GFT}, cspace),
    bdmr = Fields.Field(BidiagonalMatrixRow{GFT}, cspace),
)

# `Fields.Field(T, space)` leaves the data uninitialized, so fill it in: the
# comparison below needs deterministic values, and `ᶜu₃ʲ` needs both signs in
# order to exercise both branches of the upwinding `ifelse`.
foreach(field -> fill!(parent(field), 0), values(f))
foreach(field -> fill!(parent(field), 0), values(c))
# (`lat` spans [-90, 90], so it already takes both signs.)
ᶜlat = Fields.coordinate_field(cspace).lat
@. c.ᶜu₃ʲ = C3(ᶜlat)

const ᶜleft_bias = Operators.LeftBiasedF2C()
const ᶜright_bias = Operators.RightBiasedF2C()
const ᶜleft_bias_matrix = MatrixFields.operator_matrix(ᶜleft_bias)
const ᶜright_bias_matrix = MatrixFields.operator_matrix(ᶜright_bias)

one_C3xACT3(::Type{_FT}) where {_FT} = C3(_FT(1)) * CT3(_FT(1))'
get_I_u₃(::Type{_FT}) where {_FT} = DiagonalMatrixRow(one_C3xACT3(_FT))

function foo(c, f)
    (; ᶠtridiagonal_matrix_c3, ᶠu₃, ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ, adj_u₃) = f
    space = axes(ᶠtridiagonal_matrix_c3)
    FT = Spaces.undertype(space)
    I_u₃ = get_I_u₃(FT)
    dtγ = FT(1)

    @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
        dtγ * ᶠtridiagonal_matrix_c3 * DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) -
        (I_u₃,)

    @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ = dtγ * ᶠtridiagonal_matrix_c3 * adj_u₃ - (I_u₃,)

    return nothing
end

# The upwinded stencil, written as a single fused broadcast expression. The
# conversion goes through `ConvertTo{T}`, an empty struct, rather than the
# obvious `Base.Fix1(convert, T)`: the latter stores a `Type` field and so is
# not a bitstype, which a fused broadcast cannot compile for the GPU.
function fused_stencil!(c, f)
    (; ᶠtridiagonal_matrix_c3) = f
    (; ᶜu₃ʲ) = c
    FT = Spaces.undertype(axes(ᶠtridiagonal_matrix_c3))
    to_bidiagonal_row = ConvertTo{BidiagonalMatrixRow{FT}}()

    @. ᶠtridiagonal_matrix_c3 =
        -(ᶠgradᵥ_matrix()) * ifelse(
            ᶜu₃ʲ.components.data.:1 > 0,
            to_bidiagonal_row(ᶜleft_bias_matrix()),
            to_bidiagonal_row(ᶜright_bias_matrix()),
        )
    return nothing
end

# The same stencil, decomposed into simpler broadcast expressions, kept as
# coverage of the same computation.
function decomposed_stencil!(c, f)
    (; ᶠtridiagonal_matrix_c3) = f
    (; ᶜu₃ʲ, bdmr_l, bdmr_r, bdmr) = c
    FT = Spaces.undertype(axes(ᶠtridiagonal_matrix_c3))
    to_bidiagonal_row = ConvertTo{BidiagonalMatrixRow{FT}}()

    @. bdmr_l = to_bidiagonal_row(ᶜleft_bias_matrix())
    @. bdmr_r = to_bidiagonal_row(ᶜright_bias_matrix())
    @. bdmr = ifelse(ᶜu₃ʲ.components.data.:1 > 0, bdmr_l, bdmr_r)
    @. ᶠtridiagonal_matrix_c3 = -(ᶠgradᵥ_matrix()) * bdmr
    return nothing
end

using Test
@testset "gpu_compat_bidiag_matrix_row" begin
    foo(c, f)
    @test all(isfinite, parent(f.∂ᶠu₃ʲ_err_∂ᶠu₃ʲ))

    decomposed_stencil!(c, f)
    @test all(isfinite, parent(f.ᶠtridiagonal_matrix_c3))

    # Both forms now compile on every device, so they must agree everywhere.
    decomposed = Array(parent(f.ᶠtridiagonal_matrix_c3))
    fused_stencil!(c, f)
    @test Array(parent(f.ᶠtridiagonal_matrix_c3)) == decomposed
end
