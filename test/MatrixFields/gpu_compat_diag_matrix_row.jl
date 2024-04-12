#=
julia -g2 --check-bounds=yes --project=test
using Revise; include(joinpath("test", "MatrixFields", "gpu_compat_diag_matrix_row.jl"))
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

import ClimaCore: Spaces, Geometry, Fields
import ClimaCore.Geometry: AxisTensor, CovariantAxis, ContravariantAxis
import StaticArrays: SArray
import LinearAlgebra: adjoint
using ClimaCore.MatrixFields:
    BandMatrixRow,
    DiagonalMatrixRow,
    TridiagonalMatrixRow,
    MultiplyColumnwiseBandMatrixField
const C3 = Geometry.Covariant3Vector
const CT3 = Geometry.Contravariant3Vector
FT = Float64
const ⋅ = MultiplyColumnwiseBandMatrixField()

device = ClimaComms.device()
context = ClimaComms.context(device)
cspace =
    TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 25, helem = 10, context)
fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
@info "device = $device"

one_C3xACT3(::Type{FT}) where {FT} = C3(FT(1)) * CT3(FT(1))'
get_I_u₃(::Type{FT}) where {FT} = DiagonalMatrixRow(one_C3xACT3(FT))

∂ᶠu₃ʲ_err_∂ᶠu₃ʲ_type = BandMatrixRow{
    -1,
    3,
    AxisTensor{
        FT,
        2,
        Tuple{CovariantAxis{(3,)}, ContravariantAxis{(3,)}},
        SArray{Tuple{1, 1}, FT, 2, 1},
    },
}

f = (;
    ᶠtridiagonal_matrix_c3 = Fields.Field(TridiagonalMatrixRow{C3{FT}}, fspace),
    u₃ = Fields.Field(C3{FT}, fspace),
    ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ = Fields.Field(∂ᶠu₃ʲ_err_∂ᶠu₃ʲ_type, fspace),
)

function foo(f)
    (; ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ, ᶠtridiagonal_matrix_c3) = f
    space = axes(ᶠtridiagonal_matrix_c3)
    FT = Spaces.undertype(space)
    I_u₃ = get_I_u₃(FT)
    dtγ = FT(1)
    @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
        dtγ * ᶠtridiagonal_matrix_c3 ⋅ DiagonalMatrixRow(adjoint(CT3(f.u₃))) -
        (I_u₃,)
    return nothing
end

foo(f)
