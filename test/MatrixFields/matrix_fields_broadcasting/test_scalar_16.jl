#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_16.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "matrix constructions and multiplications" begin
    bc = @lazy @. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
             TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat ⋅
             DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)
    result = materialize(bc)

    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec)

    ref_set_result! =
        (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _ᶜvec,
            _ᶠvec,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
            _temp5,
            _temp6,
        ) -> begin
            mul!(view(_temp1, band(0)), _ᶜᶠmat, _ᶠvec)
            mul!(view(_temp1, band(1)), _ᶜᶜmat, _ᶜvec)
            copyto!(view(_temp2, band(-1)), 1, _ᶠvec, 2)
            mul!(view(_temp2, band(0)), _ᶠᶜmat, _ᶜvec)
            fill!(view(_temp2, band(1)), 1)
            mul!(_temp3, _temp1, _temp2)
            mul!(_temp4, _temp3, _ᶠᶠmat)
            copyto!(view(_temp5, band(0)), 1, _ᶠvec, 1)
            mul!(view(_temp6, band(0)), _temp5, _ᶠvec)
            mul!(_result, _temp4, _temp6)
        end

    temp_value_fields = (
        (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec)),
        (@. TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1)),
        (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
            TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1)),
        (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
            TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat),
        (@. DiagonalMatrixRow(ᶠvec)),
        (@. DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)),
    )
    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        temp_value_fields,
        ref_set_result!,
        using_cuda,
        allowed_max_eps_error = 4,
    )
    test_opt && opt_test_field_broadcast_against_array_reference(
        result,
        bc;
        input_fields,
        temp_value_fields,
        ref_set_result!,
        using_cuda,
    )
    test_opt && !using_cuda && perf_getidx(bc)
end
