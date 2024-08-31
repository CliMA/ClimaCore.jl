#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_15.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "matrix times matrix times linear combination times matrix \
                 times another linear combination times matrix" begin
    bc = @lazy @. ᶠᶜmat ⋅ ᶜᶠmat ⋅
             (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
             ᶠᶠmat ⋅
             (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)) ⋅
             ᶠᶠmat
    result = materialize(bc)

    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat)

    ref_set_result! =
        (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
            _temp5,
            _temp6,
            _temp7,
            _temp8,
            _temp9,
            _temp10,
            _temp11,
            _temp12,
            _temp13,
            _temp14,
        ) -> begin
            mul!(_temp1, _ᶠᶜmat, _ᶜᶠmat)
            @. _temp2 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp3, _temp2, _ᶜᶜmat)
            mul!(_temp4, _temp3, _ᶜᶠmat)
            mul!(_temp5, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_temp6, 4I) # We can't directly use I in array broadcasts.
            @. _temp6 = _temp4 + _temp5 / 3 - _temp6
            mul!(_temp7, _temp1, _temp6)
            mul!(_temp8, _temp7, _ᶠᶠmat)
            mul!(_temp9, _ᶠᶜmat, _ᶜᶜmat)
            mul!(_temp10, _temp9, _ᶜᶠmat)
            @. _temp11 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
            mul!(_temp12, _temp11, _ᶠᶠmat)
            copyto!(_temp13, 4I) # We can't directly use I in array broadcasts.
            @. _temp13 = _temp10 * 2 - _temp12 + _temp13
            mul!(_temp14, _temp8, _temp13)
            mul!(_result, _temp14, _ᶠᶠmat)
        end

    temp_value_fields = (
        (@. ᶠᶜmat ⋅ ᶜᶠmat),
        (@. 2 * ᶠᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat ⋅ ᶠᶠmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,))),
        (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat / 3),
        (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
        (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))),
    )

    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        temp_value_fields,
        ref_set_result!,
        using_cuda,
        allowed_max_eps_error = 128,
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
