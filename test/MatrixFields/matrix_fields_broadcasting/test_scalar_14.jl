#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_14.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "linear combination times another linear combination" begin
    bc = @lazy @. (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
             (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))
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
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_temp5, 4I) # We can't directly use I in array broadcasts.
            @. _temp5 = _temp3 + _temp4 / 3 - _temp5
            mul!(_temp6, _ᶠᶜmat, _ᶜᶜmat)
            mul!(_temp7, _temp6, _ᶜᶠmat)
            @. _temp8 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
            mul!(_temp9, _temp8, _ᶠᶠmat)
            copyto!(_temp10, 4I) # We can't directly use I in array broadcasts.
            @. _temp10 = _temp7 * 2 - _temp9 + _temp10
            mul!(_result, _temp5, _temp10)
        end

    temp_value_fields = (
        (@. 2 * ᶠᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat ⋅ ᶠᶠmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        (@. ᶠᶜmat ⋅ ᶜᶜmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat / 3),
        (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
    )

    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        temp_value_fields,
        ref_set_result!,
        using_cuda,
        allowed_max_eps_error = 30,
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
