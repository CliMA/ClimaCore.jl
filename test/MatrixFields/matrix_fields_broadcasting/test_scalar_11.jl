#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_11.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "linear combination of matrix products and LinearAlgebra.I" begin
    bc = @lazy @. 2 * ᶠᶜmat * ᶜᶜmat * ᶜᶠmat + ᶠᶠmat * ᶠᶠmat / 3 - (4I,)
    result = materialize(bc)

    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat)
    temp_value_fields = (
        (@. 2 * ᶠᶜmat),
        (@. 2 * ᶠᶜmat * ᶜᶜmat),
        (@. 2 * ᶠᶜmat * ᶜᶜmat * ᶜᶠmat),
        (@. ᶠᶠmat * ᶠᶠmat),
    )
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
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_result, 4I) # We can't directly use I in array broadcasts.
            @. _result = _temp3 + _temp4 / 3 - _result
        end

    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        temp_value_fields,
        ref_set_result!,
        using_cuda,
        allowed_max_eps_error = 10,
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
