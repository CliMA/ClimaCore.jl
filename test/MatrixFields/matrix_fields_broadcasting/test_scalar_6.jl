#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_6.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "quad-diagonal matrix times diagonal matrix" begin
    bc = @lazy @. ᶠᶜmat * ᶜᶜmat
    result = materialize(bc)

    input_fields = (ᶠᶜmat, ᶜᶜmat)
    ref_set_result! = (_result, _ᶠᶜmat, _ᶜᶜmat) -> mul!(_result, _ᶠᶜmat, _ᶜᶜmat)

    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        ref_set_result!,
        using_cuda,
        allowed_max_eps_error = 10,
    )
    test_opt && opt_test_field_broadcast_against_array_reference(
        result,
        bc;
        input_fields,
        ref_set_result!,
        using_cuda,
    )
    test_opt && !using_cuda && perf_getidx(bc)
end
