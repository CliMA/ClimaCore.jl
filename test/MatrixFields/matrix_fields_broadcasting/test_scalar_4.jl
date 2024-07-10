#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_4.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "diagonal matrix times bi-diagonal matrix" begin
    bc = @lazy @. ᶜᶜmat ⋅ ᶜᶠmat
    result = materialize(bc)

    input_fields = (ᶜᶜmat, ᶜᶠmat)
    ref_set_result! = (_result, _ᶜᶜmat, _ᶜᶠmat) -> mul!(_result, _ᶜᶜmat, _ᶜᶠmat)

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
    test_opt && !using_cuda && benchmark_getidx(bc)
end
