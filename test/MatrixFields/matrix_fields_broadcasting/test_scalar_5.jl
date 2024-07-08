#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_5.jl"))
=#
import ClimaCore
#! format: off
if !(@isdefined(unit_test_field_broadcast_vs_array_reference))
    include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
end
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "tri-diagonal matrix times tri-diagonal matrix" begin
    bc = @lazy @. ᶠᶠmat ⋅ ᶠᶠmat
    result = materialize(bc)

    input_fields = (ᶠᶠmat,)
    ref_set_result! = (_result, _ᶠᶠmat) -> mul!(_result, _ᶠᶠmat, _ᶠᶠmat)

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
end
