import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
# Opt checks (JET + allocation gates) run on CI; set CLIMACORE_TEST_OPT=true
# to also run them locally.
test_opt =
    get(ENV, "CLIMACORE_TEST_OPT", get(ENV, "BUILDKITE", "false")) == "true"
@testset "tri-diagonal matrix times tri-diagonal matrix" begin
    bc = @lazy @. ᶠᶠmat * ᶠᶠmat
    result = materialize(bc)

    input_fields = (ᶠᶠmat,)
    ref_set_result! = (_result, _ᶠᶠmat) -> mul!(_result, _ᶠᶠmat, _ᶠᶠmat)

    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        ref_set_result!,
        USING_CUDA,
        allowed_max_eps_error = 10,
    )
    test_opt && opt_test_field_broadcast_against_array_reference(
        result,
        bc;
        input_fields,
        ref_set_result!,
        USING_CUDA,
    )
    test_opt && !USING_CUDA && perf_getidx(bc)
end
