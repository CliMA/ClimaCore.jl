import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
# Opt checks (JET + allocation gates) run on CI; set CLIMACORE_TEST_OPT=true
# to also run them locally.
test_opt =
    get(ENV, "CLIMACORE_TEST_OPT", get(ENV, "BUILDKITE", "false")) == "true"
@testset "diagonal matrix times vector" begin
    bc = @lazy @. ᶜᶜmat * ᶜvec
    result = materialize(bc)

    input_fields = (ᶜᶜmat, ᶜvec)
    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        USING_CUDA,
    )
    test_opt && opt_test_field_broadcast_against_array_reference(
        result,
        bc;
        input_fields,
        USING_CUDA,
    )
    test_opt && !USING_CUDA && perf_getidx(bc)
end
