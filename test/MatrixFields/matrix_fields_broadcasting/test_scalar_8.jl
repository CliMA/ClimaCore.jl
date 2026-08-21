import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
# Opt checks (JET + allocation gates) run on CI; set CLIMACORE_TEST_OPT=true
# to also run them locally.
test_opt =
    get(ENV, "CLIMACORE_TEST_OPT", get(ENV, "BUILDKITE", "false")) == "true"
@testset "diagonal matrix times bi-diagonal matrix times \
                 tri-diagonal matrix times quad-diagonal matrix, but with \
                 forced right-associativity" begin
    bc = @lazy @. ᶜᶜmat * (ᶜᶠmat * (ᶠᶠmat * ᶠᶜmat))
    # CUDA cannot compile this expression. Assert the failure it does produce,
    # then skip the CPU-only remainder. Do not `exit` here: these files are
    # `include`d into the shared test process, so exiting would terminate the
    # whole run and silently skip every test after this one.
    if USING_CUDA
        @test_throws invalid_ir_error materialize(bc)
        @warn "cuda is broken for this test, skipping its remainder."
    else
        result = materialize(bc)

        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat)
        temp_value_fields = ((@. ᶠᶠmat * ᶠᶜmat), (@. ᶜᶠmat * (ᶠᶠmat * ᶠᶜmat)))
        ref_set_result! =
            (_result, _ᶜᶜmat, _ᶜᶠmat, _ᶠᶠmat, _ᶠᶜmat, _temp1, _temp2) -> begin
                mul!(_temp1, _ᶠᶠmat, _ᶠᶜmat)
                mul!(_temp2, _ᶜᶠmat, _temp1)
                mul!(_result, _ᶜᶜmat, _temp2)
            end

        unit_test_field_broadcast_vs_array_reference(
            result,
            bc;
            input_fields,
            temp_value_fields,
            ref_set_result!,
            USING_CUDA,
            allowed_max_eps_error = 10,
        )
        test_opt && opt_test_field_broadcast_against_array_reference(
            result,
            bc;
            input_fields,
            temp_value_fields,
            ref_set_result!,
            USING_CUDA,
        )
        test_opt && !USING_CUDA && perf_getidx(bc)
    end
end
