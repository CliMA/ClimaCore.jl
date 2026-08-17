import ClimaCore
#! format: off
if !(@isdefined(unit_test_field_broadcast))
    include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_non_scalar_utils.jl"))
end
#! format: on
# Opt checks (JET + allocation gates) run on CI; set CLIMACORE_TEST_OPT=true
# to also run them locally.
test_opt =
    get(ENV, "CLIMACORE_TEST_OPT", get(ENV, "BUILDKITE", "false")) == "true"
@testset "matrix of covectors and numbers times matrix of vectors \
                 and covectors times matrix of numbers and vectors times \
                 vector of numbers" begin

    bc = @lazy @. ᶜᶠmat_AC1_num * ᶠᶜmat_C12_AC1 * ᶜᶠmat_num_C12 * ᶠvec
    result = materialize(bc)

    ref_set_result! =
        result -> (@. result = tuple(
            ᶜᶠmat *
            (
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) * ᶠᶜmat2 +
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) * ᶠᶜmat3
            ) *
            ᶜᶠmat *
            ᶠvec,
            ᶜᶠmat *
            ᶠᶜmat *
            (
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:1) * ᶜᶠmat2 +
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:2) * ᶜᶠmat3
            ) *
            ᶠvec,
        ))

    unit_test_field_broadcast(
        result,
        bc;
        ref_set_result!,
        allowed_max_eps_error = 10,
    )

    test_opt && opt_test_field_broadcast(result, bc; ref_set_result!)
    test_opt && !USING_CUDA && perf_getidx(bc)
end
