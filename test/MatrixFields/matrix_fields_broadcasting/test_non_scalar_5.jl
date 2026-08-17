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

# Wrap the per-entry constructor in a named, top-level function. Writing the
# reference implementation as `@. map(Geometry.Covariant12Vector, ...)` makes
# the bare *type* a broadcast argument, and `Base.broadcastable(::Type)`
# allocates a `Ref` for it (16 bytes per call), so the reference implementation
# would look like it allocates when the computation itself does not.
map_C12(row1, row2) = map(Geometry.Covariant12Vector, row1, row2)

@testset "matrix of vectors divided by scalar" begin

    bc = @lazy @. ᶜᶠmat_C12 / 2
    result = materialize(bc)

    ref_set_result! = result -> (@. result = map_C12(ᶜᶠmat2 / 2, ᶜᶠmat3 / 2))

    unit_test_field_broadcast(
        result,
        bc;
        ref_set_result!,
        allowed_max_eps_error = 0,
    )

    test_opt && opt_test_field_broadcast(result, bc; ref_set_result!)
    test_opt && !USING_CUDA && perf_getidx(bc)
end
