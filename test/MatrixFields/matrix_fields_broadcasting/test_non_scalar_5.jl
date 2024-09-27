#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_non_scalar_1.jl"))
=#
import ClimaCore
#! format: off
if !(@isdefined(unit_test_field_broadcast))
    include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_non_scalar_utils.jl"))
end
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "matrix of vectors divided by scalar" begin

    bc = @lazy @. ᶜᶠmat_C12 / 2
    result = materialize(bc)

    ref_set_result! =
        result -> (@. result =
            map(Geometry.Covariant12Vector, ᶜᶠmat2 / 2, ᶜᶠmat3 / 2))

    unit_test_field_broadcast(
        result,
        bc;
        ref_set_result!,
        allowed_max_eps_error = 0,
    )

    test_opt && opt_test_field_broadcast(result, bc; ref_set_result!)
    test_opt && !using_cuda && perf_getidx(bc)
end
