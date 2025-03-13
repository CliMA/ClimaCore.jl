#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_2.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "tri-diagonal matrix times vector" begin
    bc = @lazy @. ᶠᶠmat * ᶠvec
    result = materialize(bc)

    input_fields = (ᶠᶠmat, ᶠvec)
    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        using_cuda,
        allowed_max_eps_error = 1,
    )
    test_opt && opt_test_field_broadcast_against_array_reference(
        result,
        bc;
        input_fields,
        using_cuda,
    )
    test_opt && !using_cuda && perf_getidx(bc)
end
