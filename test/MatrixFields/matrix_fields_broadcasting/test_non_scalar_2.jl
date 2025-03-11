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
@testset "matrix of covectors times matrix of vectors times matrix \
                 of numbers times matrix of covectors times matrix of \
                 vectors" begin

    bc = @lazy @. ᶜᶠmat_AC1 * ᶠᶜmat_C12 * ᶜᶠmat * ᶠᶜmat_AC1 * ᶜᶠmat_C12
    result = materialize(bc)

    ref_set_result! =
        result -> (@. result =
            ᶜᶠmat *
            (
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) * ᶠᶜmat2 +
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) * ᶠᶜmat3
            ) *
            ᶜᶠmat *
            ᶠᶜmat *
            (
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:1) * ᶜᶠmat2 +
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:2) * ᶜᶠmat3
            ))

    unit_test_field_broadcast(
        result,
        bc;
        ref_set_result!,
        allowed_max_eps_error = 10,
    )

    test_opt && opt_test_field_broadcast(result, bc; ref_set_result!)
    test_opt && !using_cuda && perf_getidx(bc)
end
