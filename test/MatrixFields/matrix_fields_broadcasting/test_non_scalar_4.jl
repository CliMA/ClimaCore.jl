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
@testset "matrix of nested values times matrix of nested values \
                 times matrix of numbers times matrix of numbers times \
                 vector of nested values" begin

    bc = @lazy @. ᶜᶠmat_NT ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat_NT ⋅ ᶜvec_NT
    result = materialize(bc)

    ref_set_result! =
        result -> (@. result = nested_type(
            ᶜᶠmat ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec,
            ᶜᶠmat2 ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat2 ⋅ ᶜvec,
            ᶜᶠmat3 ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat3 ⋅ ᶜvec,
        ))

    unit_test_field_broadcast(
        result,
        bc;
        ref_set_result!,
        allowed_max_eps_error = 10,
    )

    test_opt && opt_test_field_broadcast(result, bc; ref_set_result!)
end
