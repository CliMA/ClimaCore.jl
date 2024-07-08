#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_9.jl"))
=#
import ClimaCore
#! format: off
if !(@isdefined(unit_test_field_broadcast_vs_array_reference))
    include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
end
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "diagonal matrix times bi-diagonal matrix times \
                 tri-diagonal matrix times quad-diagonal matrix times \
                 vector" begin
    bc = @lazy @. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec
    result = materialize(bc)

    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec)
    temp_value_fields = (
        (@. ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat),
        (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat),
    )
    ref_set_result! =
        (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _ᶜvec,
            _temp1,
            _temp2,
            _temp3,
        ) -> begin
            mul!(_temp1, _ᶜᶜmat, _ᶜᶠmat)
            mul!(_temp2, _temp1, _ᶠᶠmat)
            mul!(_temp3, _temp2, _ᶠᶜmat)
            mul!(_result, _temp3, _ᶜvec)
        end

    unit_test_field_broadcast_vs_array_reference(
        result,
        bc;
        input_fields,
        temp_value_fields,
        ref_set_result!,
        using_cuda,
        allowed_max_eps_error = 10,
    )
    test_opt && opt_test_field_broadcast_against_array_reference(
        result,
        bc;
        input_fields,
        temp_value_fields,
        ref_set_result!,
        using_cuda,
    )
end
