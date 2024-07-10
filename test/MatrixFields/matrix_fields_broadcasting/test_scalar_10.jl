#=
julia --project
using Revise; include(joinpath("test", "MatrixFields", "matrix_fields_broadcasting", "test_scalar_10.jl"))
=#
import ClimaCore
#! format: off
include(joinpath(pkgdir(ClimaCore),"test","MatrixFields","matrix_fields_broadcasting","test_scalar_utils.jl"))
#! format: on
test_opt = get(ENV, "BUILDKITE", "") == "true"
@testset "diagonal matrix times bi-diagonal matrix times \
                 tri-diagonal matrix times quad-diagonal matrix times \
                 vector, but with forced right-associativity" begin
    bc = @lazy @. ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)))
    if using_cuda
        @test_throws invalid_ir_error materialize(bc)
        @warn "cuda is broken for this test, exiting."
        exit(0)
    end
    result = materialize(bc)

    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec)
    temp_value_fields = (
        (@. ᶠᶜmat ⋅ ᶜvec),
        (@. ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)),
        (@. ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec))),
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
            mul!(_temp1, _ᶠᶜmat, _ᶜvec)
            mul!(_temp2, _ᶠᶠmat, _temp1)
            mul!(_temp3, _ᶜᶠmat, _temp2)
            mul!(_result, _ᶜᶜmat, _temp3)
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
    test_opt && !using_cuda && benchmark_getidx(bc)
end
