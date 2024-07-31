if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "diagonal matrix times bi-diagonal matrix times \
                 tri-diagonal matrix times quad-diagonal matrix times \
                 vector, but with forced right-associativity",
    get_result = () -> (@. ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)))),
    set_result! = result ->
        (@. result = ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)))),
    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec),
    get_temp_value_fields = () -> (
        (@. ᶠᶜmat ⋅ ᶜvec),
        (@. ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)),
        (@. ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec))),
    ),
    ref_set_result! = (
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
    end,
    time_ratio_limit = 15, # This case's ref function is fast on Buildkite.
    test_broken_with_cuda = true, # TODO: Fix this.
)
