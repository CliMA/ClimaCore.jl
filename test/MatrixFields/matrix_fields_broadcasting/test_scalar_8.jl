if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "diagonal matrix times bi-diagonal matrix times \
                 tri-diagonal matrix times quad-diagonal matrix, but with \
                 forced right-associativity",
    get_result = () -> (@. ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
    set_result! = result -> (@. result = ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
    get_temp_value_fields = () ->
        ((@. ᶠᶠmat ⋅ ᶠᶜmat), (@. ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
    ref_set_result! = (
        _result,
        _ᶜᶜmat,
        _ᶜᶠmat,
        _ᶠᶠmat,
        _ᶠᶜmat,
        _temp1,
        _temp2,
    ) -> begin
        mul!(_temp1, _ᶠᶠmat, _ᶠᶜmat)
        mul!(_temp2, _ᶜᶠmat, _temp1)
        mul!(_result, _ᶜᶜmat, _temp2)
    end,
    test_broken_with_cuda = true, # TODO: Fix this.
)
