if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "another linear combination of matrix products and \
                 LinearAlgebra.I",
    get_result = () ->
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
    set_result! = result ->
        (@. result = ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
    get_temp_value_fields = () -> (
        (@. ᶠᶜmat ⋅ ᶜᶜmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat / 3),
        (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
    ),
    ref_set_result! = (
        _result,
        _ᶜᶜmat,
        _ᶜᶠmat,
        _ᶠᶠmat,
        _ᶠᶜmat,
        _temp1,
        _temp2,
        _temp3,
        _temp4,
    ) -> begin
        mul!(_temp1, _ᶠᶜmat, _ᶜᶜmat)
        mul!(_temp2, _temp1, _ᶜᶠmat)
        @. _temp3 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
        mul!(_temp4, _temp3, _ᶠᶠmat)
        copyto!(_result, 4I) # We can't directly use I in array broadcasts.
        @. _result = _temp2 * 2 - _temp4 + _result
    end,
)
