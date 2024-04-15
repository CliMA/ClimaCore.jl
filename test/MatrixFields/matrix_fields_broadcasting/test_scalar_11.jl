if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "linear combination of matrix products and LinearAlgebra.I",
    get_result = () ->
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
    set_result! = result ->
        (@. result = 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
    get_temp_value_fields = () -> (
        (@. 2 * ᶠᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat ⋅ ᶠᶠmat),
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
        @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
        mul!(_temp2, _temp1, _ᶜᶜmat)
        mul!(_temp3, _temp2, _ᶜᶠmat)
        mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
        copyto!(_result, 4I) # We can't directly use I in array broadcasts.
        @. _result = _temp3 + _temp4 / 3 - _result
    end,
)
