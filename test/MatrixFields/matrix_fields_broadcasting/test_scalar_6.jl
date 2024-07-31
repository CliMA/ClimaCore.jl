if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "quad-diagonal matrix times diagonal matrix",
    get_result = () -> (@. ᶠᶜmat ⋅ ᶜᶜmat),
    set_result! = result -> (@. result = ᶠᶜmat ⋅ ᶜᶜmat),
    input_fields = (ᶠᶜmat, ᶜᶜmat),
    ref_set_result! = (_result, _ᶠᶜmat, _ᶜᶜmat) ->
        mul!(_result, _ᶠᶜmat, _ᶜᶜmat),
)
