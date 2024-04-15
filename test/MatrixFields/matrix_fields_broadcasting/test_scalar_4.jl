if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "diagonal matrix times bi-diagonal matrix",
    get_result = () -> (@. ᶜᶜmat ⋅ ᶜᶠmat),
    set_result! = result -> (@. result = ᶜᶜmat ⋅ ᶜᶠmat),
    input_fields = (ᶜᶜmat, ᶜᶠmat),
    ref_set_result! = (_result, _ᶜᶜmat, _ᶜᶠmat) ->
        mul!(_result, _ᶜᶜmat, _ᶜᶠmat),
)
