if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "tri-diagonal matrix times tri-diagonal matrix",
    get_result = () -> (@. ᶠᶠmat ⋅ ᶠᶠmat),
    set_result! = result -> (@. result = ᶠᶠmat ⋅ ᶠᶠmat),
    input_fields = (ᶠᶠmat,),
    ref_set_result! = (_result, _ᶠᶠmat) -> mul!(_result, _ᶠᶠmat, _ᶠᶠmat),
)
