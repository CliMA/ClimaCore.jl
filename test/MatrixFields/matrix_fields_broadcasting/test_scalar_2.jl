if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "tri-diagonal matrix times vector",
    get_result = () -> (@. ᶠᶠmat ⋅ ᶠvec),
    set_result! = result -> (@. result = ᶠᶠmat ⋅ ᶠvec),
    input_fields = (ᶠᶠmat, ᶠvec),
    ref_set_result! = (_result, _ᶠᶠmat, _ᶠvec) -> mul!(_result, _ᶠᶠmat, _ᶠvec),
)
