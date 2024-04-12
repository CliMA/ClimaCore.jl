if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "quad-diagonal matrix times vector",
    get_result = () -> (@. ᶠᶜmat ⋅ ᶜvec),
    set_result! = result -> (@. result = ᶠᶜmat ⋅ ᶜvec),
    input_fields = (ᶠᶜmat, ᶜvec),
    ref_set_result! = (_result, _ᶠᶜmat, _ᶜvec) -> mul!(_result, _ᶠᶜmat, _ᶜvec),
)
