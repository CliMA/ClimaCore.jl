if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "diagonal matrix times vector",
    get_result = () -> (@. ᶜᶜmat ⋅ ᶜvec),
    set_result! = result -> (@. result = ᶜᶜmat ⋅ ᶜvec),
    input_fields = (ᶜᶜmat, ᶜvec),
    ref_set_result! = (_result, _ᶜᶜmat, _ᶜvec) -> mul!(_result, _ᶜᶜmat, _ᶜvec),
)
