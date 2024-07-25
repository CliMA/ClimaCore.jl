if !(@isdefined(test_field_broadcast))
    include("test_non_scalar_utils.jl")
end

test_field_broadcast(;
    test_name = "matrix of nested values times matrix of nested values \
                 times matrix of numbers times matrix of numbers times \
                 vector of nested values",
    get_result = () -> (@. ᶜᶠmat_NT ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat_NT ⋅ ᶜvec_NT),
    set_result! = result ->
        (@. result = ᶜᶠmat_NT ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat_NT ⋅ ᶜvec_NT),
    ref_set_result! = result -> (@. result = nested_type(
        ᶜᶠmat ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec,
        ᶜᶠmat2 ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat2 ⋅ ᶜvec,
        ᶜᶠmat3 ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat3 ⋅ ᶜvec,
    )),
)
