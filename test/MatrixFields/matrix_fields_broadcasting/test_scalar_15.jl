if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "matrix times matrix times linear combination times matrix \
                 times another linear combination times matrix",
    get_result = () -> (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
        (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅ ᶠᶠmat ⋅
        (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)) ⋅ ᶠᶠmat),
    set_result! = result -> (@. result =
        ᶠᶜmat ⋅ ᶜᶠmat ⋅
        (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅ ᶠᶠmat ⋅
        (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)) ⋅ ᶠᶠmat),
    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
    get_temp_value_fields = () -> (
        (@. ᶠᶜmat ⋅ ᶜᶠmat),
        (@. 2 * ᶠᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat ⋅ ᶠᶠmat),
        (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,))),
        (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
        (@. ᶠᶠmat / 3),
        (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
        (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
        (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))),
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
        _temp5,
        _temp6,
        _temp7,
        _temp8,
        _temp9,
        _temp10,
        _temp11,
        _temp12,
        _temp13,
        _temp14,
    ) -> begin
        mul!(_temp1, _ᶠᶜmat, _ᶜᶠmat)
        @. _temp2 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
        mul!(_temp3, _temp2, _ᶜᶜmat)
        mul!(_temp4, _temp3, _ᶜᶠmat)
        mul!(_temp5, _ᶠᶠmat, _ᶠᶠmat)
        copyto!(_temp6, 4I) # We can't directly use I in array broadcasts.
        @. _temp6 = _temp4 + _temp5 / 3 - _temp6
        mul!(_temp7, _temp1, _temp6)
        mul!(_temp8, _temp7, _ᶠᶠmat)
        mul!(_temp9, _ᶠᶜmat, _ᶜᶜmat)
        mul!(_temp10, _temp9, _ᶜᶠmat)
        @. _temp11 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
        mul!(_temp12, _temp11, _ᶠᶠmat)
        copyto!(_temp13, 4I) # We can't directly use I in array broadcasts.
        @. _temp13 = _temp10 * 2 - _temp12 + _temp13
        mul!(_temp14, _temp8, _temp13)
        mul!(_result, _temp14, _ᶠᶠmat)
    end,
    max_eps_error_limit = 70, # This case's roundoff error is large on GPUs.
)
