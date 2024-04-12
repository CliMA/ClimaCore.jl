if !(@isdefined(test_field_broadcast_against_array_reference))
    include("test_scalar_utils.jl")
end

test_field_broadcast_against_array_reference(;
    test_name = "matrix constructions and multiplications",
    get_result = () -> (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
        TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat ⋅
        DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)),
    set_result! = result -> (@. result =
        BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
        TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat ⋅
        DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)),
    input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec),
    get_temp_value_fields = () -> (
        (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec)),
        (@. TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1)),
        (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
            TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1)),
        (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
            TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat),
        (@. DiagonalMatrixRow(ᶠvec)),
        (@. DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)),
    ),
    ref_set_result! = (
        _result,
        _ᶜᶜmat,
        _ᶜᶠmat,
        _ᶠᶠmat,
        _ᶠᶜmat,
        _ᶜvec,
        _ᶠvec,
        _temp1,
        _temp2,
        _temp3,
        _temp4,
        _temp5,
        _temp6,
    ) -> begin
        mul!(view(_temp1, band(0)), _ᶜᶠmat, _ᶠvec)
        mul!(view(_temp1, band(1)), _ᶜᶜmat, _ᶜvec)
        copyto!(view(_temp2, band(-1)), 1, _ᶠvec, 2)
        mul!(view(_temp2, band(0)), _ᶠᶜmat, _ᶜvec)
        fill!(view(_temp2, band(1)), 1)
        mul!(_temp3, _temp1, _temp2)
        mul!(_temp4, _temp3, _ᶠᶠmat)
        copyto!(view(_temp5, band(0)), 1, _ᶠvec, 1)
        mul!(view(_temp6, band(0)), _temp5, _ᶠvec)
        mul!(_result, _temp4, _temp6)
    end,
)
