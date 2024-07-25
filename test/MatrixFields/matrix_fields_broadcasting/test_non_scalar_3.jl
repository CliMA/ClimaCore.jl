if !(@isdefined(test_field_broadcast))
    include("test_non_scalar_utils.jl")
end

test_field_broadcast(;
    test_name = "matrix of covectors and numbers times matrix of vectors \
                 and covectors times matrix of numbers and vectors times \
                 vector of numbers",
    get_result = () ->
        (@. ᶜᶠmat_AC1_num ⋅ ᶠᶜmat_C12_AC1 ⋅ ᶜᶠmat_num_C12 ⋅ ᶠvec),
    set_result! = result ->
        (@. result = ᶜᶠmat_AC1_num ⋅ ᶠᶜmat_C12_AC1 ⋅ ᶜᶠmat_num_C12 ⋅ ᶠvec),
    ref_set_result! = result -> (@. result = tuple(
        ᶜᶠmat ⋅ (
            DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) ⋅ ᶠᶜmat2 +
            DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) ⋅ ᶠᶜmat3
        ) ⋅ ᶜᶠmat ⋅ ᶠvec,
        ᶜᶠmat ⋅ ᶠᶜmat ⋅ (
            DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:1) ⋅ ᶜᶠmat2 +
            DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:2) ⋅ ᶜᶠmat3
        ) ⋅ ᶠvec,
    )),
)
