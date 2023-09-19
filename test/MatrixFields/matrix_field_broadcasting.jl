import BandedMatrices: band
import LinearAlgebra: I, mul!

include("matrix_field_test_utils.jl")

@testset "Scalar Matrix Field Broadcasting" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)

    seed!(1) # ensures reproducibility
    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    ᶜᶜmat = random_field(DiagonalMatrixRow{FT}, center_space)
    ᶜᶠmat = random_field(BidiagonalMatrixRow{FT}, center_space)
    ᶠᶠmat = random_field(TridiagonalMatrixRow{FT}, face_space)
    ᶠᶜmat = random_field(QuaddiagonalMatrixRow{FT}, face_space)

    test_field_broadcast_against_array_reference(;
        test_name = "diagonal matrix times vector",
        get_result = () -> (@. ᶜᶜmat ⋅ ᶜvec),
        set_result! = result -> (@. result = ᶜᶜmat ⋅ ᶜvec),
        input_fields = (ᶜᶜmat, ᶜvec),
        ref_set_result! = (_result, _ᶜᶜmat, _ᶜvec) ->
            mul!(_result, _ᶜᶜmat, _ᶜvec),
    )

    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30
    test_field_broadcast_against_array_reference(;
        test_name = "tri-diagonal matrix times vector",
        get_result = () -> (@. ᶠᶠmat ⋅ ᶠvec),
        set_result! = result -> (@. result = ᶠᶠmat ⋅ ᶠvec),
        input_fields = (ᶠᶠmat, ᶠvec),
        ref_set_result! = (_result, _ᶠᶠmat, _ᶠvec) ->
            mul!(_result, _ᶠᶠmat, _ᶠvec),
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "quad-diagonal matrix times vector",
        get_result = () -> (@. ᶠᶜmat ⋅ ᶜvec),
        set_result! = result -> (@. result = ᶠᶜmat ⋅ ᶜvec),
        input_fields = (ᶠᶜmat, ᶜvec),
        ref_set_result! = (_result, _ᶠᶜmat, _ᶜvec) ->
            mul!(_result, _ᶠᶜmat, _ᶜvec),
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix",
        get_result = () -> (@. ᶜᶜmat ⋅ ᶜᶠmat),
        set_result! = result -> (@. result = ᶜᶜmat ⋅ ᶜᶠmat),
        input_fields = (ᶜᶜmat, ᶜᶠmat),
        ref_set_result! = (_result, _ᶜᶜmat, _ᶜᶠmat) ->
            mul!(_result, _ᶜᶜmat, _ᶜᶠmat),
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "tri-diagonal matrix times tri-diagonal matrix",
        get_result = () -> (@. ᶠᶠmat ⋅ ᶠᶠmat),
        set_result! = result -> (@. result = ᶠᶠmat ⋅ ᶠᶠmat),
        input_fields = (ᶠᶠmat,),
        ref_set_result! = (_result, _ᶠᶠmat) -> mul!(_result, _ᶠᶠmat, _ᶠᶠmat),
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "quad-diagonal matrix times diagonal matrix",
        get_result = () -> (@. ᶠᶜmat ⋅ ᶜᶜmat),
        set_result! = result -> (@. result = ᶠᶜmat ⋅ ᶜᶜmat),
        input_fields = (ᶠᶜmat, ᶜᶜmat),
        ref_set_result! = (_result, _ᶠᶜmat, _ᶜᶜmat) ->
            mul!(_result, _ᶠᶜmat, _ᶜᶜmat),
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix",
        get_result = () -> (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat),
        set_result! = result -> (@. result = ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_temp_value_fields = () ->
            ((@. ᶜᶜmat ⋅ ᶜᶠmat), (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat)),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
        ) -> begin
            mul!(_temp1, _ᶜᶜmat, _ᶜᶠmat)
            mul!(_temp2, _temp1, _ᶠᶠmat)
            mul!(_result, _temp2, _ᶠᶜmat)
        end,
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix, but with \
                     forced right-associativity",
        get_result = () -> (@. ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
        set_result! = result -> (@. result = ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_temp_value_fields = () ->
            ((@. ᶠᶠmat ⋅ ᶠᶜmat), (@. ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
        ) -> begin
            mul!(_temp1, _ᶠᶠmat, _ᶠᶜmat)
            mul!(_temp2, _ᶜᶠmat, _temp1)
            mul!(_result, _ᶜᶜmat, _temp2)
        end,
        test_broken_with_cuda = true, # TODO: Fix this.
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix times \
                     vector",
        get_result = () -> (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec),
        set_result! = result ->
            (@. result = ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec),
        get_temp_value_fields = () -> (
            (@. ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat),
            (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _ᶜvec,
            _temp1,
            _temp2,
            _temp3,
        ) -> begin
            mul!(_temp1, _ᶜᶜmat, _ᶜᶠmat)
            mul!(_temp2, _temp1, _ᶠᶠmat)
            mul!(_temp3, _temp2, _ᶠᶜmat)
            mul!(_result, _temp3, _ᶜvec)
        end,
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix times \
                     vector, but with forced right-associativity",
        get_result = () -> (@. ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)))),
        set_result! = result ->
            (@. result = ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)))),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec),
        get_temp_value_fields = () -> (
            (@. ᶠᶜmat ⋅ ᶜvec),
            (@. ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)),
            (@. ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec))),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _ᶜvec,
            _temp1,
            _temp2,
            _temp3,
        ) -> begin
            mul!(_temp1, _ᶠᶜmat, _ᶜvec)
            mul!(_temp2, _ᶠᶠmat, _temp1)
            mul!(_temp3, _ᶜᶠmat, _temp2)
            mul!(_result, _ᶜᶜmat, _temp3)
        end,
        time_ratio_limit = 15, # This case's ref function is fast on Buildkite.
        test_broken_with_cuda = true, # TODO: Fix this.
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "linear combination of matrix products and LinearAlgebra.I",
        get_result = () ->
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        set_result! = result ->
            (@. result = 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_temp_value_fields = () -> (
            (@. 2 * ᶠᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat ⋅ ᶠᶠmat),
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
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_result, 4I) # We can't directly use I in array broadcasts.
            @. _result = _temp3 + _temp4 / 3 - _result
        end,
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "another linear combination of matrix products and \
                     LinearAlgebra.I",
        get_result = () ->
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
        set_result! = result -> (@. result =
            ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_temp_value_fields = () -> (
            (@. ᶠᶜmat ⋅ ᶜᶜmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat / 3),
            (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
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
        ) -> begin
            mul!(_temp1, _ᶠᶜmat, _ᶜᶜmat)
            mul!(_temp2, _temp1, _ᶜᶠmat)
            @. _temp3 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
            mul!(_temp4, _temp3, _ᶠᶠmat)
            copyto!(_result, 4I) # We can't directly use I in array broadcasts.
            @. _result = _temp2 * 2 - _temp4 + _result
        end,
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "matrix times linear combination",
        get_result = () -> (@. ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,))),
        set_result! = result -> (@. result =
            ᶜᶠmat ⋅ (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,))),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_temp_value_fields = () -> (
            (@. 2 * ᶠᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat ⋅ ᶠᶠmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
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
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_temp5, 4I) # We can't directly use I in array broadcasts.
            @. _temp5 = _temp3 + _temp4 / 3 - _temp5
            mul!(_result, _ᶜᶠmat, _temp5)
        end,
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "linear combination times another linear combination",
        get_result = () ->
            (@. (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
                (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))),
        set_result! = result -> (@. result =
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))),
        input_fields = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_temp_value_fields = () -> (
            (@. 2 * ᶠᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat ⋅ ᶠᶠmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
            (@. ᶠᶜmat ⋅ ᶜᶜmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat / 3),
            (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
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
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_temp5, 4I) # We can't directly use I in array broadcasts.
            @. _temp5 = _temp3 + _temp4 / 3 - _temp5
            mul!(_temp6, _ᶠᶜmat, _ᶜᶜmat)
            mul!(_temp7, _temp6, _ᶜᶠmat)
            @. _temp8 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
            mul!(_temp9, _temp8, _ᶠᶠmat)
            copyto!(_temp10, 4I) # We can't directly use I in array broadcasts.
            @. _temp10 = _temp7 * 2 - _temp9 + _temp10
            mul!(_result, _temp5, _temp10)
        end,
        max_eps_error_limit = 30, # This case's roundoff error is large on GPUs.
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "matrix times matrix times linear combination times matrix \
                     times another linear combination times matrix",
        get_result = () -> (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)) ⋅
            ᶠᶠmat),
        set_result! = result -> (@. result =
            ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)) ⋅
            ᶠᶠmat),
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
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast_against_array_reference(;
        test_name = "matrix constructions and multiplications",
        get_result = () ->
            (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
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
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30
end

GC.gc();
@info "mem usage" rss = Sys.maxrss() / 2^30;

@testset "Non-scalar Matrix Field Broadcasting" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)

    ᶜlg = Fields.local_geometry_field(center_space)
    ᶠlg = Fields.local_geometry_field(face_space)

    seed!(1) # ensures reproducibility
    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    ᶜᶠmat = random_field(BidiagonalMatrixRow{FT}, center_space)
    ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space)
    ᶜᶠmat3 = random_field(BidiagonalMatrixRow{FT}, center_space)
    ᶠᶜmat = random_field(QuaddiagonalMatrixRow{FT}, face_space)
    ᶠᶜmat2 = random_field(QuaddiagonalMatrixRow{FT}, face_space)
    ᶠᶜmat3 = random_field(QuaddiagonalMatrixRow{FT}, face_space)

    ᶜᶠmat_AC1 = map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶜᶠmat)
    ᶜᶠmat_C12 = map(
        (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
        ᶜᶠmat2,
        ᶜᶠmat3,
    )
    ᶠᶜmat_AC1 = map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶠᶜmat)
    ᶠᶜmat_C12 = map(
        (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
        ᶠᶜmat2,
        ᶠᶜmat3,
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast(;
        test_name = "matrix of covectors times matrix of vectors",
        get_result = () -> (@. ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12),
        set_result! = result -> (@. result = ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12),
        ref_set_result! = result -> (@. result =
            ᶜᶠmat ⋅ (
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) ⋅ ᶠᶜmat2 +
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) ⋅ ᶠᶜmat3
            )),
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    test_field_broadcast(;
        test_name = "matrix of covectors times matrix of vectors times matrix \
                     of numbers times matrix of covectors times matrix of \
                     vectors",
        get_result = () ->
            (@. ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12 ⋅ ᶜᶠmat ⋅ ᶠᶜmat_AC1 ⋅ ᶜᶠmat_C12),
        set_result! = result ->
            (@. result = ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12 ⋅ ᶜᶠmat ⋅ ᶠᶜmat_AC1 ⋅ ᶜᶠmat_C12),
        ref_set_result! = result -> (@. result =
            ᶜᶠmat ⋅ (
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) ⋅ ᶠᶜmat2 +
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) ⋅ ᶠᶜmat3
            ) ⋅ ᶜᶠmat ⋅ ᶠᶜmat ⋅ (
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:1) ⋅ ᶜᶠmat2 +
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:2) ⋅ ᶜᶠmat3
            )),
    )
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    ᶜᶠmat_AC1_num =
        map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat_AC1, ᶜᶠmat)
    ᶜᶠmat_num_C12 =
        map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat, ᶜᶠmat_C12)
    ᶠᶜmat_C12_AC1 =
        map((row1, row2) -> map(tuple, row1, row2), ᶠᶜmat_C12, ᶠᶜmat_AC1)

    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30
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
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30

    ᶜvec_NT = @. nested_type(ᶜvec, ᶜvec, ᶜvec)
    ᶜᶠmat_NT =
        map((rows...) -> map(nested_type, rows...), ᶜᶠmat, ᶜᶠmat2, ᶜᶠmat3)
    ᶠᶜmat_NT =
        map((rows...) -> map(nested_type, rows...), ᶠᶜmat, ᶠᶜmat2, ᶠᶜmat3)

    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30
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
    GC.gc()
    @info "mem usage" rss = Sys.maxrss() / 2^30
end

GC.gc();
@info "mem usage" rss = Sys.maxrss() / 2^30;
