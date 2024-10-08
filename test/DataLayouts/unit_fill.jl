#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_fill.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaComms
ClimaComms.@import_required_backends

function test_fill!(data, vals::Tuple{<:Any, <:Any})
    fill!(data, vals)
    @test all(parent(data.:1) .== vals[1])
    @test all(parent(data.:2) .== vals[2])
end
function test_fill!(data, val::Real)
    fill!(data, val)
    @test all(parent(data) .== val)
end

@testset "fill! with Nf = 1" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6

    data = DataF{S}(ArrayType{FT}, zeros)
    test_fill!(data, 3)
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_fill!(data, 3)
    data = IHF{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_fill!(data, 3)
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_fill!(data, 3)
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_fill!(data, 3)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_fill!(data, 3)
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_fill!(data, 3)
    data = VIHF{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_fill!(data, 3)

    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_fill!(data, 3) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_fill!(data, 3) # TODO: test
end

@testset "fill! with Nf > 1" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = Tuple{FT, FT}
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6

    data = DataF{S}(ArrayType{FT}, zeros)
    test_fill!(data, (2, 3))
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_fill!(data, (2, 3))
    data = IHF{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_fill!(data, (2, 3))
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_fill!(data, (2, 3))
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_fill!(data, (2, 3))
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_fill!(data, (2, 3))
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_fill!(data, (2, 3))
    data = VIHF{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_fill!(data, (2, 3))

    # TODO: test this
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_fill!(data, (2,3)) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_fill!(data, (2,3)) # TODO: test
end

@testset "fill! views with Nf > 1" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    data_view(data) = DataLayouts.rebuild(
        data,
        SubArray(
            parent(data),
            ntuple(
                i -> Base.OneTo(DataLayouts.farray_size(data, i)),
                ndims(data),
            ),
        ),
    )
    FT = Float64
    S = Tuple{FT, FT}
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    # Rather than using level/slab/column, let's just make views/SubArrays
    # directly so that we can easily test all cases:

    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_fill!(data_view(data), (2, 3))
    data = IHF{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_fill!(data_view(data), (2, 3))
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_fill!(data_view(data), (2, 3))
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_fill!(data_view(data), (2, 3))
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_fill!(data_view(data), (2, 3))
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_fill!(data_view(data), (2, 3))
    data = VIHF{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_fill!(data_view(data), (2, 3))

    # TODO: test this
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_fill!(data, (2,3)) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_fill!(data, (2,3)) # TODO: test
end

@testset "Reshaped Arrays" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    function reshaped_array(data2)
        # `reshape` does not always return a `ReshapedArray`, which
        # we need to specialize on to correctly dispatch when its
        # parent array is backed by a CuArray. So, let's first
        # In order to get a ReshapedArray back, let's first create view
        # via `data.:2`. This doesn't guarantee that the result is a
        # ReshapedArray, but it works for several cases. Tests when
        # are commented out for cases when Julia Base manages to return
        # a parent-similar array.

        # After moving from FH -> HF, we no longer make
        # `Base.ReshapedArray`s, because field views
        # simply return arrays.
        data = data.:2
        array₀ = DataLayouts.data2array(data)
        @test typeof(array₀) <: Base.AbstractArray
        rdata = DataLayouts.array2data(array₀, data)
        newdata = DataLayouts.rebuild(
            data,
            SubArray(
                parent(rdata),
                ntuple(
                    i -> Base.OneTo(DataLayouts.farray_size(rdata, i)),
                    ndims(rdata),
                ),
            ),
        )
        rarray = parent(parent(newdata))
        @test typeof(rarray) <: Base.AbstractArray
        subarray = parent(rarray)
        @test typeof(subarray) <: Base.AbstractArray
        array = parent(subarray)
        newdata
    end
    FT = Float64
    S = Tuple{FT, FT} # need at least 2 components to make a SubArray
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    # directly so that we can easily test all cases:

    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_fill!(reshaped_array(data), 2)
    data = IHF{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_fill!(reshaped_array(data), 2)
    # data = IJF{S}(ArrayType{FT}, zeros; Nij);          test_fill!(reshaped_array(data), 2)
    # data = IF{S}(ArrayType{FT}, zeros; Ni);            test_fill!(reshaped_array(data), 2)
    # data = VF{S}(ArrayType{FT}, zeros; Nv);            test_fill!(reshaped_array(data), 2)
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_fill!(reshaped_array(data), 2)
    data = VIHF{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_fill!(reshaped_array(data), 2)

    # TODO: test this
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_fill!(reshaped_array(data), 2) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_fill!(reshaped_array(data), 2) # TODO: test
end
