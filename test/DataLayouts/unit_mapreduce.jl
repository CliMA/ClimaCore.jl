#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_mapreduce.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore
import ClimaComms
import Random
ClimaComms.@import_required_backends

device = ClimaComms.device()
context = ClimaComms.context(device)
ClimaComms.init(context)

function wrapper(context, fn, op, data)
    local_reduce = DataLayouts.mapreduce_cuda(fn, op, data)
    ClimaComms.allreduce!(context, parent(local_reduce), op)
    return local_reduce[]
end

"""test mapreduce with data layouts with 1 component"""
function test_mapreduce_1!(context, data)
    Random.seed!(1234)
    device = ClimaComms.device(context)
    ArrayType = ClimaComms.array_type(device)
    rand_data =
        ArrayType(rand(eltype(parent(data)), DataLayouts.farray_size(data)))
    parent(data) .= rand_data
    if device isa ClimaComms.CUDADevice
        @test wrapper(context, identity, min, data) == minimum(parent(data))
        @test wrapper(context, identity, max, data) == maximum(parent(data))
    else
        @test minimum(data) == minimum(parent(data))
        @test maximum(data) == maximum(parent(data))
    end
end

"""test mapreduce with data layouts with 2 components"""
function test_mapreduce_2!(context, data)
    Random.seed!(1234)
    device = ClimaComms.device(context)
    ArrayType = ClimaComms.array_type(device)
    rand_data =
        ArrayType(rand(eltype(parent(data)), DataLayouts.farray_size(data)))
    parent(data) .= rand_data
    # mapreduce orders tuples lexicographically:
    #    minimum(((2,3), (1,4))) # (1, 4)
    #    minimum(((1,4), (2,3))) # (1, 4)
    #    minimum(((4,1), (3,2))) # (3, 2)
    # so, for now, let's just assign the two components to match:
    parent(data.:2) .= parent(data.:1)
    # @test minimum(data) == (minimum(parent(data.:1)), minimum(parent(data.:2)))
    # @test maximum(data) == (maximum(parent(data.:1)), maximum(parent(data.:2)))
    if device isa ClimaComms.CUDADevice
        @test wrapper(context, identity, min, data.:1) ==
              minimum(parent(data.:1))
        @test wrapper(context, identity, max, data.:2) ==
              maximum(parent(data.:2))
    else
        @test minimum(data.:1) == minimum(parent(data.:1))
        @test minimum(data.:2) == minimum(parent(data.:2))
        @test maximum(data.:1) == maximum(parent(data.:1))
        @test maximum(data.:2) == maximum(parent(data.:2))
    end
end

@testset "mapreduce with Nf = 1" begin
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(ArrayType{FT}, zeros)
    test_mapreduce_1!(context, data)
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_mapreduce_1!(context, data)
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_mapreduce_1!(context, data)
    # data = IFH{S}(ArrayType{FT}, zeros; Ni,Nh);               test_mapreduce_1!(context, data)
    # data = IJF{S}(ArrayType{FT}, zeros; Nij);                 test_mapreduce_1!(context, data)
    # data = IF{S}(ArrayType{FT}, zeros; Ni);                   test_mapreduce_1!(context, data)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_mapreduce_1!(context, data)
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_mapreduce_1!(context, data)
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_mapreduce_1!(context, data)
    # data = VIFH{S}(ArrayType{FT}, zeros; Nv,Nij,Nh);                  test_mapreduce_1!(context, data)
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_mapreduce_1!(context, data) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);             test_mapreduce_1!(context, data) # TODO: test
end

@testset "mapreduce with Nf > 1" begin
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = Tuple{FT, FT}
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(ArrayType{FT}, zeros)
    test_mapreduce_2!(context, data)
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_mapreduce_2!(context, data)
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_mapreduce_2!(context, data)
    # data = IFH{S}(ArrayType{FT}, zeros; Ni,Nh);               test_mapreduce_2!(context, data)
    # data = IJF{S}(ArrayType{FT}, zeros; Nij);                 test_mapreduce_2!(context, data)
    # data = IF{S}(ArrayType{FT}, zeros; Ni);                   test_mapreduce_2!(context, data)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_mapreduce_2!(context, data)
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_mapreduce_2!(context, data)
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_mapreduce_2!(context, data)
    # data = VIFH{S}(ArrayType{FT}, zeros; Nv,Nij,Nh);                  test_mapreduce_2!(context, data)
    # TODO: test this
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_mapreduce_2!(context, data) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);             test_mapreduce_2!(context, data) # TODO: test
end

@testset "mapreduce views with Nf > 1" begin
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
    data = DataF{S}(ArrayType{FT}, zeros)
    test_mapreduce_2!(context, data_view(data))
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_mapreduce_2!(context, data_view(data))
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_mapreduce_2!(context, data_view(data))
    # data = IFH{S}(ArrayType{FT}, zeros; Ni,Nh);               test_mapreduce_2!(context, data_view(data))
    # data = IJF{S}(ArrayType{FT}, zeros; Nij);                 test_mapreduce_2!(context, data_view(data))
    # data = IF{S}(ArrayType{FT}, zeros; Ni);                   test_mapreduce_2!(context, data_view(data))
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_mapreduce_2!(context, data_view(data))
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_mapreduce_2!(context, data_view(data))
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_mapreduce_2!(context, data_view(data))
    # data = VIFH{S}(ArrayType{FT}, zeros; Nv,Nij,Nh);                  test_mapreduce_2!(context, data_view(data))
    # TODO: test this
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_mapreduce_2!(context, data_view(data)) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);             test_mapreduce_2!(context, data_view(data)) # TODO: test
end
