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
    copyto!(parent(data), rand_data)
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
    copyto!(parent(data), rand_data)
    # mapreduce orders tuples lexicographically:
    #    minimum(((2,3), (1,4))) # (1, 4)
    #    minimum(((1,4), (2,3))) # (1, 4)
    #    minimum(((4,1), (3,2))) # (3, 2)
    # so, for now, let's just assign the two components to match:
    copyto!(parent(data.:2), parent(data.:1))
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
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = FT
    Nf = 1
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
#! format: off
    data = DataF{S}(device_zeros(FT,Nf));                              test_mapreduce_1!(context, data)
    data = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));           test_mapreduce_1!(context, data)
    # data = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));              test_mapreduce_1!(context, data)
    # data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));                 test_mapreduce_1!(context, data)
    # data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                      test_mapreduce_1!(context, data)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                          test_mapreduce_1!(context, data)
    data = VIJFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));   test_mapreduce_1!(context, data)
    # data = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_mapreduce_1!(context, data)
#! format: on
    # data = DataLayouts.IJKFVH{S, Nij, Nk, Nh}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_mapreduce_1!(context, data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));                 test_mapreduce_1!(context, data) # TODO: test
end

@testset "mapreduce with Nf > 1" begin
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = Tuple{FT, FT}
    Nf = 2
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
#! format: off
    data = DataF{S}(device_zeros(FT,Nf));                              test_mapreduce_2!(context, data)
    data = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));           test_mapreduce_2!(context, data)
    # data = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));              test_mapreduce_2!(context, data)
    # data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));                 test_mapreduce_2!(context, data)
    # data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                      test_mapreduce_2!(context, data)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                          test_mapreduce_2!(context, data)
    data = VIJFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));   test_mapreduce_2!(context, data)
    # data = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_mapreduce_2!(context, data)
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk, Nh}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_mapreduce_2!(context, data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));                 test_mapreduce_2!(context, data) # TODO: test
end
