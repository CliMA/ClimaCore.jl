#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_mapreduce.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaComms
import Random
ClimaComms.@import_required_backends

"""test mapreduce with data layouts with 1 component"""
function test_mapreduce_1!(device, data)
    device isa ClimaComms.CUDADevice && return nothing
    Random.seed!(1234)
    ArrayType = ClimaComms.array_type(device)
    rand_data = ArrayType(rand(eltype(parent(data)), size(parent(data))))
    parent(data) .= rand_data
    @test minimum(data) == minimum(parent(data))
    @test maximum(data) == maximum(parent(data))
end

"""test mapreduce with data layouts with 2 components"""
function test_mapreduce_2!(device, data)
    device isa ClimaComms.CUDADevice && return nothing
    Random.seed!(1234)
    ArrayType = ClimaComms.array_type(device)
    rand_data = ArrayType(rand(eltype(parent(data)), size(parent(data))))
    parent(data) .= rand_data
    # mapreduce orders tuples lexicographically:
    #    minimum(((2,3), (1,4))) # (1, 4)
    #    minimum(((1,4), (2,3))) # (1, 4)
    #    minimum(((4,1), (3,2))) # (3, 2)
    # so, for now, let's just assign the two components to match:
    parent(data.:2) .= parent(data.:1)
    @test minimum(data) == (minimum(parent(data.:1)), minimum(parent(data.:2)))
    @test maximum(data) == (maximum(parent(data.:1)), maximum(parent(data.:2)))
end

@testset "mapreduce with Nf = 1" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = FT
    Nf = 1
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
#! format: off
    data = DataF{S}(device_zeros(FT,Nf));                        test_mapreduce_1!(device, data)
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         test_mapreduce_1!(device, data)
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              test_mapreduce_1!(device, data)
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_mapreduce_1!(device, data)
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_mapreduce_1!(device, data)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_mapreduce_1!(device, data)
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); test_mapreduce_1!(device, data)
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_mapreduce_1!(device, data)
#! format: on
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_mapreduce_1!(device, data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_mapreduce_1!(device, data) # TODO: test
end

@testset "mapreduce with Nf > 1" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = Tuple{FT, FT}
    Nf = 2
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
#! format: off
    data = DataF{S}(device_zeros(FT,Nf));                        test_mapreduce_2!(device, data)
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         test_mapreduce_2!(device, data)
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              test_mapreduce_2!(device, data)
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_mapreduce_2!(device, data)
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_mapreduce_2!(device, data)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_mapreduce_2!(device, data)
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); test_mapreduce_2!(device, data)
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_mapreduce_2!(device, data)
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_mapreduce_2!(device, data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_mapreduce_2!(device, data) # TODO: test
end

@testset "mapreduce views with Nf > 1" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    data_view(data) = DataLayouts.rebuild(
        data,
        SubArray(
            parent(data),
            ntuple(i -> Base.OneTo(size(parent(data), i)), ndims(data)),
        ),
    )
    FT = Float64
    S = Tuple{FT, FT}
    Nf = 2
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
    # Rather than using level/slab/column, let's just make views/SubArrays
    # directly so that we can easily test all cases:
#! format: off
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         test_mapreduce_2!(device, data_view(data))
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              test_mapreduce_2!(device, data_view(data))
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_mapreduce_2!(device, data_view(data))
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_mapreduce_2!(device, data_view(data))
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_mapreduce_2!(device, data_view(data))
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); test_mapreduce_2!(device, data_view(data))
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_mapreduce_2!(device, data_view(data))
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_mapreduce_2!(device, data_view(data)) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_mapreduce_2!(device, data_view(data)) # TODO: test
end
