#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_copyto.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore.DataLayouts: elsize
import ClimaCore.Geometry
import ClimaComms
using StaticArrays
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)

function test_copyto_float!(data)
    pdata = parent(data)
    Random.seed!(1234)
    # Normally we'd use `similar` here, but https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(pdata))
    ArrayType = ClimaComms.array_type(ClimaComms.device())
    parent(rand_data).arrays[1] .= ArrayType(rand(eltype(pdata), elsize(pdata)))
    Base.copyto!(data, rand_data) # test copyto!(::AbstractData, ::AbstractData)
    @test all(pdata.arrays[1] .== parent(rand_data).arrays[1])
    Base.copyto!(data, Base.Broadcast.broadcasted(+, rand_data, 1)) # test copyto!(::AbstractData, ::Broadcasted)
    @test all(pdata .== parent(rand_data) .+ 1)
end

function test_copyto!(data)
    Random.seed!(1234)
    # Normally we'd use `similar` here, but https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(parent(data)))
    ArrayType = ClimaComms.array_type(ClimaComms.device())
    parent(rand_data) .=
        ArrayType(rand(eltype(parent(data)), size(parent(data))))
    Base.copyto!(data, rand_data) # test copyto!(::AbstractData, ::AbstractData)
    @test all(parent(data.:1) .== parent(rand_data.:1))
    @test all(parent(data.:2) .== parent(rand_data.:2))
    @test all(parent(data) .== parent(rand_data))
    Base.copyto!(data.:1, Base.Broadcast.broadcasted(+, rand_data.:1, 1)) # test copyto!(::AbstractData, ::Broadcasted)
    Base.copyto!(data.:2, Base.Broadcast.broadcasted(+, rand_data.:2, 1)) # test copyto!(::AbstractData, ::Broadcasted)
    @test all(parent(data.:1) .== parent(rand_data.:1) .+ 1)
    @test all(parent(data.:2) .== parent(rand_data.:2) .+ 1)
end

@testset "copyto! with Nf = 1" begin
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
    data = DataF{S}(device_zeros(FT,Nf));                        test_copyto_float!(data)
    data = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));     test_copyto_float!(data)
    data = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));          test_copyto_float!(data)
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_copyto_float!(data)
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_copyto_float!(data)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_copyto_float!(data)
    data = VIJFH{S,Nv,Nij,Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));test_copyto_float!(data)
    data = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));  test_copyto_float!(data)
#! format: on
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_copyto_float!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_copyto_float!(data) # TODO: test
end

@testset "copyto! with Nf > 1" begin
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
    data = DataF{S}(device_zeros(FT,Nf));                        test_copyto!(data)
    data = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));     test_copyto!(data)
    data = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));          test_copyto!(data)
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_copyto!(data)
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_copyto!(data)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_copyto!(data)
    data = VIJFH{S,Nv,Nij,Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));test_copyto!(data)
    data = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));  test_copyto!(data)
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_copyto!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_copyto!(data) # TODO: test
end

@testset "copyto! views with Nf > 1" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    data_view(data) = DataLayouts.rebuild(
        data,
        SubArray(
            parent(data),
            ntuple(
                i -> Base.Slice(Base.OneTo(size(parent(data), i))),
                ndims(data),
            ),
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
    data = IJFH{S, Nij, Nh}(device_zeros(FT,Nij,Nij,Nf,Nh));     test_copyto!(data_view(data))
    data = IFH{S, Nij, Nh}(device_zeros(FT,Nij,Nf,Nh));          test_copyto!(data_view(data))
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_copyto!(data_view(data))
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_copyto!(data_view(data))
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_copyto!(data_view(data))
    data = VIJFH{S,Nv,Nij,Nh}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh));test_copyto!(data_view(data))
    data = VIFH{S, Nv, Nij, Nh}(device_zeros(FT,Nv,Nij,Nf,Nh));  test_copyto!(data_view(data))
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_copyto!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_copyto!(data) # TODO: test
end
