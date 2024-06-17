#=
julia --check-bounds=yes --project
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
function test_fill2!(data, val::Real, fdim)
    println("------------ $(typeof(data))")
    @test all(parent(parent(data)) .== 0)
    fill!(data, val)
    subarray = parent(data)
    @show typeof(subarray)
    array = parent(subarray)
    @show typeof(array)
    @show size(array)
    dim1 = ntuple(i -> i == fdim ? 1 : Colon(), ndims(data))
    dim2 = ntuple(i -> i == fdim ? 2 : Colon(), ndims(data))
    @show dim1
    @show dim2
    @test all(Array(array)[dim1...] .== 0)
    @test all(Array(array)[dim2...] .== val)
end

@testset "fill! with Nf = 1" begin
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
    data = DataF{S}(device_zeros(FT,Nf));                        test_fill!(data, 3)
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         test_fill!(data, 3)
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              test_fill!(data, 3)
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_fill!(data, 3)
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_fill!(data, 3)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_fill!(data, 3)
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); test_fill!(data, 3)
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_fill!(data, 3)
#! format: on
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_fill!(data, 3) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_fill!(data, 3) # TODO: test
end

@testset "fill! with Nf > 1" begin
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
    data = DataF{S}(device_zeros(FT,Nf));                        test_fill!(data, (2,3))
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         test_fill!(data, (2,3))
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              test_fill!(data, (2,3))
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_fill!(data, (2,3))
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_fill!(data, (2,3))
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_fill!(data, (2,3))
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); test_fill!(data, (2,3))
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_fill!(data, (2,3))
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_fill!(data, (2,3)) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_fill!(data, (2,3)) # TODO: test
end

@testset "fill! SubArrays with Nf > 1" begin
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
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         test_fill!(data_view(data), (2,3))
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              test_fill!(data_view(data), (2,3))
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_fill!(data_view(data), (2,3))
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_fill!(data_view(data), (2,3))
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_fill!(data_view(data), (2,3))
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); test_fill!(data_view(data), (2,3))
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_fill!(data_view(data), (2,3))
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_fill!(data, (2,3)) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_fill!(data, (2,3)) # TODO: test
end

@testset "fill! getproperty(::DataLayouts, ::Symbol) with Nf > 1" begin
    device = ClimaComms.device()
    field_dim(::Type{<:IJFH}) = 3
    field_dim(::Type{<:IFH}) = 2
    field_dim(::Type{<:IJF}) = 3
    field_dim(::Type{<:IF}) = 2
    field_dim(::Type{<:VF}) = 2
    field_dim(::Type{<:VIJFH}) = 4
    field_dim(::Type{<:VIFH}) = 3
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = Tuple{FT, FT, FT}
    Nf = 3
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
    # Rather than using level/slab/column, let's just make SubArrays
    # directly so that we can easily test all cases:
#! format: off
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         test_fill2!(data.:2, 2, field_dim(IJFH))
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              test_fill2!(data.:2, 2, field_dim(IFH))
    # data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             test_fill2!(data.:2, 2, field_dim(IJF)) # broken on gpu
    # data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  test_fill2!(data.:2, 2, field_dim(IF)) # broken on gpu
    # data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    test_fill2!(data.:2, 2, field_dim(VF)) # broken on gpu
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); test_fill2!(data.:2, 2, field_dim(VIJFH))
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      test_fill2!(data.:2, 2, field_dim(VIFH))
#! format: on
    # TODO: test this
    # data = DataLayouts.IJKFVH{S, Nij, Nk}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_fill2!(data.:2, 2, field_dim(IJKFVH)) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));             test_fill2!(data.:2, 2, field_dim(IH1JH2)) # TODO: test
end
