#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_copyto.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaCore.Geometry
import ClimaComms
using StaticArrays
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)

function test_copyto_float!(data)
    Random.seed!(1234)
    # Normally we'd use `similar` here, but https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(parent(data)))
    ArrayType = ClimaComms.array_type(ClimaComms.device())
    parent(rand_data) .=
        ArrayType(rand(eltype(parent(data)), DataLayouts.farray_size(data)))
    Base.copyto!(data, rand_data) # test copyto!(::AbstractData, ::AbstractData)
    @test all(parent(data) .== parent(rand_data))
    Base.copyto!(data, Base.Broadcast.broadcasted(+, rand_data, 1)) # test copyto!(::AbstractData, ::Broadcasted)
    @test all(parent(data) .== parent(rand_data) .+ 1)
end

function test_copyto!(data)
    Random.seed!(1234)
    # Normally we'd use `similar` here, but https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(parent(data)))
    ArrayType = ClimaComms.array_type(ClimaComms.device())
    parent(rand_data) .=
        ArrayType(rand(eltype(parent(data)), DataLayouts.farray_size(data)))
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
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(ArrayType{FT}, zeros)
    test_copyto_float!(data)
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_copyto_float!(data)
    data = IFH{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_copyto_float!(data)
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_copyto_float!(data)
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_copyto_float!(data)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_copyto_float!(data)
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_copyto_float!(data)
    data = VIFH{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_copyto_float!(data)
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_copyto_float!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_copyto_float!(data) # TODO: test
end

@testset "copyto! with Nf > 1" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = Tuple{FT, FT}
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(ArrayType{FT}, zeros)
    test_copyto!(data)
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_copyto!(data)
    data = IFH{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_copyto!(data)
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_copyto!(data)
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_copyto!(data)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_copyto!(data)
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_copyto!(data)
    data = VIFH{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_copyto!(data)
    # TODO: test this
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_copyto!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_copyto!(data) # TODO: test
end

@testset "copyto! views with Nf > 1" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    data_view(data) = DataLayouts.rebuild(
        data,
        SubArray(
            parent(data),
            ntuple(
                i -> Base.Slice(Base.OneTo(DataLayouts.farray_size(data, i))),
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
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_copyto!(data_view(data))
    data = IFH{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_copyto!(data_view(data))
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_copyto!(data_view(data))
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_copyto!(data_view(data))
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_copyto!(data_view(data))
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_copyto!(data_view(data))
    data = VIFH{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_copyto!(data_view(data))
    # TODO: test this
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_copyto!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_copyto!(data) # TODO: test
end
