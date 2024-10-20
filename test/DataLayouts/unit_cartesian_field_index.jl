#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_cartesian_field_index.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore.DataLayouts: CartesianFieldIndex
using ClimaCore.DataLayouts: to_data_specific_field, singleton
import ClimaCore.Geometry
import ClimaComms
using StaticArrays
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)

universal_axes(data) =
    map(size(data)) do i
        s =
            i == DataLayouts.field_dim(singleton(data)) ?
            DataLayouts.ncomponents(data) : i
        Base.OneTo(s)
    end

universal_field_index(I::CartesianIndex, f) =
    CartesianIndex(map(i -> i == 3 ? f : i, I.I))

function test_copyto_float!(data)
    Random.seed!(1234)
    # Normally we'd use `similar` here, but https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(parent(data)))
    ArrayType = ClimaComms.array_type(ClimaComms.device())
    FT = eltype(parent(data))
    parent(rand_data) .= ArrayType(rand(FT, DataLayouts.farray_size(data)))
    # For a float, CartesianIndex and CartesianFieldIndex return the same thing
    for I in CartesianIndices(universal_axes(data))
        CI = CartesianFieldIndex(I.I)
        @test data[CI] == data[I]
    end
    for I in CartesianIndices(universal_axes(data))
        CI = CartesianFieldIndex(I.I)
        data[CI] = FT(prod(I.I))
    end
    for I in CartesianIndices(universal_axes(data))
        CI = CartesianFieldIndex(I.I)
        @test data[CI] == prod(I.I)
    end
end

function test_copyto!(data)
    Random.seed!(1234)
    # Normally we'd use `similar` here, but https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(parent(data)))
    ArrayType = ClimaComms.array_type(ClimaComms.device())
    FT = eltype(parent(data))
    parent(rand_data) .= ArrayType(rand(FT, DataLayouts.farray_size(data)))

    for I in CartesianIndices(universal_axes(data))
        for f in 1:DataLayouts.ncomponents(data)
            UFI = universal_field_index(I, f)
            DSI = CartesianIndex(to_data_specific_field(singleton(data), UFI.I))
            @test data[CartesianFieldIndex(UFI)] == parent(data)[DSI]
        end
    end

    for I in CartesianIndices(universal_axes(data))
        for f in 1:DataLayouts.ncomponents(data)
            UFI = universal_field_index(I, f)
            DSI = CartesianIndex(to_data_specific_field(singleton(data), UFI.I))
            val = parent(data)[DSI]
            data[CartesianFieldIndex(UFI)] = val + 1
            @test parent(data)[DSI] == val + 1
        end
    end
end

@testset "CartesianFieldIndex with Nf = 1" begin
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

@testset "CartesianFieldIndex with Nf > 1" begin
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

@testset "CartesianFieldIndex views with Nf > 1" begin
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
