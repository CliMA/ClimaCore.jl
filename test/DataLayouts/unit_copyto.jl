#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_copyto.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaCore.RecursiveApply: ⊞
import ClimaCore.Geometry
import ClimaComms
using StaticArrays
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)

all_layouts(ArrayType, S; Ni = 3, Nij = 3, Nv = 4, Nh = 5, Nk = 6) = (
    DataF{S}(ArrayType, zeros),
    VF{S}(ArrayType, zeros; Nv),
    IF{S}(ArrayType, zeros; Ni),
    IJF{S}(ArrayType, zeros; Nij),
    IFH{S}(ArrayType, zeros; Ni, Nh),
    IHF{S}(ArrayType, zeros; Ni, Nh),
    IJFH{S}(ArrayType, zeros; Nij, Nh),
    IJHF{S}(ArrayType, zeros; Nij, Nh),
    VIFH{S}(ArrayType, zeros; Nv, Ni, Nh),
    VIHF{S}(ArrayType, zeros; Nv, Ni, Nh),
    VIJFH{S}(ArrayType, zeros; Nv, Nij, Nh),
    VIJHF{S}(ArrayType, zeros; Nv, Nij, Nh),
    # DataLayouts.IJKFVH{S}(ArrayType, zeros; Nij, Nk, Nv, Nh),
    # DataLayouts.IH1JH2{S}(ArrayType, zeros; Nij),
)

function test_copyto_single_F!(data)
    # Avoid using similar here due to https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(parent(data)))
    Random.rand!(parent(rand_data))
    to_data(array) = DataLayouts.bitcast_struct.(eltype(data), array)

    Base.copyto!(data, rand_data)
    @test all(to_data(parent(data)) .== to_data(parent(rand_data)))

    Base.copyto!(data, Base.Broadcast.broadcasted(⊞, rand_data, 0x1))
    @test all(to_data(parent(data)) .== to_data(parent(rand_data)) .⊞ 0x1)
end

function test_copyto_multiple_F!(data)
    # Avoid using similar here due to https://github.com/CliMA/ClimaCore.jl/issues/1803
    rand_data = DataLayouts.rebuild(data, similar(parent(data)))
    Random.rand!(parent(rand_data))
    to_data(array) = DataLayouts.bitcast_struct.(eltype(data.:1), array)

    Base.copyto!(data, rand_data)
    @test all(to_data(parent(data.:1)) .== to_data(parent(rand_data.:1)))
    @test all(parent(data.:2) .== parent(rand_data.:2))
    # No need to convert the second component, since it has no internal padding

    Base.copyto!(data, Base.Broadcast.broadcasted(⊞, rand_data, 0x1))
    @test all(to_data(parent(data.:1)) .== to_data(parent(rand_data.:1)) .⊞ 0x1)
    # Do not test the second component, since it spans multiple array indices
end

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

@testset "copyto!" begin
    ArrayType = ClimaComms.array_type(ClimaComms.device()){Float64}
    @testset "Nf = 1 (uniform)" begin
        for data in all_layouts(ArrayType, Float64)
            test_copyto_single_F!(data)
            test_copyto_single_F!(data_view(data))
        end
    end
    @testset "Nf = 1 (nonuniform)" begin
        for data in all_layouts(ArrayType, Tuple{Int32, UInt8})
            test_copyto_single_F!(data)
            test_copyto_single_F!(data_view(data))
        end
    end
    @testset "Nf = 3 (uniform)" begin
        for data in all_layouts(ArrayType, Tuple{Float64, NTuple{2, Float64}})
            test_copyto_multiple_F!(data)
            test_copyto_multiple_F!(data_view(data))
        end
    end
    @testset "Nf = 3 (nonuniform)" begin
        for data in all_layouts(ArrayType, Tuple{Tuple{Int32, UInt8}, UInt128})
            test_copyto_multiple_F!(data)
            test_copyto_multiple_F!(data_view(data))
        end
    end
end
