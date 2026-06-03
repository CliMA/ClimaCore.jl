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
Random.seed!(1234)

device = ClimaComms.device()
context = ClimaComms.context(device)
ClimaComms.init(context)

function test_mapreduce_1!(data)
    parent(data) .= rand.(eltype(parent(data)))
    @test minimum(data) == minimum(parent(data))
    @test maximum(sqrt, data) == maximum(sqrt, parent(data))
end

function test_mapreduce_2!(data)
    parent(data) .= rand.(eltype(parent(data)))
    @test minimum(data) == (minimum(parent(data.:1)), minimum(parent(data.:2)))
    @test maximum(sqrt, data) ==
          (maximum(sqrt, parent(data.:1)), maximum(sqrt, parent(data.:2)))
end

@testset "mapreduce with Nf = 1" begin
    FT = Float64
    A = ClimaComms.array_type(device){FT}
    (Nv, Nij, Nh) = (3, 4, 5)
    for data in (
        DataLayouts.DataF{FT}(A),
        DataLayouts.VIJFH{FT, Nv, Nij, Nij, missing}(A, Nh),
        DataLayouts.VIJHF{FT, Nv, Nij, Nij, missing}(A, Nh),
        DataLayouts.VIH1{FT, Nv, Nij, missing}(A, Nh),
        DataLayouts.IH1JH2{FT, Nij, Nij, missing}(A, Nh),
    )
        test_mapreduce_1!(data)
        subarray_parent = view(parent(data), axes(parent(data))...)
        test_mapreduce_1!(DataLayouts.rebuild(data, subarray_parent))
    end
end

@testset "mapreduce with Nf > 1" begin
    FT = Float64
    A = ClimaComms.array_type(device){FT}
    (Nv, Nij, Nh) = (3, 4, 5)
    for data in (
        DataLayouts.DataF{Tuple{FT, FT}}(A),
        DataLayouts.VIJFH{Tuple{FT, FT}, Nv, Nij, Nij, missing}(A, Nh),
        DataLayouts.VIJHF{Tuple{FT, FT}, Nv, Nij, Nij, missing}(A, Nh),
    )
        test_mapreduce_2!(data)
        subarray_parent = view(parent(data), axes(parent(data))...)
        test_mapreduce_2!(DataLayouts.rebuild(data, subarray_parent))
    end
end

# @testset "mapreduce with space with some non-round blocks" begin
#     # https://github.com/CliMA/ClimaCore.jl/issues/2097
#     space = ClimaCore.CommonSpaces.RectangleXYSpace(;
#         x_min = 0,
#         x_max = 1,
#         y_min = 0,
#         y_max = 1,
#         periodic_x = false,
#         periodic_y = false,
#         n_quad_points = 4,
#         x_elem = 129,
#         y_elem = 129,
#     )
#     @test minimum(ones(space)) == 1

#     if ClimaComms.context isa ClimaComms.SingletonCommsContext
#         # Less than 256 threads
#         space = ClimaCore.CommonSpaces.RectangleXYSpace(;
#             x_min = 0,
#             x_max = 1,
#             y_min = 0,
#             y_max = 1,
#             periodic_x = false,
#             periodic_y = false,
#             n_quad_points = 2,
#             x_elem = 2,
#             y_elem = 2,
#         )
#         @test minimum(ones(space)) == 1
#     end
# end
