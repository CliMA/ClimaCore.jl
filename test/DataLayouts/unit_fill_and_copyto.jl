using Test
import Random
import ClimaComms
import ClimaCore: DataLayouts, Geometry
import ClimaCore.RecursiveApply: ⊞
ClimaComms.@import_required_backends
Random.seed!(1234)

# Loop over different layout shapes and different types of parent arrays.
function testable_layouts(A, T)
    (Nv, Nij, Nh) = (3, 4, 5)
    layouts = (
        DataLayouts.DataF{T}(A),
        DataLayouts.VIJFH{T, Nv, Nij, Nij, 1}(A),
        DataLayouts.VIJFH{T, Nv, Nij, Nij, missing}(A, Nh),
        DataLayouts.VIJHF{T, Nv, Nij, Nij, 1}(A),
        DataLayouts.VIJHF{T, Nv, Nij, Nij, missing}(A, Nh),
    )
    if sizeof(T) == sizeof(eltype(A))
        layouts = (
            layouts...,
            DataLayouts.VIH1{T, Nv, Nij, 1}(A),
            DataLayouts.VIH1{T, Nv, Nij, missing}(A, Nh),
            DataLayouts.IH1JH2{T, Nij, Nij, 1}(A),
            DataLayouts.IH1JH2{T, Nij, Nij, missing}(A, Nh),
        )
    end
    return Iterators.flatmap(layouts) do data
        subarray_parent = view(parent(data), axes(parent(data))...)
        reshaped_array_parent = reshape(subarray_parent, size(parent(data))...)
        subarray_data = DataLayouts.rebuild(data, subarray_parent)
        reshaped_array_data = DataLayouts.rebuild(data, reshaped_array_parent)
        (data, subarray_data, reshaped_array_data)
    end
end

function test_single_F!(data)
    rand_data = similar(data)
    Random.rand!(parent(rand_data))
    to_data(array) = DataLayouts.bitcast_struct.(eltype(data), array)

    Base.fill!(data, first(rand_data))
    @test all(to_data(parent(data)) .== to_data(parent(view(rand_data, 1))))

    Base.copyto!(data, rand_data)
    @test all(to_data(parent(data)) .== to_data(parent(rand_data)))

    Base.copyto!(data, Base.Broadcast.broadcasted(+, rand_data, 0x1))
    @test all(to_data(parent(data)) .== to_data(parent(rand_data)) .⊞ 0x1)
end

function test_multiple_F!(data)
    rand_data = similar(data)
    Random.rand!(parent(rand_data))
    to_data(array) = DataLayouts.bitcast_struct.(eltype(data.:1), array)

    Base.fill!(data, first(rand_data))
    @test all(to_data(parent(data.:1)) .== to_data(parent(view(rand_data.:1, 1))))
    @test all(parent(data.:2) .== parent(view(rand_data.:2, 1)))
    # We do not need to convert the second component, since it has no padding.

    Base.copyto!(data, rand_data)
    @test all(to_data(parent(data.:1)) .== to_data(parent(rand_data.:1)))
    @test all(parent(data.:2) .== parent(rand_data.:2))
    # As in the previous test, we do not need to convert the second component.

    Base.copyto!(data, Base.Broadcast.broadcasted(+, rand_data, 0x1))
    @test all(to_data(parent(data.:1)) .== to_data(parent(rand_data.:1)) .⊞ 0x1)
    # Do not test the second component, since it spans multiple array indices.
end

@testset "fill! and copyto!" begin
    device = ClimaComms.device()
    A = ClimaComms.array_type(device){Float64}
    @testset "Nf = 1 (uniform)" begin
        for data in testable_layouts(A, Float64)
            test_single_F!(data)
        end
    end
    @testset "Nf = 1 (nonuniform)" begin
        for data in testable_layouts(A, Tuple{Int32, UInt8})
            test_single_F!(data)
        end
    end
    @testset "Nf = 3 (uniform)" begin
        for data in testable_layouts(A, Tuple{Float64, NTuple{2, Float64}})
            test_multiple_F!(data)
        end
    end
    @testset "Nf = 3 (nonuniform)" begin
        for data in testable_layouts(A, Tuple{Tuple{Int32, UInt8}, UInt128})
            test_multiple_F!(data)
        end
    end
end
