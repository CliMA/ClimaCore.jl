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
        DataLayouts.VIJFH{T, Nv, Nij, Nij, nothing}(A, Nh),
        DataLayouts.VIJHF{T, Nv, Nij, Nij, 1}(A),
        DataLayouts.VIJHF{T, Nv, Nij, Nij, nothing}(A, Nh),
    )
    if sizeof(T) == sizeof(eltype(A))
        layouts = (
            layouts...,
            DataLayouts.VIH1{T, Nv, Nij, 1}(A),
            DataLayouts.VIH1{T, Nv, Nij, nothing}(A, Nh),
            DataLayouts.IH1JH2{T, Nij, Nij, 1}(A),
            DataLayouts.IH1JH2{T, Nij, Nij, nothing}(A, Nh),
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

# Compare a filled layout against the value at the first point of another
# layout, using copies on the CPU (reading device data directly would require
# scalar indexing of GPU arrays) and single-point views at every index (the
# parent of a single-point view is a vector of the point's array entries, which
# cannot be broadcast against the multidimensional parent of the full layout).
function test_filled_with_first_point(to_data, data, rand_data)
    cpu_data = DataLayouts.rebuild(data, Array)
    cpu_first_point = to_data(parent(view(DataLayouts.rebuild(rand_data, Array), 1)))
    return all(eachindex(cpu_data)) do index
        to_data(parent(view(cpu_data, index))) == cpu_first_point
    end
end

function test_single_F!(data)
    rand_data = similar(data)
    Random.rand!(parent(rand_data))
    to_data(array) = DataLayouts.bitcast_struct.(eltype(data), array)

    Base.fill!(data, first(DataLayouts.rebuild(rand_data, Array)))
    @test test_filled_with_first_point(to_data, data, rand_data)

    Base.copyto!(data, rand_data)
    @test all(to_data(parent(data)) .== to_data(parent(rand_data)))

    Base.copyto!(data, Base.Broadcast.broadcasted(+, rand_data, 0x1))
    @test all(to_data(parent(data)) .== to_data(parent(rand_data)) .⊞ 0x1)
end

function test_multiple_F!(data)
    rand_data = similar(data)
    Random.rand!(parent(rand_data))
    to_data(array) = DataLayouts.bitcast_struct.(eltype(data.:1), array)

    Base.fill!(data, first(DataLayouts.rebuild(rand_data, Array)))
    @test test_filled_with_first_point(to_data, data.:1, rand_data.:1)
    @test test_filled_with_first_point(identity, data.:2, rand_data.:2)
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
    @testset "scalar broadcasts of impure functions" begin
        # Functions like rand must be evaluated at every point, so only flat
        # identity broadcasts can be replaced with a single call to fill!.
        for data in testable_layouts(A, Float64)
            length(data) > 1 || continue
            data .= rand.()
            @test length(unique(Array(parent(data)))) > 1
        end
    end
end
