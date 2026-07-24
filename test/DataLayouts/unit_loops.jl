#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_loops.jl"))
=#
using Test
import Random
import ClimaComms
import ClimaCore.DataLayouts
ClimaComms.@import_required_backends
Random.seed!(1234)

device_array(device, array) = ClimaComms.array_type(device)(array)

# Use integer values so that sums are exact regardless of iteration order,
# which makes comparisons insensitive to how threads partition the data.
function test_data(device, ::Type{T}, Nf, Nv) where {T}
    (Ni, Nj, Nh) = (4, 4, 5)
    array = device_array(device, Float64.(rand(1:(2^20), Nv, Ni, Nj, Nf, Nh)))
    return DataLayouts.VIJFH{T, Nv, Ni, Nj, nothing}(array)
end

sum_of_columns!(dest, arg) = DataLayouts.column_reduce!(+, dest, arg)
function manual_sum_of_columns!(dest, arg)
    for h in 1:size(arg, 4), j in 1:size(arg, 3), i in 1:size(arg, 2)
        fill!(
            DataLayouts.column(dest, i, j, h),
            sum(DataLayouts.column(arg, i, j, h)),
        )
    end
    return dest
end

@testset "nested loop functions" begin
    device = ClimaComms.device()
    arg = test_data(device, Float64, 1, 10)
    dest = test_data(device, Float64, 1, 1)
    reference_dest = test_data(device, Float64, 1, 1)

    # Nest fill!, sum, and mapreduce within the function passed to a slice
    # iterator, as in DataLayouts.column_reduce!.
    sum_of_columns!(dest, arg)
    manual_sum_of_columns!(reference_dest, arg)
    @test dest == reference_dest

    DataLayouts.foreach_column((dc, ac) -> fill!(dc, mapreduce(abs, +, ac)), dest, arg)
    manual_dest_array = sum(abs, Array(parent(arg)); dims = 1)
    @test Array(parent(dest)) == manual_dest_array

    if device isa ClimaComms.CPUSingleThreaded
        # Nested loop functions rely on the recursion_relation overrides at the
        # end of the DataLayouts module; without them, the inner loops box
        # their arguments and allocate at every column.
        sum_allocs = @allocated sum_of_columns!(dest, arg)
        @test sum_allocs == 0
    end
end

# Pairwise sum with the same structure as safe_mapreduce (sequential blocks of
# up to 1024 values, midpoint splits), but with no @simd reassociation.
function strict_pairwise_sum(values, ifirst, ilast)
    if ilast - ifirst >= 1024
        imid = (ifirst + ilast) >> 1
        return strict_pairwise_sum(values, ifirst, imid) +
               strict_pairwise_sum(values, imid + 1, ilast)
    end
    value = values[ifirst]
    for index in (ifirst + 1):ilast
        value += values[index]
    end
    return value
end

@testset "reduction accuracy and masks" begin
    device = ClimaComms.device()

    # Layouts with several components use Cartesian indexing, whose indices are
    # stored in a CartesianIndices object. Every single-threaded reduction
    # should be identical to a strict pairwise reduction over positions (Base's
    # mapreduce falls back to a sequential fold for CartesianIndices, whose
    # roundoff error grows linearly with the number of points, while the
    # roundoff error of a pairwise reduction only grows logarithmically). The
    # reference is written out by hand because the @simd blocks in Base's
    # mapreduce and in safe_mapreduce reassociate values at contiguous
    # positions whenever bounds checking is disabled. Multithreaded and GPU
    # reductions partition the points across threads, so they are only
    # approximately pairwise.
    (Ni, Nj, Nh) = (4, 4, 5)
    array = Float32.(rand(64, Ni, Nj, 2, Nh)) ./ 3
    T = Tuple{Float32, Float32}
    data = DataLayouts.VIJFH{T, 64, Ni, Nj, nothing}(device_array(device, array))
    first_values = vec(Array(parent(data))[:, :, :, 1, :])
    pairwise_sum = strict_pairwise_sum(first_values, 1, length(first_values))
    if device isa ClimaComms.CPUSingleThreaded
        @test sum(value -> value[1], data) == pairwise_sum
    else
        @test sum(value -> value[1], data) ≈ pairwise_sum
    end
    @test sum(value -> value[1], data) ≈ sum(first_values)

    data = test_data(device, Float64, 1, 4)
    mask = DataLayouts.IJHMask(data)
    @test DataLayouts.reduce_points(+, data; mask, init = 0.0) == sum(Array(parent(data)))
end

@testset "0-dimensional data in broadcast expressions" begin
    device = ClimaComms.device()
    data = test_data(device, Float64, 1, 10)
    point = DataLayouts.DataF{Float64}(device_array(device, rand(1)))

    # Every linear or Cartesian index of a broadcast expression should access
    # the single point of any 0-dimensional data in that expression.
    @test parent(data .+ point) == parent(data) .+ Array(parent(point))[]
    @test parent(point .+ data) == parent(data) .+ Array(parent(point))[]
end

# Measure allocations from a top-level function, since the @allocated macro has
# a small constant overhead when it is used in a local scope.
assign_scalar!(data) = data .= 0.5
assign_ref!(data) = data .= Ref(0.5)
assign_tuple!(data) = data .= (0.5,)
measured_allocations(f!::F, data) where {F} = @allocated f!(data)

@testset "scalar broadcast allocations" begin
    device = ClimaComms.device()
    data = test_data(device, Float64, 1, 10)
    assign_scalar!(data)
    assign_ref!(data)
    assign_tuple!(data)
    @test all(==(0.5), Array(parent(data)))
    if device isa ClimaComms.CPUSingleThreaded
        @test measured_allocations(assign_scalar!, data) == 0
        @test measured_allocations(assign_ref!, data) == 0
        @test measured_allocations(assign_tuple!, data) == 0
    end
end

@testset "equality of layouts with different shapes" begin
    device = ClimaComms.device()
    data_a = test_data(device, Float64, 1, 10)
    data_b = test_data(device, Float64, 1, 11)

    # Comparing layouts with different sizes should return false instead of
    # throwing a DimensionMismatch from the elementwise fallback of ==.
    @test data_a != data_b
    @test data_a == copy(data_a)

    # NaNs make layouts unequal, matching the behavior of == on Base arrays.
    nan_data = test_data(device, Float64, 1, 10)
    parent(nan_data) .= NaN
    @test nan_data != nan_data
end

@testset "views and equality of properties without data" begin
    device = ClimaComms.device()
    T = @NamedTuple{value::Float64, unit::Nothing}
    data = test_data(device, T, 1, 10)
    point = DataLayouts.DataF{T}(device_array(device, rand(1)))

    # Zero-size fields are hidden from propertynames, but they are still
    # accessible through getproperty, which returns a view with Nf = 0.
    for arg in (data, point)
        @test propertynames(arg) == (:value,)
        @test eltype(arg.unit) == Nothing
        @test DataLayouts.ncomponents(arg.unit) == 0
        @test size(arg.unit) == size(arg)

        # Views with no data are equal whenever their sizes match, even when
        # the layouts they were created from are unequal.
        modified_arg = copy(arg)
        parent(modified_arg) .+= 1
        @test arg != modified_arg
        @test arg.value != modified_arg.value
        @test arg.unit == arg.unit
        @test arg.unit == modified_arg.unit
    end
    @test data.unit != test_data(device, T, 1, 11).unit
end
