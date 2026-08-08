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

# Point loops broadcast their arguments like ordinary broadcast expressions:
# 0-dimensional layouts and statically singleton dimensions (e.g. single-level
# surface data) are expanded to the combined loop bounds, while genuinely
# mismatched extents are rejected on the host, since the error path can be
# neither compiled nor cleanly reported in GPU kernels.
@testset "point loops over arguments with singleton dimensions" begin
    device = ClimaComms.device()
    volume = test_data(device, Float64, 1, 10)
    surface = test_data(device, Float64, 1, 1) # statically singleton Nv
    point = DataLayouts.DataF{Float64}(device_array(device, rand(1)))
    dest = test_data(device, Float64, 1, 10)

    # Singleton dimensions inside broadcast expressions expand like Base's.
    dest .= volume .+ surface
    @test parent(dest) == parent(volume) .+ parent(surface)

    # Singleton and 0-dimensional layouts may also be top-level loop arguments.
    fill!(parent(dest), 0)
    DataLayouts.foreach_point(
        (d, a, s, p) -> (@inbounds d[] = a[] + s[] + p[]),
        dest,
        volume,
        surface,
        point,
    )
    @test parent(dest) == parent(volume) .+ parent(surface) .+ parent(point)

    # Reductions over expressions with mixed shapes require Cartesian indices,
    # which Broadcast.newindex projects onto singleton dimensions; expressions
    # whose layouts all share a shape permit linear indices.
    mixed_bc = Base.broadcasted(+, volume, surface)
    @test Base.IndexStyle(mixed_bc) == IndexCartesian()
    @test parent(Base.materialize(mixed_bc)) == parent(volume) .+ parent(surface)
    # The sum is checked as well, since reductions iterate lazy expressions
    # directly, without materializing them first.
    @test sum(identity, mixed_bc) == sum(parent(volume) .+ parent(surface))
    @test Base.IndexStyle(Base.broadcasted(+, volume, volume)) == IndexLinear()

    # Genuinely mismatched extents throw before any kernel is launched.
    mismatched = test_data(device, Float64, 1, 7)
    @test_throws DimensionMismatch dest .= volume .+ mismatched
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

# Nv is deliberately not a multiple of a GPU warp. A block whose thread count exceeded the
# number of points in a column would leave its last threads with no points to reduce, and a
# reduction without an init value has no placeholder to use in their place.
@testset "reductions over slices whose length is not a multiple of a warp" begin
    device = ClimaComms.device()
    for Nv in (33, 63, 100)
        arg = test_data(device, Float64, 1, Nv)
        dest = test_data(device, Float64, 1, 1)
        reference_array = Array(parent(arg))
        # A column reduction assigns one column to each block, so the block's threads
        # divide up the points of a column.
        sum_of_columns!(dest, arg)
        @test Array(parent(dest)) == sum(reference_array; dims = 1)
        @test sum(identity, arg) == sum(reference_array)
        # min has no init value, so every thread of the reduction must have a point.
        @test DataLayouts.reduce_points(min, arg) == minimum(reference_array)
    end
end

@testset "thread pool resolution and sharing" begin
    device = ClimaComms.device()
    if device isa ClimaComms.AbstractCPUDevice
        data = test_data(device, Float64, 1, 64)
        total = sum(Array(parent(data)))
        pool_accounting_drained() =
            DataLayouts.POOL_THREADS_IN_USE[] == 0 &&
            DataLayouts.PENDING_POOL_LOOPS[] == 0

        # Loops nested in an external threaded loop use one thread each, but they
        # must still cover every point of their arguments.
        external_loop_sums = zeros(Threads.nthreads())
        Threads.@threads for i in eachindex(external_loop_sums)
            external_loop_sums[i] = sum(identity, data)
        end
        @test all(==(total), external_loop_sums)
        @test pool_accounting_drained()

        # Concurrent loops that divide the pool between them must each compute a
        # complete result, and small reductions must not disturb the division.
        point = DataLayouts.DataF{Float64}(device_array(device, rand(1)))
        concurrent_results = map(Base.OneTo(4)) do _
            Threads.@spawn begin
                is_correct = true
                for _ in Base.OneTo(50)
                    is_correct &= sum(identity, data) == total
                    is_correct &= parent(data .+ data) == 2 .* Array(parent(data))
                    is_correct &= sum(identity, point) == Array(parent(point))[]
                end
                is_correct
            end
        end
        @test all(fetch, concurrent_results)
        @test pool_accounting_drained()

        # Loops launched from tasks outside the default pool must also be complete.
        interactive_task = Threads.@spawn :interactive sum(identity, data)
        @test fetch(interactive_task) == total
        @test pool_accounting_drained()

        # Explicit-scope loops must cover every point without scope resolution. Loops given
        # a scope take a mask explicitly, since only the scope-free methods default it.
        dest = test_data(device, Float64, 1, 64)
        fill!(parent(dest), 0)
        copy_point!(d, a) = (@inbounds d[] = a[])
        DataLayouts.foreach_slice(
            DataLayouts.ThisThreadPool(),
            view,
            copy_point!,
            dest,
            data;
            mask = DataLayouts.NoMask(),
        )
        @test parent(dest) == parent(data)
        @test DataLayouts.reduce_points(
            DataLayouts.ThisThreadPool(),
            +,
            data;
            mask = DataLayouts.NoMask(),
        ) == total
        @test pool_accounting_drained()

        # An error thrown from inside a loop must not leak the loop's thread claim.
        @test_throws Exception DataLayouts.foreach_point(_ -> error("!"), data)
        @test pool_accounting_drained()

        # A reduction nested in another reduction's op must not overwrite the outer
        # reduction's results, which live in the task's reduction buffer. Over all-ones
        # data, no worker's fold accumulator exceeds its chunk length, so a threshold
        # just above the largest chunk only triggers in the launcher's final combine.
        if Threads.threadpoolsize(:default) > 1
            ones_data = DataLayouts.VIJFH{Float64, 64, 4, 4, nothing}(
                device_array(device, ones(64, 4, 4, 1, 5)),
            )
            n_points = length(parent(ones_data))
            threshold = cld(n_points, Threads.threadpoolsize(:default)) + 1
            other = test_data(device, Float64, 1, 64)
            nested_op(a, b) =
                a + b + 0.0 * (a >= threshold ? sum(identity, other) : 0.0)
            @test DataLayouts.reduce_points(nested_op, ones_data) == n_points
            @test pool_accounting_drained()
        end

        # Loops over the pool cannot be nested inside its own worker threads.
        if Threads.threadpoolsize(:default) > 1
            @test_throws CompositeException DataLayouts.foreach_point(
                _ -> sum(identity, data),
                data,
            )
            @test pool_accounting_drained()
        end
    end
end
