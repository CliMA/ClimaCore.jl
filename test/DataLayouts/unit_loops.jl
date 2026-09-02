using Test
import Random
import ClimaComms
import ClimaCore.DataLayouts
import StaticArrays

import ClimaCore  # for `pkgdir` below
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU
ClimaComms.@import_required_backends
Random.seed!(1234)

device_array(device, array) = ClimaComms.array_type(device)(array)

# Use integer values so that sums are exact regardless of iteration order,
# which makes comparisons insensitive to how threads partition the data.
function test_data(
    device,
    ::Type{T},
    Nf,
    Nv,
    ::Type{FT} = T isa Type{<:Number} ? T : Float64,
) where {T, FT}
    (Ni, Nj, Nh) = (4, 4, 5)
    array = device_array(device, FT.(rand(1:1000, Nv, Ni, Nj, Nf, Nh)))
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

@testset "nested loop functions [$FT]" for FT in (Float32, Float64)
    device = ClimaComms.device()
    arg = test_data(device, FT, 1, 10)
    dest = test_data(device, FT, 1, 1)
    reference_dest = test_data(device, FT, 1, 1)

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
        TU.allocation_checks_meaningful() && @test sum_allocs == 0
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

@testset "0-dimensional data in broadcast expressions [$FT]" for FT in (Float32, Float64)
    device = ClimaComms.device()
    data = test_data(device, FT, 1, 10)
    point = DataLayouts.DataF{FT}(device_array(device, rand(FT, 1)))

    # Every linear or Cartesian index of a broadcast expression should access
    # the single point of any 0-dimensional data in that expression.
    @test parent(data .+ point) == parent(data) .+ Array(parent(point))[]
    @test parent(point .+ data) == parent(data) .+ Array(parent(point))[]
end

parent_broadcasted(data::DataLayouts.DataLayout) = parent(data)
parent_broadcasted(bc::DataLayouts.LazyDataLayout) =
    Base.broadcasted(bc.f, map(parent_broadcasted, bc.args)...)

# Loops over mixed layouts require Cartesian indices, which are extruded by
# Broadcast.newindex; loops over layouts with the same shape use linear indices.
# Point layouts are ignored when deciding between linear and Cartesian indexing.
@testset "broadcast expression indexing" begin
    device = ClimaComms.device()
    volume = test_data(device, Float64, 1, 10)
    surface = test_data(device, Float64, 1, 1)
    point = view(volume, 1)

    for (bc, expected_style) in (
        (Base.broadcasted(+, volume, volume, volume), IndexLinear()),
        (Base.broadcasted(+, volume, volume, surface), IndexCartesian()),
        (Base.broadcasted(+, volume, volume, point), IndexLinear()),
        (Base.broadcasted(+, volume, surface, point), IndexCartesian()),
        (Base.broadcasted(abs, Base.broadcasted(+, volume, volume)), IndexLinear()),
        (Base.broadcasted(abs, Base.broadcasted(+, volume, surface)), IndexCartesian()),
        (Base.broadcasted(abs, Base.broadcasted(+, volume, point)), IndexLinear()),
    )
        @test Base.IndexStyle(bc) == expected_style
        @test parent(Base.materialize(bc)) == Base.materialize(parent_broadcasted(bc))
        @test sum(bc) == sum(parent_broadcasted(bc))
    end

    # Genuinely mismatched extents throw before any kernel is launched.
    mismatched_volume = test_data(device, Float64, 1, 7)
    @test_throws DimensionMismatch volume .+ mismatched_volume
end

# Measure allocations from a top-level function, since the @allocated macro has
# a small constant overhead when it is used in a local scope.
assign_scalar!(data) = data .= eltype(data)(0.5)
assign_ref!(data) = data .= Ref(eltype(data)(0.5))
assign_tuple!(data) = data .= (eltype(data)(0.5),)
measured_allocations(f!::F, data) where {F} = @allocated f!(data)

@testset "scalar broadcast allocations [$FT]" for FT in (Float32, Float64)
    device = ClimaComms.device()
    data = test_data(device, FT, 1, 10)
    assign_scalar!(data)
    assign_ref!(data)
    assign_tuple!(data)
    @test all(==(FT(0.5)), Array(parent(data)))
    if device isa ClimaComms.CPUSingleThreaded
        @test measured_allocations(assign_scalar!, data) == 0
        @test measured_allocations(assign_ref!, data) == 0
        @test measured_allocations(assign_tuple!, data) == 0
    end
end

@testset "equality of layouts with different shapes [$FT]" for FT in (Float32, Float64)
    device = ClimaComms.device()
    data_a = test_data(device, FT, 1, 10)
    data_b = test_data(device, FT, 1, 11)

    # Comparing layouts with different sizes should return false instead of
    # throwing a DimensionMismatch from the elementwise fallback of ==.
    @test data_a != data_b
    @test data_a == copy(data_a)

    # NaNs make layouts unequal, matching the behavior of == on Base arrays.
    nan_data = test_data(device, FT, 1, 10)
    parent(nan_data) .= FT(NaN)
    @test nan_data != nan_data
end

@testset "views and equality of properties without data [$FT]" for FT in (Float32, Float64)
    device = ClimaComms.device()
    T = @NamedTuple{value::FT, unit::Nothing}
    data = test_data(device, T, 1, 10, FT)
    point = DataLayouts.DataF{T}(device_array(device, rand(FT, 1)))

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
        parent(modified_arg) .+= FT(1)
        @test arg != modified_arg
        @test arg.value != modified_arg.value
        @test arg.unit == arg.unit
        @test arg.unit == modified_arg.unit
    end
    @test data.unit != test_data(device, T, 1, 11, FT).unit
end

# Nv is deliberately not a multiple of a GPU warp. A block whose thread count exceeded the
# number of points in a column would leave its last threads with no points to reduce, and a
# reduction without an init value has no placeholder to use in their place.
@testset "reductions over slices whose length is not a multiple of a warp [$FT]" for FT in (
    Float32,
    Float64,
)
    device = ClimaComms.device()
    for Nv in (33, 63, 100)
        arg = test_data(device, FT, 1, Nv)
        dest = test_data(device, FT, 1, 1)
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
        DataLayouts.foreach_point(copy_point!, dest, data; mask = DataLayouts.NoMask())
        @test parent(dest) == parent(data)
        @test DataLayouts.reduce_points(+, data; mask = DataLayouts.NoMask()) == total
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

# A slice's point count must be a compile-time constant for foreach_slice's
# chosen scope to be one, even when the layout's element count is dynamic.
@testset "num_slice_points is a compile-time constant" begin
    # Wrapping the result in a Val makes inference's constant visible in a type.
    static_points(op, data) =
        Base.return_types(data -> Val(DataLayouts.num_slice_points(op, data)),
            Tuple{typeof(data)})[1]

    static = DataLayouts.VIJFH{Float64, 7, 4, 4, 5}(Array{Float64}, 5)
    dynamic = DataLayouts.VIJFH{Float64, 7, 4, 4, nothing}(Array{Float64}, 5)
    for data in (static, dynamic) # statically inferrable and dynamic
        @test DataLayouts.has_inferred_size(data) == (data === static)
        @test static_points(DataLayouts.slab, data) === Val{16}
        @test static_points(DataLayouts.column, data) === Val{7}
        @test static_points(view, data) === Val{1}
    end

    # Level slices: an uninferrable element count is reported as unbounded.
    @test static_points(DataLayouts.level, static) === Val{16 * 5}
    @test static_points(DataLayouts.level, dynamic) === Val{typemax(Int)}

    # Empty layouts report the slice size they would have; loops are no-ops.
    empty_data = DataLayouts.VIJFH{Float64, 7, 4, 4, nothing}(Array{Float64}, 0)
    @test static_points(DataLayouts.slab, empty_data) === Val{16}
    @test static_points(DataLayouts.column, empty_data) === Val{7}
    slices = 0
    DataLayouts.foreach_slab(_ -> (global slices += 1), empty_data)
    @test slices == 0

    # A broadcast's slices are constant whenever every argument's slices are.
    bc = Base.broadcasted(+, dynamic, dynamic)
    @test static_points(DataLayouts.slab, bc) === Val{16}
    @test static_points(DataLayouts.column, bc) === Val{7}
end

# Every range subindex of a StridedRange must give back another StridedRange:
# the generic AbstractArray fallback allocates a Vector and throws an
# interpolated size error, neither of which can be compiled for a GPU.
@testset "range subsets of a StridedRange" begin
    range = DataLayouts.StridedRange(3, 5, 4) # 3:5:18
    values = collect(range)
    @test values == [3, 8, 13, 18]

    for sub in (Base.OneTo(3), 2:4, 2:2:4, 4:-2:2, 3:2, DataLayouts.StridedRange(2, 2, 2))
        subset = range[sub]
        @test subset isa DataLayouts.StridedRange
        @test collect(subset) == values[sub]
    end

    # Views of views must reindex without allocating.
    array = reshape(collect(1:24), 4, 6)
    for outer in (DataLayouts.StridedRange(2, 4, 3), 2:4:14)
        outer_view = view(array, outer)
        for inner in (Base.OneTo(2), 2:3, DataLayouts.StridedRange(1, 2, 2))
            @test collect(view(outer_view, inner)) == collect(outer_view)[inner]
        end
    end
end

# Each thread of a RegisterArray must map its own points onto its own storage
# injectively, independently of its rank. FakeGroup stands in for the GPU
# sub-block scopes; it is defined outside the testsets, which wrap their bodies
# in functions.
struct FakeGroup{N} <: DataLayouts.DataScope end
DataLayouts.partition(::FakeGroup{N}) where {N} =
    N == 2 ? DataLayouts.ThisThread() : FakeGroup{N ÷ 2}()
DataLayouts.num_threads(::FakeGroup{N}) where {N} = N
DataLayouts.static_num_threads(::FakeGroup{N}) where {N} = N
# Rank of the thread currently "running", stepped through by one CPU thread.
const FAKE_RANK = Ref(1)
DataLayouts.thread_rank(::FakeGroup) = FAKE_RANK[]

@testset "RegisterArray stores one thread's points" begin
    for Nq in (2, 4, 5, 17), Nf in (1, 3), N in (2, 4, 16, 32, 256)
        array_size = (1, Nq, Nq, Nf, 1) # a slab of a VIJFH layout
        (; Np, SB) = DataLayouts.register_array_params(array_size, Val(4))
        Nl = cld(Np, N)
        @test Np == Nq^2 && SB == Nq^2 && Nl == cld(Nq^2, N)
        array = DataLayouts.RegisterArray{array_size, 4, N}(
            StaticArrays.MArray{Tuple{Nl * Nf}, Float64}(undef),
        )
        @test size(array) == array_size
        for rank in 1:min(N, Np) # every thread that is assigned points
            # Linear indices of this thread's components, in full-array order.
            indices = [
                f * Np + p for f in 0:(Nf - 1) for p in (rank - 1):N:(Np - 1)
            ]
            slots = map(i -> DataLayouts.register_index(array, i + 1), indices)
            @test allunique(slots)
            @test all(slot -> 1 <= slot <= Nl * Nf, slots)
        end
    end

    # Registers are only used for a scope with a statically known thread count
    # covering more than one thread; CPU loops keep the ordinary allocation.
    data = DataLayouts.VIJFH{Float64, 1, 4, 4, 1}(Array{Float64})
    slab = DataLayouts.slab(data, 1, 1)
    is_register(scope) =
        DataLayouts.parent_type(
            DataLayouts.register_similar(DataLayouts.reassign(slab, scope), Float64),
        ) <: DataLayouts.RegisterArray
    @test is_register(FakeGroup{16}())
    @test !is_register(DataLayouts.ThisThread())
    @test !is_register(DataLayouts.ThisThreadPool()) # no static_num_threads
end

# A launch unit holding only part of a subscope silently skips some points; see
# the whole subscopes invariant in DataLayouts.subscope_launch_threads.
# visited_points returns the points of an Np-point slice that a subscope of N
# threads reaches when only its first `present` threads were launched.
function visited_points(::FakeGroup{N}, present, Np) where {N}
    points = Int[]
    for rank in 1:present
        FAKE_RANK[] = rank
        indices = @inbounds DataLayouts.subscope_indices(
            DataLayouts.ThisThread(),
            FakeGroup{N}(),
            Base.OneTo(Np),
        )
        append!(points, indices)
    end
    FAKE_RANK[] = 1
    return sort!(points)
end

# Threads of the last subscope in a launch unit of the given size, which is a
# whole subscope only when the size is a multiple of the subscope's thread count.
present_threads(threads, N) = threads % N == 0 ? N : threads % N

@testset "a launch unit holds whole subscopes" begin
    # Every sub-block width that the GPU scope chain descends through, against
    # every block size that an occupancy search over whole warps could return.
    for N in (2, 4, 8, 16, 32, 64, 128, 256), max_threads in 32:32:256
        max_threads < N && continue
        threads = DataLayouts.subscope_launch_threads(FakeGroup{N}(), max_threads)
        @test threads % N == 0
        @test N <= threads <= max_threads

        # With whole subscopes, every point of a slice is visited exactly once.
        for Np in (max(N - 1, 1), N, N + 1, 3N + 1)
            @test visited_points(FakeGroup{N}(), present_threads(threads, N), Np) ==
                  collect(1:Np)
        end

        # Rounding the block size to whole warps instead is what loses points.
        if max_threads % N != 0
            @test visited_points(FakeGroup{N}(), present_threads(max_threads, N), N) !=
                  collect(1:N)
        end
    end
end
