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

column_accumulate!(
    op::O,
    dest,
    arg;
    reverse = false,
    init = DataLayouts.NoInit(),
) where {O} =
    DataLayouts.foreach_column(dest, arg) do dest_column, arg_column
        DataLayouts.scan_points(op, arg_column, reverse, init) do index, value
            @inbounds dest_column[index] = value
        end
    end

column_reduce!(op::O, dest, arg; reverse = false, init = DataLayouts.NoInit()) where {O} =
    DataLayouts.foreach_column(dest, arg) do dest_column, arg_column
        fill!(dest_column, DataLayouts.fold_points(op, arg_column, reverse, init))
    end

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

sum_of_columns!(dest, arg) = column_reduce!(+, dest, arg)
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
    # iterator, as in column_reduce!.
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
# up to 1024 values, midpoint splits), but with one accumulator per block.
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

# The level is checked at every entry point, on every device, so that a misspelled level
# does not pass on CPUs and fail on the first GPU run.
@testset "shared memory levels are checked on every device" begin
    device = ClimaComms.device()
    data = test_data(device, Float64, 1, 4)
    body!(c) = (fill!(c, 1); nothing)
    for level in (:auto, :none, :medium, :large, :max, 0, 100)
        @test isnothing(DataLayouts.foreach_column(body!, data; shmem = Val(level)))
        @test isnothing(DataLayouts.foreach_point(body!, data; shmem = Val(level)))
    end
    for level in (Val(:high), Val(:Auto), Val(-1), Val(1.5), :max, 100, nothing)
        @test_throws ArgumentError DataLayouts.foreach_column(body!, data; shmem = level)
        @test_throws ArgumentError DataLayouts.foreach_slice(
            DataLayouts.DataScope(data), DataLayouts.column, body!, data; shmem = level,
        )
    end
    # The register cap is checked the same way.
    for cap in (:auto, :none, 32, 255)
        @test isnothing(DataLayouts.foreach_column(body!, data; maxregs = Val(cap)))
        @test isnothing(DataLayouts.foreach_point(body!, data; maxregs = Val(cap)))
    end
    for cap in (Val(:device), Val(:Auto), Val(0), Val(-8), Val(1.5), :auto, 64, nothing)
        @test_throws ArgumentError DataLayouts.foreach_column(body!, data; maxregs = cap)
        @test_throws ArgumentError DataLayouts.foreach_slice(
            DataLayouts.DataScope(data), DataLayouts.column, body!, data; maxregs = cap,
        )
    end
end

@testset "reduction accuracy and masks" begin
    device = ClimaComms.device()

    # Layouts with several components use Cartesian indexing, whose indices are
    # stored in a CartesianIndices object. Single-threaded reductions split the
    # points like a strict pairwise reduction over positions (Base's mapreduce
    # falls back to a sequential fold for CartesianIndices, whose roundoff error
    # grows linearly with the number of points, while the roundoff error of a
    # pairwise reduction only grows logarithmically), but safe_mapreduce interleaves
    # several accumulators in each block, so it only matches the reference to a few
    # ulps. Multithreaded and GPU reductions partition the
    # points across threads, and they require a looser error bound.
    (Ni, Nj, Nh) = (4, 4, 5)
    array = Float32.(rand(64, Ni, Nj, 2, Nh)) ./ 3
    T = Tuple{Float32, Float32}
    data = DataLayouts.VIJFH{T, 64, Ni, Nj, nothing}(device_array(device, array))
    first_values = vec(Array(parent(data))[:, :, :, 1, :])
    pairwise_sum = strict_pairwise_sum(first_values, 1, length(first_values))
    if device isa ClimaComms.CPUSingleThreaded
        @test sum(value -> value[1], data) ≈ pairwise_sum rtol = 4 * eps(Float32)
    else
        @test sum(value -> value[1], data) ≈ pairwise_sum
    end
    @test sum(value -> value[1], data) ≈ sum(first_values)

    data = test_data(device, Float64, 1, 4)
    mask = DataLayouts.IJHMask(data)
    @test DataLayouts.reduce_points(+, data; mask, init = 0.0) == sum(Array(parent(data)))

    # An init value of a wider type than the points widens the fold on every device (the
    # warp results of a GPU block are stored with the widened type, which sizes the block's
    # shared memory).
    data32 = test_data(device, Float32, 1, 64)
    values32 = Array(parent(data32))
    @test DataLayouts.reduce_points(+, data32; init = 0.0) === sum(Float64, values32)
    @test DataLayouts.reduce_points(max, data32; init = -Inf) === maximum(Float64, values32)

    # A container init value is combined with the containers in the data, and the result has
    # a concrete type (on GPUs, its size is the size of the block's warp results). Like the
    # init of Base.reduce, it must be a neutral element, since every thread may fold it.
    data = DataLayouts.VIJFH{T, 64, Ni, Nj, nothing}(device_array(device, array))
    result = @inferred reduce(+, data; init = (0.0f0, 0.0f0))
    @test collect(result) ≈ [sum(array[:, :, :, f, :]) for f in 1:2]
end

# Commutative folds and reductions apply op to NamedTuples and Tuples (read as
# AutoBroadcasters) like they apply it to each of their components, but CPU column sums of
# floats use @simd loops, which LLVM can reassociate, while CPU column sums of containers
# interleave accumulators, so their results only match up to roundoff. For 63 values in
# [0, 1), any two accumulator counts from 1 to 32 give relative differences of at most
# about 5 eps. CPU sums over whole layouts add the values of containers and of their
# components in the same order, so their results match exactly.
@testset "reductions of containers match reductions of their components" begin
    device = ClimaComms.device()
    (Nv, Ni, Nj, Nh) = (63, 4, 4, 5)
    # Containers with more values than the accumulators of a sequential reduction, and
    # nested containers, are also reduced like each of their components.
    for T in (
        Tuple{Float64, Float64},
        @NamedTuple{a::Float64, b::Float64},
        NTuple{12, Float64},
        @NamedTuple{a::Float64, b::Tuple{Float64, Float64}, c::@NamedTuple{d::Float64}},
    )
        Nf = sizeof(T) ÷ sizeof(Float64)
        array = rand(Nv, Ni, Nj, Nf, Nh)
        components = map(1:Nf) do f
            DataLayouts.VIJFH{Float64, Nv, Ni, Nj, nothing}(
                device_array(device, array[:, :, :, f:f, :]),
            )
        end
        component_dest = test_data(device, Float64, 1, 1)
        data = DataLayouts.VIJFH{T, Nv, Ni, Nj, nothing}(device_array(device, array))
        dest = Base.Broadcast.broadcastable(test_data(device, T, Nf, 1))
        for reverse in (false, true), init in (DataLayouts.NoInit(), 0.5)
            container_init =
                init isa DataLayouts.NoInit ? init :
                DataLayouts.bitcast_struct(T, ntuple(_ -> init, Nf))
            column_reduce!(
                +,
                dest,
                Base.Broadcast.broadcastable(data);
                reverse,
                init = container_init,
            )
            for f in 1:Nf
                column_reduce!(+, component_dest, components[f]; reverse, init)
                container_values = Array(parent(dest))[:, :, :, f:f, :]
                component_values = Array(parent(component_dest))
                if device isa ClimaComms.AbstractCPUDevice
                    rtol = 16 * eps(Float64)
                    @test all(isapprox.(container_values, component_values; rtol))
                else
                    @test container_values == component_values
                end
            end
        end
        # Partitions of whole-layout GPU reductions depend on the registers of each kernel.
        device isa ClimaComms.AbstractCPUDevice || continue
        values(x) = DataLayouts.bitcast_struct(NTuple{Nf, Float64}, x)
        @test collect(values(reduce(+, data))) ==
              map(component -> reduce(+, component), components)
        container_init = DataLayouts.bitcast_struct(T, ntuple(_ -> 0.5, Nf))
        @test collect(values(reduce(+, data; init = container_init))) ==
              map(component -> reduce(+, component; init = 0.5), components)
    end
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

# A multi-element tuple is indexed along the first (level) dimension, as in Base,
# and an IJHMask leaves inactive columns untouched, as for any other broadcast.
@testset "multi-element tuple broadcasts with a mask" begin
    device = ClimaComms.device()
    data = test_data(device, Float64, 1, 3)
    fill!(parent(data), 0)
    mask = DataLayouts.IJHMask(data)
    is_active_copy = Array(parent(DataLayouts.is_active(mask)))
    is_active_copy[1, 1, 1, 1, 1] = false
    copyto!(parent(DataLayouts.is_active(mask)), is_active_copy)
    DataLayouts.set_mask_maps!(mask)
    bc = Base.broadcasted(identity, (1.0, 2.0, 3.0))
    copyto!(data, bc, mask)
    parent_copy = Array(parent(data))
    @test parent_copy[:, 1, 1, 1, 1] == zeros(3)
    @test parent_copy[:, 2, 1, 1, 1] == [1.0, 2.0, 3.0]
    @test parent_copy[:, 4, 4, 1, 5] == [1.0, 2.0, 3.0]
    ArrayType = ClimaComms.array_type(device){Float64}
    for data in (data, DataLayouts.VIJHF{Float64, 3, 2, 2, 1}(ArrayType))
        data .= (1.0, 2.0, 3.0)
        parent_copy = Array(parent(data))
        @test all(v -> all(==(v), selectdim(parent_copy, 1, v)), 1:3)
    end
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

# Broadcasts whose source corresponds to a slice of the destination must use the
# destination axes when materializing. Broadcasts whose destination corresponds
# to a slice of the source should throw a DimensionMismatch error.
@testset "extruding broadcast axes" begin
    device = ClimaComms.device()
    dest = test_data(device, Float64, 1, 10)
    src = test_data(device, Float64, 1, 10)
    for src_slice in (
        view(src, 1),
        DataLayouts.level(src, 1),
        DataLayouts.slab(src, 1, 1),
        DataLayouts.column(src, 1, 1, 1),
    )
        dest .= src_slice
        @test all(parent(dest) .== parent(src_slice))
        @test_throws DimensionMismatch src_slice .= dest
    end
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


# Column accumulations over a warp (Nv <= 32), several warps (Nv <= 128), and several points
# per thread (Nv > 128) on GPUs, with operators that the threads of a column can reorder and
# operators that one thread must apply in order.
@testset "column accumulations and reductions in order [$FT]" for FT in (Float32, Float64)
    device = ClimaComms.device()
    for Nv in (20, 63, 200)
        arg = test_data(device, FT, 1, Nv)
        dest = test_data(device, FT, 1, Nv)
        dest_level = test_data(device, FT, 1, 1)
        column_array = Array(parent(arg))
        reversed_array = reverse(column_array; dims = 1)
        column_accumulate!(+, dest, arg)
        @test Array(parent(dest)) == cumsum(column_array; dims = 1)
        column_accumulate!(max, dest, arg; reverse = true, init = FT(500))
        @test Array(parent(dest)) ==
              reverse(accumulate(max, reversed_array; dims = 1, init = FT(500)); dims = 1)
        column_accumulate!((a, b) -> a - b, dest, arg; init = FT(1))
        @test Array(parent(dest)) == accumulate(-, column_array; dims = 1, init = FT(1))
        column_reduce!((a, b) -> a - b, dest_level, arg; reverse = true)
        @test Array(parent(dest_level)) ==
              mapslices(c -> foldl(-, c), reversed_array; dims = 1)
        # In place, one thread reads every point of the column before overwriting it.
        copyto!(dest, arg)
        column_accumulate!((a, b) -> a - b, dest, dest; init = FT(1))
        @test Array(parent(dest)) == accumulate(-, column_array; dims = 1, init = FT(1))
        copyto!(dest, arg)
        DataLayouts.foreach_column(dest) do dest_column
            fill!(dest_column, DataLayouts.fold_points((a, b) -> a - b, dest_column, true))
        end
        @test Array(parent(dest)) ==
              repeat(mapslices(c -> foldl(-, c), reversed_array; dims = 1), Nv)

        # Products of matrices are associative but do not commute, so the threads of a
        # column must combine them in order.
        M = StaticArrays.SMatrix{2, 2, FT, 4}
        matrices = DataLayouts.VIJFH{M, Nv, 4, 4, nothing}(
            device_array(device, 1 .+ (rand(FT, Nv, 4, 4, 4, 5) .- FT(0.5)) ./ 10),
        )
        products = DataLayouts.VIJFH{M, 1, 4, 4, nothing}(
            device_array(device, zeros(FT, 1, 4, 4, 4, 5)),
        )
        matrix_array = Array(parent(matrices))
        column_matrices(i, j, h) = map(v -> M(matrix_array[v, i, j, :, h]), 1:Nv)
        for reverse_order in (false, true)
            column_reduce!(*, products, matrices; reverse = reverse_order)
            maybe_reverse = reverse_order ? reverse : identity
            @test all(Iterators.product(1:4, 1:4, 1:5)) do (i, j, h)
                product = M(Array(parent(products))[1, i, j, :, h])
                product ≈ foldl(*, maybe_reverse(column_matrices(i, j, h)))
            end
        end
    end
end

# A Tuple or NamedTuple argument of a slice loop is a container of arguments, which can be
# nested or empty, and its slice is the same container of their slices. Every argument in a
# container is scoped, masked, and checked for compatible dimensions like the other arguments,
# and a container can hold more arguments than a call can splat.
@testset "containers of loop arguments [$FT]" for FT in (Float32, Float64)
    device = ClimaComms.device()
    (a, b, c, dest) = ntuple(_ -> test_data(device, FT, 1, 10), 4)
    (a_array, b_array, c_array) = map(data -> Array(parent(data)), (a, b, c))

    nested_args = (a, (; b, c = (c,), e = ((),)))
    DataLayouts.foreach_column((), dest, nested_args) do e1, dest, (a, (; b, c, e))
        n = length(e1) + length(e[1])
        @. dest = a + b * c[1] + n
    end
    @test Array(parent(dest)) == a_array .+ b_array .* c_array
    DataLayouts.foreach_point((; dest, args = (a, b))) do (; dest, args)
        dest[] = args[1][] - args[2][]
    end
    @test Array(parent(dest)) == a_array .- b_array

    # Level slices need a static number of elements.
    (a_levels, dest_levels) =
        map(data -> DataLayouts.VIJFH{FT, 10, 4, 4, 5}(parent(data)), (a, dest))
    DataLayouts.foreach_level(
        (dest_levels, (a_levels,));
        enumerate = Val(true),
    ) do v, (dest, (a,))
        @. dest = a * v[1]
    end
    @test Array(parent(dest)) == a_array .* (1:10)
    DataLayouts.foreach_column((; dest, a); enumerate = Val(true)) do index, (; dest, a)
        fill!(dest, sum(a) + index[3])
    end
    @test Array(parent(dest)) ==
          repeat(sum(a_array; dims = 1), 10) .+ reshape(1:5, 1, 1, 1, 1, 5)

    mask = DataLayouts.IJHMask(dest)
    is_active = Array(parent(DataLayouts.is_active(mask)))
    is_active[1, 1, 1, 1, 1] = false
    copyto!(parent(DataLayouts.is_active(mask)), is_active)
    DataLayouts.set_mask_maps!(mask)
    fill!(parent(dest), 0)
    DataLayouts.foreach_column((dest, (a,)), (;); mask) do (dest, (a,)), e
        n = length(e)
        @. dest = a + n
    end
    DataLayouts.foreach_point(((),), (; a, dest); mask) do e, (; a, dest)
        dest[] += a[] + length(e[1])
    end
    masked_array = 2 .* a_array
    masked_array[:, 1, 1, :, 1] .= 0
    @test Array(parent(dest)) == masked_array

    # Point loops over containers use linear indices like loops over the same arguments, and a
    # NamedTuple of more values than a call can splat is sliced without allocating.
    @test DataLayouts.each_maskable_slice_index(DataLayouts.NoMask(), view, dest, (a, b)) ==
          eachindex(dest, a, b)
    names = ntuple(n -> Symbol(:x, n), 40)
    args = NamedTuple{names}(ntuple(_ -> test_data(device, FT, 1, 4), 40))
    sums = test_data(device, FT, 1, 4)
    sum_args!(sums, args) = DataLayouts.foreach_column(sums, args) do sums, args
        fill!(sums, 0)
        for n in 1:40
            @. sums += n * args[n]
        end
    end
    sum_args!(sums, args)
    @test Array(parent(sums)) == sum(n -> n .* Array(parent(args[n])), 1:40)
    if device isa ClimaComms.CPUSingleThreaded && TU.allocation_checks_meaningful()
        @test measured_allocations(Base.Fix2(sum_args!, args), sums) == 0
    end

    @test_throws DimensionMismatch DataLayouts.foreach_point(
        Returns(nothing),
        dest,
        (a, test_data(device, FT, 1, 1)),
    )
    @test_throws ArgumentError DataLayouts.foreach_column(Returns(nothing), (), ((;),))
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

# Views of views of StridedRanges must reindex without allocating (range subsets are
# tested with the 32-bit index helpers below).
@testset "views of views of StridedRanges" begin
    array = reshape(collect(1:24), 4, 6)
    for outer in (DataLayouts.StridedRange(2, 4, 10), 2:4:14)
        outer_view = view(array, outer)
        for inner in (Base.OneTo(2), 2:3, DataLayouts.StridedRange(1, 2, 3))
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

# The 32-bit index helpers of GPU kernels must agree with Base on every index.
@testset "32-bit index helpers agree with Base" begin
    for dims in ((7,), (3, 5), (4, 1, 6, 2), (1, 63, 1, 1), (5, 4, 4, 1, 7))
        cartesian = CartesianIndices(dims)
        offset = CartesianIndices(map(d -> 3:(d + 2), dims))
        for i in 1:prod(dims)
            for T in (Int32, UInt32)
                @test DataLayouts.cartesian_index_from_linear(dims, T(i)) == cartesian[i]
            end
            for T in (Int, Int32, UInt32)
                @test DataLayouts.single_axis_cartesian_index(offset, T(i)) == offset[i]
            end
        end
    end

    for I in (Int32, UInt32, Int), start in 1:4, step in 1:3, stop in 0:12
        range = DataLayouts.StridedRange(I(start), I(step), I(stop))
        values = collect(start:step:stop)
        @test collect(range) == values
        @test length(range) == length(values)
        @test isempty(range) == isempty(values)
        isempty(values) || @test last(range) == last(values)
        # Every range subindex must give back a range of an inferrable type: the generic
        # fallback allocates a Vector and builds an error string, which GPUs cannot
        # compile. Subindices whose steps are positive by type give StridedRanges.
        n = length(values)
        for sub in (Base.OneTo(n), 2:n, 1:UInt(2):n, DataLayouts.StridedRange(2, 2, n))
            @test (@inferred range[sub]) isa DataLayouts.StridedRange
            @test collect(range[sub]) == values[collect(sub)]
        end
        for sub in (1:2:n, n:-1:1, n:-2:1, 1:-1:2)
            @test (@inferred range[sub]) isa AbstractRange{Int}
            @test collect(range[sub]) == values[collect(sub)]
        end
    end
    for I in (Int32, UInt32, Int), n in 1:5, n_indices in 0:11, rank in 1:n
        @test collect(DataLayouts.strided_range(I(rank), I(n), n_indices)) ==
              collect(rank:n:n_indices)
    end

    # The Cartesian methods of a RegisterArray address the same storage as its
    # linear methods, for points owned by any thread.
    for (array_size, F) in (
            ((1, 4, 4, 3, 1), 4),
            ((9, 1, 1, 2, 1), 4),
            ((5, 1, 1, 1, 1), 4),
            ((6, 1, 1, 1), nothing),
        ),
        Stride in (1, 2, 4, 16)

        (; Nf, Np) = DataLayouts.register_array_params(array_size, Val(F))
        storage = StaticArrays.MArray{Tuple{cld(Np, Stride) * Nf}, Float64}(undef)
        array = DataLayouts.RegisterArray{array_size, F, Stride}(storage)
        for (i, index) in enumerate(CartesianIndices(array_size))
            @test DataLayouts.register_index(array, Tuple(index)) ==
                  DataLayouts.register_index(array, i)
        end
    end
end

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

    # Points of a column in registers are accessed through the Cartesian methods
    # of the array, also at linear indices (see has_cartesian_components).
    T = Tuple{Float64, Float64}
    column = DataLayouts.VIJFH{T, 9, 1, 1, 1}(Array{Float64})
    registers =
        DataLayouts.register_similar(DataLayouts.reassign(column, FakeGroup{16}()), T)
    @test DataLayouts.parent_type(registers) <: DataLayouts.RegisterArray
    for v in 1:9
        registers[v] = (v, -v)
        @test registers[v] == registers[CartesianIndex(v, 1, 1, 1)] == (v, -v)
        view(registers, v)[] = (2v, 3v)
        @test registers[CartesianIndex(v, 1, 1, 1)] == (2v, 3v)
        @test first(DataLayouts.array_and_index_args(registers, v)[2]) ==
              CartesianIndex(v, 1, 1, 1)
    end

    # Field views and frozen point views of registers stay in the scope's registers.
    @test DataLayouts.stored_in_scope_registers(registers.:2)
    for v in 1:9
        registers[v] = (v, -v)
        registers.:2[v] = 4v
        @test registers[v] == (v, 4v)
        frozen_point =
            DataLayouts.unchecked_rebuild(view(registers, v), StaticArrays.SArray)
        @test parent(parent(frozen_point)).array isa StaticArrays.SArray
        @test frozen_point[] == (v, 4v)
    end
end

# A launch unit holding only part of a subscope silently skips some points; see
# the whole subscopes invariant in DataLayouts.subscope_indices.
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

@testset "a launch unit holds whole subscopes" begin
    # Every sub-block width that the GPU scope chain descends through.
    for N in (2, 4, 8, 16, 32, 64, 128, 256)
        # With whole subscopes, every point of a slice is visited exactly once.
        for Np in (max(N - 1, 1), N, N + 1, 3N + 1)
            @test visited_points(FakeGroup{N}(), N, Np) == collect(1:Np)
        end

        # A launch unit holding only part of a subscope is what loses points.
        @test visited_points(FakeGroup{N}(), N - 1, N) != collect(1:N)
    end
end
