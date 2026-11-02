#=
julia --project=.buildkite
using Revise; include("test/Operators/gpu_shared_memory_levels.jl")
=#
using Test
import Logging
using ClimaComms, ClimaCore
ClimaComms.@import_required_backends
import ClimaCore: DataLayouts, Fields, Geometry, Operators
import CUDA
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

const CUDA_EXT = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
@assert ClimaComms.device() isa ClimaComms.CUDADevice
const FT = Float64

const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(;
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠwinterp = Operators.WeightedInterpolateC2F(;
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜdiv = Operators.DivergenceF2C()
const ᶠgrad = Operators.GradientC2F()
const ᶠfct = Operators.FCTZalesak()

function body!(o, x, y)
    t = @. ᶜdiv(ᶠgrad(x)) * y
    s = @. ᶜinterp(ᶠwinterp(x + y, t))
    @. o = t + s + ᶜinterp(ᶠinterp(t * s))
    return nothing
end
# FCTZalesak caches its two arguments at once: 1536 bytes per 64-level column.
function fct_body!(o, x, y)
    A = @. Geometry.Contravariant3Vector(ᶠinterp(x) * FT(0.1))
    Φ = @. tuple(ᶜinterp(ᶠinterp(y)) + 2, ᶜinterp(ᶠinterp(x)) + 2)
    @. o = ᶜdiv(ᶠfct(A, Φ))
    return nothing
end
# A temporary in registers that is read at neighboring levels is cached in shared memory
# whatever its size: 8 Float64s per level need 4032 bytes per 63-level column, and 16 need
# 8064 bytes.
tuple8(z) = ntuple(i -> i * z, Val(8))
tuple16(z) = ntuple(i -> i * z, Val(16))
function register_body!(o, z, _)
    t = @. tuple8(z)
    @. o = sum(ᶜinterp(ᶠinterp(t)))
    return nothing
end
function register16_body!(o, z, _)
    t = @. tuple16(z)
    @. o = sum(ᶜinterp(ᶠinterp(t)))
    return nothing
end

# Inner operators and expressions read at neighboring levels whose results do not fit are
# recomputed at every read, caching the values they read on their own.
tuple8_body!(o, x, y) = (@. o = sum(ᶜinterp(ᶠwinterp(sin(cos(x + y)), tuple8(x)))); nothing)
expression_body!(o, x, y) =
    (@. o = sum(ᶜinterp(tuple8(ᶠinterp(sin(cos(x + y)))))); nothing)
function recomputed_register_body!(o, x, y)
    t = @. ᶠinterp(sin(cos(x + y)))
    @. o = sum(ᶜinterp(tuple8(t)))
    return nothing
end

# A body whose statements keep many values live (its kernel compiles to 106 registers per
# thread on an A100 without a cap), with scalar buffers that fit the :medium level.
function heavy_body!(o, x, y)
    a = @. sin(x) * cos(y) + exp(-x * y)
    b = @. ᶜdiv(ᶠgrad(a)) * y + ᶜinterp(ᶠinterp(a * x))
    @. o =
        ᶜdiv(ᶠgrad(a * b)) * exp(a) + ᶜdiv(ᶠgrad(b * x)) * exp(b) +
        ᶜdiv(ᶠgrad(a + y)) * sin(b) + ᶜinterp(ᶠwinterp(a + b, x * y)) * cos(a) +
        ᶜinterp(ᶠinterp(a * b * x * y))^2 + atan(a, b) * tanh(a - b)
    return nothing
end

# Spectral element operators cannot recompute their buffers.
const grad = Operators.Gradient()
const wgrad = Operators.Gradient{Operators.WeakForm}()
const div = Operators.Divergence()
const wdiv = Operators.Divergence{Operators.WeakForm}()
const curl = Operators.Curl()
const wcurl = Operators.Curl{Operators.WeakForm}()
const split_div = Operators.SplitDivergence()
const lumped = Operators.LumpedRestriction()
function spectral_body!(o, x)
    lg = Fields.local_geometry_field(x)
    @. o =
        wdiv(grad(x)) + div(wgrad(x)) +
        split_div(Geometry.Contravariant12Vector(x, 3x), x) +
        Geometry.contravariant3(curl(Geometry.Covariant12Vector(x, 2x)), lg) +
        Geometry.contravariant1(wcurl(Geometry.Covariant123Vector(x, 2x, x^2)), lg) +
        lumped(x)
    return nothing
end
function spectral_register_body!(o, x)
    t = @. sin(x)
    @. o = wdiv(grad(t))
    return nothing
end

function columns(context; zelem = 63)
    space = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem,
        helem = 2,
        Nq = 4,
        context,
    )
    z = Fields.coordinate_field(space).z
    x = sin.(z)
    return (; x, y = cos.(z), z, xf = ᶠinterp.(x), o = similar(z))
end
const GPU = columns(ClimaComms.context())
const CPU = columns(ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()))
# Columns of fewer levels than a warp has lanes, whose slices are warps.
const SHORT_GPU = columns(ClimaComms.context(); zelem = 19)
const SHORT_CPU =
    columns(ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()); zelem = 19)

# Two buffers of different sizes in one body (19 Float64s, then 19 pairs of them), each
# allocated at the start of its sub-block's shared memory (see foreach_slice), after the
# first is consumed, and each thread reading its own points of them. The regions of the
# sub-blocks of a block are disjoint whatever their allocations, so a sub-block in another
# warp (which no barrier of this one orders) writing its second buffer does not overwrite
# the first buffer of this one (see the CUDA extension's scoped_static_array).
function two_buffers_body!(o, x, _)
    a = similar(x)
    @. a = x
    @. o = a
    b = similar(x, Tuple{FT, FT})
    @. b = tuple(x, 2 * x)
    @. o += first(b) + last(b)
    return nothing
end

# A broadcast that reads a Field in global memory at neighboring levels caches nothing.
interp_body!(o, _, xf) = (@. o = ᶜinterp(xf); nothing)

# A statement that skips the barrier publishing its result (for a loop that synchronizes
# once per statement), whose result is not read at other threads' points afterwards.
function unfinished_body!(o, x, _)
    bc = Base.Broadcast.broadcasted(ᶜinterp, Base.Broadcast.broadcasted(ᶠinterp, x))
    Operators.apply_operators!(o, Base.Broadcast.instantiate(bc), Val(false))
    return nothing
end

# The subscope of the slices of the one slice loop that f launches.
slice_loop_subscope(f) = TU.slice_subscope_type(only(TU.recorded_kernels(f))).instance

# Whether foreach_column gives the same values on the GPU as on the CPU, and the subscope
# of its slices on the GPU.
function compare(body!, a, b; gpu = GPU, cpu = CPU, kwargs...)
    subscope = slice_loop_subscope(
        () -> DataLayouts.foreach_column(body!, gpu.o, gpu[a], gpu[b]; kwargs...),
    )
    DataLayouts.foreach_column(body!, cpu.o, cpu[a], cpu[b]; kwargs...)
    return (isapprox(Array(parent(gpu.o)), parent(cpu.o); nans = true), subscope)
end

error_message(f) =
    try
        f()
        ""
    catch err
        err isa ArgumentError ? err.msg : rethrow()
    end

@testset "shared memory levels of foreach_slice" begin
    bytes(level, N = 64) = CUDA_EXT.shmem_level_bytes(level, Val(N))
    level(name, N = 64) = CUDA_EXT.ThisSubBlock{N, bytes(Val(name), N)}()
    # The level at which `needed` bytes of buffers fit on a sub-block of N threads (the first
    # that holds them, whose bytes per thread the device sets; see the CUDA extension's
    # slice_subscope_at_level), or nothing beyond :max, and the subscope that the :auto
    # level of a broadcast then chooses.
    function fitting(needed, N = 64)
        names = DataLayouts.SHMEM_LEVELS
        i = findfirst(name -> needed <= N * bytes(Val(name), N), names)
        return isnothing(i) ? nothing : names[i]
    end
    auto_level(needed, N = 64) =
        isnothing(fitting(needed, N)) ? DataLayouts.ThisThread() :
        level(fitting(needed, N), N)
    # The subscope of a loop whose buffers are known (e.g., a broadcast's): :medium if it
    # holds them, or else exactly their bytes, up to :max.
    known_level(needed, N = 64) =
        needed <= N * bytes(Val(:medium), N) ? level(:medium, N) :
        isnothing(fitting(needed, N)) ? DataLayouts.ThisThread() :
        CUDA_EXT.ThisSubBlock{N, cld(needed, N)}()
    # The levels come from the device: 64-thread blocks run at full occupancy in as many
    # blocks as a multiprocessor holds (32 on an A100), and blocks of narrow sub-blocks
    # (1024 threads) in as many as its threads allow (2 on an A100).
    attrs = CUDA_EXT.device_attributes()
    full_occupancy_blocks(block_threads) =
        min(attrs.max_blocks_per_sm, attrs.max_threads_per_sm ÷ block_threads)
    medium(block_threads) = fld(
        fld(attrs.max_shmem_per_sm, full_occupancy_blocks(block_threads)) -
        attrs.reserved_shmem_per_block,
        block_threads,
    )
    large(block_threads) = fld(
        fld(attrs.max_shmem_per_sm, max(1, full_occupancy_blocks(block_threads) ÷ 2)) -
        attrs.reserved_shmem_per_block,
        block_threads,
    )
    @test bytes(Val(:max)) == fld(attrs.max_shmem_per_block_optin, 64)
    @test bytes(Val(:medium)) == medium(64)
    @test bytes(Val(:medium), 16) == medium(1024)
    @test bytes(Val(:large)) == large(64)
    @test bytes(Val(:large), 16) == large(1024)
    @test bytes(Val(:medium)) < bytes(Val(:large)) <= bytes(Val(:max))
    if (attrs.max_shmem_per_sm, attrs.max_shmem_per_block_optin) == (167936, 166912) # an A100
        @test bytes(Val(:max)) == 2608
        @test bytes(Val(:medium)) == 66
        @test bytes(Val(:medium), 16) == 81
        @test bytes(Val(:large)) == 148
        @test bytes(Val(:large), 16) == 163
    end
    @test bytes(Val(100)) == 100
    @test bytes(Val(:none)) == 0

    # A loop body uses the :medium level unless another one is chosen.
    (ok, subscope) = compare(body!, :x, :y)
    @test ok && subscope === level(:medium)
    @test CUDA_EXT.next_shmem_level(level(0)) === level(:medium)
    @test CUDA_EXT.next_shmem_level(level(:medium)) === level(:large)
    @test CUDA_EXT.next_shmem_level(level(:large)) === level(:max)
    @test CUDA_EXT.next_shmem_level(level(:max)) === DataLayouts.ThisThread()
    for name in (:medium, :large, :max, 100)
        (ok, subscope) = compare(body!, :x, :y; shmem = Val(name))
        @test ok
        @test subscope === level(name)
    end
    (ok, subscope) = compare(body!, :x, :y; shmem = Val(:none))
    @test ok && subscope === DataLayouts.ThisThread()
    # A launch above the default limit on dynamic shared memory per block (48 KB) raises the
    # function's limit. A device that cannot opt in above the default (like a P100) has its
    # :max level at the default, so the launch leaves the limit alone.
    max_kernel = only(
        TU.recorded_kernels(
            () -> DataLayouts.foreach_column(body!, GPU.o, GPU.x, GPU.y; shmem = Val(:max)),
        ),
    )
    limit =
        CUDA.attributes(max_kernel.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]
    max_bytes = 64 * bytes(Val(:max))
    @test max_bytes > CUDA_EXT.MAX_SHMEM_PER_BLOCK ? limit == max_bytes :
          limit <= CUDA_EXT.MAX_SHMEM_PER_BLOCK
    @test_throws ArgumentError DataLayouts.foreach_column(
        body!, GPU.o, GPU.x, GPU.y; shmem = Val(:huge),
    )
    # A level beyond the shared memory per block of the device cannot be launched, and the
    # error names the dynamic and static shared memory of a block and the limit.
    too_many = bytes(Val(:max)) + 1
    message = error_message(
        () -> DataLayouts.foreach_column(body!, GPU.o, GPU.x, GPU.y; shmem = Val(too_many)),
    )
    @test occursin("a block of 64 threads needs $(64 * too_many) bytes of dynamic", message)
    @test occursin(
        "bytes of static shared memory, more than the $(attrs.max_shmem_per_block_optin) \
         bytes of shared memory available per block",
        message,
    )

    # Buffers that are live at the same time are placed after one another.
    @test first(compare(fct_body!, :x, :y))

    # The buffers of the sub-blocks of a block are in disjoint regions of its shared memory,
    # so that sub-blocks in different warps cannot race. A loop body over slices of at most
    # a warp is given no shared memory, and moves to :medium when it allocates some.
    for _ in 1:8
        (ok, subscope) =
            compare(two_buffers_body!, :x, :y; gpu = SHORT_GPU, cpu = SHORT_CPU)
        @test ok
        @test subscope === level(:medium, 32)
    end
    (ok, subscope) = compare(body!, :x, :y; gpu = SHORT_GPU, cpu = SHORT_CPU)
    @test ok && subscope === level(0, 32)
    for name in (:medium, 20)
        (ok, subscope) =
            compare(body!, :x, :y; gpu = SHORT_GPU, cpu = SHORT_CPU, shmem = Val(name))
        @test ok && subscope === level(name, 32)
    end
    # FCTZalesak's register temporaries are cached in registers too on slices of a warp.
    (ok, subscope) = compare(fct_body!, :x, :y; gpu = SHORT_GPU, cpu = SHORT_CPU)
    @test ok && subscope === level(0, 32)

    # A statement can skip the barrier that publishes its result.
    @test first(compare(unfinished_body!, :x, :y))

    # A level is the most shared memory that a slice is given: buffers that do not fit at a
    # level are reported with the bytes they need and the level's bytes by a warning, and
    # the loop is launched at the next level that holds them, with a warning for each level
    # it moves past; the loop's later launches are silent (whether the kernel of a level
    # compiles is cached by its type, which names the level). The cache is cleared before
    # each case, since a level that an earlier case found not to compile is passed silently
    # (the explicit level of one case can be a named level on some devices).
    # The warnings of the extension while f runs (a kernel that fails to compile can make
    # GPUCompiler warn about its debug information as well).
    is_level_warning(log) = log.level == Logging.Warn && log._module === CUDA_EXT
    warnings(f) = filter(is_level_warning, first(Test.collect_test_logs(f)))
    walked_levels(per_thread, needed) =
        1 + count(DataLayouts.SHMEM_LEVELS) do name
            bytes(Val(name)) > per_thread && 64 * bytes(Val(name)) < needed
        end
    for (body!, (a, b), shmem, needed, per_thread) in (
        (fct_body!, (:x, :y), Val(16), 1536, 16),
        (register_body!, (:z, :z), Val(32), 4032, 32),
        (register_body!, (:z, :z), Val(16), 4032, 16),
        (register16_body!, (:z, :z), Val(100), 8064, 100),
    )
        empty!(CUDA_EXT.SHMEM_LEVEL_CACHE)
        (logs, (ok, subscope)) = Test.collect_test_logs(() -> compare(body!, a, b; shmem))
        @test ok && subscope === auto_level(needed)
        logged = filter(is_level_warning, logs)
        @test length(logged) == walked_levels(per_thread, needed)
        @test occursin(
            "$(body!) needs $needed bytes of shared memory for slices of 64 threads, more \
             than the $(64 * per_thread) bytes of its shmem level ($per_thread bytes per \
             thread); it is launched",
            first(logged).message,
        )
        @test occursin("Pass shmem = Val(", first(logged).message)
        @test isempty(warnings(() -> compare(body!, a, b; shmem)))
    end
    # The buffers of a loop body are not known from its slices, so it is launched at
    # :medium, and at the next level when they do not fit there (at :large for 8064 bytes
    # on an A100), with the same warning.
    kernel_subscope(kernel_type) =
        only(filter(T -> T <: DataLayouts.DataScope, collect(fieldtypes(kernel_type))))
    cached_levels(body!) = [
        (kernel_subscope(kernel_type).instance, compiles) for
        ((kernel_type, _), compiles) in CUDA_EXT.SHMEM_LEVEL_CACHE if
        typeof(body!) in fieldtypes(kernel_type)
    ]
    for (body!, needed) in ((register_body!, 4032), (register16_body!, 8064))
        empty!(CUDA_EXT.SHMEM_LEVEL_CACHE)
        (logs, (ok, subscope)) = Test.collect_test_logs(() -> compare(body!, :z, :z))
        @test ok && subscope === auto_level(needed)
        logged = filter(is_level_warning, logs)
        @test length(logged) == walked_levels(0, needed) - 1
        isempty(logged) || @test occursin(
            "$(body!) needs $needed bytes of shared memory for slices of 64 threads, more \
             than the $(64 * bytes(Val(:medium))) bytes of its shmem level \
             ($(bytes(Val(:medium))) bytes per thread); it is launched at the :large level",
            first(logged).message,
        )
        (ok, subscope) = compare(body!, :z, :z)
        @test ok && subscope === auto_level(needed)
        @test isempty(warnings(() -> compare(body!, :z, :z)))
        names = DataLayouts.SHMEM_LEVELS
        for name in names[1:something(findfirst(==(fitting(needed)), names), length(names))]
            @test (level(name), name == fitting(needed)) in cached_levels(body!)
        end
    end
    (ok, subscope) = compare(register16_body!, :z, :z; shmem = Val(:max))
    @test ok && subscope === level(:max)

    # Results of inner operators and lazy expressions that do not fit are recomputed, in
    # loop bodies and in broadcasts.
    for body! in (tuple8_body!, expression_body!, recomputed_register_body!)
        for name in (32, :medium)
            (ok, subscope) = compare(body!, :x, :y; shmem = Val(name))
            @test ok && subscope === level(name)
        end
    end
    for body! in (tuple8_body!, expression_body!) # a broadcast each
        subscope = slice_loop_subscope(() -> body!(GPU.o, GPU.x, GPU.y))
        body!(CPU.o, CPU.x, CPU.y)
        @test subscope === level(:medium)
        @test Array(parent(GPU.o)) ≈ parent(CPU.o)
    end

    # A broadcast that caches nothing is given no shared memory, whereas a loop body that
    # caches nothing is launched at :medium (its buffers are not known from its slices).
    subscope = slice_loop_subscope(() -> interp_body!(GPU.o, GPU.x, GPU.xf))
    interp_body!(CPU.o, CPU.x, CPU.xf)
    @test subscope === level(0)
    @test Array(parent(GPU.o)) ≈ parent(CPU.o)
    (ok, subscope) = compare(interp_body!, :x, :xf)
    @test ok && subscope === level(:medium)

    # Spectral element operators on 8x8 slabs: the gradient of a Float64 field caches
    # 8 bytes per point per dimension, 512 bytes per slab, which 4 bytes per thread cannot
    # hold (the error names one of the buffers over the level, up to 1024 bytes here).
    (slab_gpu, slab_cpu) = map((GPU.z, CPU.z)) do z
        space = TU.CenterExtrudedFiniteDifferenceSpace(
            FT;
            zelem = 4,
            helem = 2,
            Nq = 8,
            context = ClimaComms.context(z),
        )
        coords = Fields.coordinate_field(space)
        @. sin(coords.long * FT(π) / 180) * (1 + coords.z) + cos(coords.lat * FT(π) / 90)
    end
    function compare_slabs(body!; kwargs...)
        (dest_gpu, dest_cpu) = (similar(slab_gpu), similar(slab_cpu))
        subscope = slice_loop_subscope(
            () -> DataLayouts.foreach_slab(body!, dest_gpu, slab_gpu; kwargs...),
        )
        DataLayouts.foreach_slab(body!, dest_cpu, slab_cpu)
        return (Array(parent(dest_gpu)) ≈ parent(dest_cpu), subscope)
    end
    for (body!, needs) in
        (
        (spectral_body!, r"needs (512|1024) bytes"),
        (spectral_register_body!, "512 bytes"),
    )
        (ok, subscope) = compare_slabs(body!)
        @test ok && subscope === level(:medium)
        (logs, (ok, subscope)) =
            Test.collect_test_logs(() -> compare_slabs(body!; shmem = Val(4)))
        @test ok && subscope === level(:medium)
        logged = filter(is_level_warning, logs)
        @test length(logged) == 1
        @test occursin(needs, first(logged).message)
        @test occursin(
            "(4 bytes per thread); it is launched at the :medium level",
            first(logged).message,
        )
    end

    # A broadcast is given the :medium level if its buffers fit, or else their bytes: on an
    # 8x8 slab, the gradient of 3 Float64s needs 3072 bytes (the :medium level on an A100),
    # of 8 Float64s 8192 bytes and of 90 Float64s 92160 bytes (up to :max), and of 170
    # Float64s 174080 bytes, more than any level, so each of its slabs is evaluated by one
    # thread.
    tuple3(z) = ntuple(i -> i * z, Val(3))
    for (tuple_f, expected) in ((tuple3, known_level(3072)), (tuple8, known_level(8192)))
        (tg, tc) = map((GPU.z, CPU.z)) do z
            space = TU.CenterExtrudedFiniteDifferenceSpace(
                FT;
                zelem = 10,
                helem = 2,
                Nq = 8,
                context = ClimaComms.context(z),
            )
            coords = Fields.coordinate_field(space)
            map((λ, z) -> tuple_f(sin(λ * FT(π) / 180) * (1 + z)), coords.long, coords.z)
        end
        dest = similar(grad.(tg))
        subscope = slice_loop_subscope(() -> (@. dest = grad(tg)))
        @test subscope === expected
        @test Array(parent(dest)) ≈ parent(grad.(tc))
    end
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 10, helem = 2, Nq = 8)
    for (n, expected) in ((90, known_level(92160)), (170, known_level(174080)))
        tg = Fields.Field(NTuple{n, FT}, space) # not initialized; only its type is read
        bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(grad, tg))
        dest = Fields.Field(eltype(bc), space)
        subscope = CUDA_EXT.slice_subscope_at_level(
            CUDA_EXT.ThisSubBlock{64}(),
            Val(:auto),
            Operators.slice_operator(bc.style),
            Operators.apply_operators!,
            dest,
            bc,
        )
        @test subscope === expected
    end
end

@testset "register caps of foreach_slice" begin
    # No cap by default, for every kind of slice; the cap tested below is the count that
    # keeps half of a multiprocessor's threads resident (64 on an A100).
    attrs = CUDA_EXT.device_attributes()
    subblock = CUDA_EXT.ThisWarp()
    thread = DataLayouts.ThisThread()
    cap = fld(attrs.max_regs_per_sm ÷ (attrs.max_threads_per_sm ÷ 2), 8) * 8
    (attrs.max_regs_per_sm, attrs.max_threads_per_sm) == (65536, 2048) && @test cap == 64
    for level in (:auto, :none), scope in (subblock, thread)
        @test isnothing(CUDA_EXT.register_cap(Val(level), scope))
    end
    @test CUDA_EXT.register_cap(Val(40), subblock) == 40
    @test CUDA_EXT.register_cap(Val(40), thread) == 40

    # Registers per thread of the one kernel that f launches.
    registers(f) = CUDA.registers(only(TU.recorded_kernels(f)))
    function run_heavy(; kwargs...)
        regs = registers(
            () ->
                DataLayouts.foreach_column(heavy_body!, GPU.o, GPU.x, GPU.y; kwargs...),
        )
        return (regs, Array(parent(GPU.o)))
    end
    DataLayouts.foreach_column(heavy_body!, CPU.o, CPU.x, CPU.y)
    reference = parent(CPU.o)

    # The body exceeds the cap by default, which an explicit cap holds it under, with the
    # same values (the cap only moves values to local memory; the boundary levels are NaN,
    # as for the other bodies with these operators).
    (uncapped_regs, uncapped) = run_heavy(; maxregs = Val(:none))
    @test uncapped_regs > cap
    @test isapprox(uncapped, reference; nans = true)
    for kwargs in ((;), (; maxregs = Val(:auto)))
        (regs, values) = run_heavy(; kwargs...)
        @test regs == uncapped_regs
        @test isequal(values, uncapped)
    end
    for cap in (cap, cap - 16)
        (regs, values) = run_heavy(; maxregs = Val(cap))
        @test regs <= cap
        @test isequal(values, uncapped)
    end
    # A cap above the body's count leaves the choice to ptxas (which may shift the
    # allocation by a few registers).
    (regs, values) = run_heavy(; maxregs = Val(255))
    @test regs > cap
    @test isequal(values, uncapped)

    # Loops whose slices are each evaluated by one thread take an explicit cap and are
    # uncapped by default.
    (uncapped_regs, uncapped) = run_heavy(; shmem = Val(:none), maxregs = Val(:none))
    @test isapprox(uncapped, reference; nans = true)
    for kwargs in ((;), (; maxregs = Val(:auto)))
        (regs, values) = run_heavy(; shmem = Val(:none), kwargs...)
        @test regs == uncapped_regs
        @test isequal(values, uncapped)
    end
    (regs, values) = run_heavy(; shmem = Val(:none), maxregs = Val(32))
    @test regs <= 32 < uncapped_regs
    @test isequal(values, uncapped)
    @test_throws ArgumentError DataLayouts.foreach_column(
        heavy_body!, GPU.o, GPU.x, GPU.y; maxregs = Val(:device),
    )
end
