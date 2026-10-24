#=
julia --project=.buildkite
using Revise; include("test/Operators/gpu_shared_memory_levels.jl")
=#
using Test
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

function columns(context)
    space = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 63,
        helem = 2,
        Nq = 4,
        context,
    )
    z = Fields.coordinate_field(space).z
    return (; x = sin.(z), y = cos.(z), z, o = similar(z))
end
const GPU = columns(ClimaComms.context())
const CPU = columns(ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()))

# The subscope of the slices of the one slice loop that f launches.
slice_loop_subscope(f) = TU.slice_subscope_type(only(TU.recorded_kernels(f))).instance

# Whether foreach_column gives the same values on the GPU as on the CPU, and the subscope
# of its slices on the GPU.
function compare(body!, a, b; kwargs...)
    subscope = slice_loop_subscope(
        () -> DataLayouts.foreach_column(body!, GPU.o, GPU[a], GPU[b]; kwargs...),
    )
    DataLayouts.foreach_column(body!, CPU.o, CPU[a], CPU[b]; kwargs...)
    return (isapprox(Array(parent(GPU.o)), parent(CPU.o); nans = true), subscope)
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
    @test bytes(Val(:max)) == fld(attrs.max_shmem_per_block_optin, 64)
    @test bytes(Val(:medium)) == medium(64)
    @test bytes(Val(:medium), 16) == medium(1024)
    if (attrs.max_shmem_per_sm, attrs.max_shmem_per_block_optin) == (167936, 166912) # an A100
        @test bytes(Val(:max)) == 2608
        @test bytes(Val(:medium)) == 66
        @test bytes(Val(:medium), 16) == 81
    end
    @test bytes(Val(100)) == 100
    @test bytes(Val(:none)) == 0

    # A loop body uses the :medium level unless another one is chosen.
    (ok, subscope) = compare(body!, :x, :y)
    @test ok && subscope === level(:medium)
    for name in (:medium, :max, 100)
        (ok, subscope) = compare(body!, :x, :y; shmem = Val(name))
        @test ok
        @test subscope === level(name)
    end
    (ok, subscope) = compare(body!, :x, :y; shmem = Val(:none))
    @test ok && subscope === DataLayouts.ThisThread()
    @test_throws ArgumentError DataLayouts.foreach_column(
        body!, GPU.o, GPU.x, GPU.y; shmem = Val(:huge),
    )

    # Buffers that are live at the same time are placed after one another.
    @test first(compare(fct_body!, :x, :y))

    # Buffers that do not fit are reported with the bytes they need and the level's bytes.
    for (body!, (a, b), shmem, needs, per_thread) in (
        (fct_body!, (:x, :y), Val(16), "1536 bytes", 16),
        (register_body!, (:z, :z), Val(32), "4032 bytes", 32),
        (register_body!, (:z, :z), Val(16), "4032 bytes", 16),
        (register16_body!, (:z, :z), Val(:auto), "8064 bytes", bytes(Val(:medium))),
    )
        message = error_message(
            () -> DataLayouts.foreach_column(body!, GPU.o, GPU[a], GPU[b]; shmem),
        )
        @test occursin(needs, message)
        @test occursin("more than the $per_thread bytes per thread", message)
    end
    # The buffers of a loop body are not known from its slices, so it runs at :medium, or
    # throws when they do not fit there (as on a device with less shared memory).
    if fitting(4032) == :medium
        (ok, subscope) = compare(register_body!, :z, :z)
        @test ok && subscope === level(:medium)
    else
        message = error_message(() -> compare(register_body!, :z, :z))
        @test occursin("4032 bytes", message)
        @test occursin("more than the $(bytes(Val(:medium))) bytes per thread", message)
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
        message = error_message(() -> compare_slabs(body!; shmem = Val(4)))
        @test occursin(needs, message)
        @test occursin("more than the 4 bytes per thread", message)
    end

    # A broadcast chooses the first level whose buffers fit: on an 8x8 slab, the gradient
    # of 3 Float64s needs 3072 bytes (the :medium level on an A100), of 8 Float64s 8192
    # bytes and of 90 Float64s 92160 bytes (:max), and of 170 Float64s 174080 bytes, more
    # than any level, so each of its slabs is evaluated by one thread.
    tuple3(z) = ntuple(i -> i * z, Val(3))
    for (tuple_f, expected) in ((tuple3, auto_level(3072)), (tuple8, auto_level(8192)))
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
    for (n, expected) in ((90, auto_level(92160)), (170, auto_level(174080)))
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
