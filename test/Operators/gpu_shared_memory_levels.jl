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
# whatever its size: 8 Float64s per level need 4032 bytes per 63-level column.
tuple8(z) = ntuple(i -> i * z, Val(8))
function register_body!(o, z, _)
    t = @. tuple8(z)
    @. o = sum(ᶜinterp(ᶠinterp(t)))
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

function slice_loop_subscope(f)
    empty!(CUDA_EXT.RECORDED_KERNELS)
    CUDA_EXT.RECORD_KERNELS[] = true
    try
        f()
    finally
        CUDA_EXT.RECORD_KERNELS[] = false
    end
    kernel = only(splice!(CUDA_EXT.RECORDED_KERNELS, eachindex(CUDA_EXT.RECORDED_KERNELS)))
    return typeof(kernel).parameters[1].parameters[5].instance
end

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
    bytes(level) = CUDA_EXT.shmem_level_bytes(level)
    @test bytes(Val(:max)) == CUDA_EXT.SHMEM_PER_THREAD
    @test bytes(Val(:medium)) == CUDA_EXT.SHMEM_PER_THREAD ÷ 2
    @test bytes(Val(:low)) == CUDA_EXT.SHMEM_PER_THREAD ÷ 4
    @test bytes(Val(100)) == 100

    (ok, subscope) = compare(body!, :x, :y)
    @test ok && subscope == CUDA_EXT.ThisSubBlock{64, bytes(Val(:medium))}()
    for level in (Val(:low), Val(:medium), Val(:max), Val(100))
        (ok, subscope) = compare(body!, :x, :y; shmem = level)
        @test ok
        @test subscope isa CUDA_EXT.ThisSubBlock{64, bytes(level)}
    end
    (ok, subscope) = compare(body!, :x, :y; shmem = Val(:none))
    @test ok && subscope == DataLayouts.ThisThread()

    # Buffers that are live at the same time are placed after one another.
    @test first(compare(fct_body!, :x, :y))

    # Buffers that do not fit are reported with a level that fits.
    message = error_message(
        () -> DataLayouts.foreach_column(fct_body!, GPU.o, GPU.x, GPU.y; shmem = Val(:low)),
    )
    @test occursin("1536 bytes", message)
    @test occursin("pass shmem = Val(:medium) or shmem = Val(:none)", message)
    message = error_message(
        () -> DataLayouts.foreach_column(register_body!, GPU.o, GPU.z, GPU.z),
    )
    @test occursin("pass shmem = Val(:max)", message)
    message = error_message(
        () -> DataLayouts.foreach_column(
            register_body!, GPU.o, GPU.z, GPU.z; shmem = Val(16),
        ),
    )
    @test occursin("shmem = Val(16)", message)
    (ok, subscope) = compare(register_body!, :z, :z; shmem = Val(:max))
    @test ok && subscope isa CUDA_EXT.ThisSubBlock{64, bytes(Val(:max))}

    @test_throws ArgumentError DataLayouts.foreach_column(
        body!, GPU.o, GPU.x, GPU.y; shmem = Val(:huge),
    )

    # A broadcast whose buffers do not fit is evaluated with one thread per slab: the
    # gradient of 8 Float64s on an 8x8 slab needs 8192 bytes.
    grad = Operators.Gradient()
    (tg, tc) = map((GPU.z, CPU.z)) do z
        space = TU.CenterExtrudedFiniteDifferenceSpace(
            FT;
            zelem = 10,
            helem = 2,
            Nq = 8,
            context = ClimaComms.context(z),
        )
        coords = Fields.coordinate_field(space)
        map((λ, z) -> tuple8(sin(λ * FT(π) / 180) * (1 + z)), coords.long, coords.z)
    end
    dest = similar(grad.(tg))
    subscope = slice_loop_subscope(() -> (@. dest = grad(tg)))
    @test subscope == DataLayouts.ThisThread()
    @test Array(parent(dest)) ≈ parent(grad.(tc))
end
