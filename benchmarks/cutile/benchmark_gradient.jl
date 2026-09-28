#=
1v1 GPU benchmark: ClimaCore's spectral-element Gradient vs a cuTile.jl
implementation of the same tensor contraction.

Contenders (all produce identical covariant components; correctness is gated
against the ClimaCore CPU result before any timing):
  - climacore:  `@. ∇f = grad(f)` — the production fused-slab kernel
  - cutile:     one batched GEMM per element (see gradient_kernels_cutile.jl)
  - cuda_ref:   hand-written flop-matched CUDA.jl kernel
  - copy_floor: same memory traffic (1R + 2W), zero flops — bandwidth ceiling

Usage (GPU node; see README.md for environment setup):
    export CLIMACOMMS_DEVICE=CUDA
    julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_gradient.jl \
        --float-type Float64 --helem 30 --zelem 63 --nq 4 --contenders all

Tiny-config correctness check (validates the KronGEMM weight ordering):
    ... benchmark_gradient.jl --helem 2 --zelem 4
=#

import CUDA
import ClimaComms
import ClimaCore:
    DataLayouts,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
import BenchmarkTools
using Printf: @printf

include(joinpath(@__DIR__, "gradient_kernels.jl"))
include(joinpath(@__DIR__, "..", "scripts", "benchmark_utils.jl"))

##### CLI

function getarg(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    i < length(ARGS) || error("missing value after $flag")
    v = ARGS[i + 1]
    return default isa Int ? parse(Int, v) : v
end

const FT = Dict("Float64" => Float64, "Float32" => Float32)[getarg(
    "--float-type",
    "Float64",
)]
const helem = getarg("--helem", 60)
const zelem = getarg("--zelem", 63)
const Nq = getarg("--nq", 4)
const tv = getarg("--tv", 64) # cuTile tile size along v (power of two)
const contender_arg = getarg("--contenders", "all")
const contenders =
    contender_arg == "all" ? ["copy_floor", "cuda_ref", "climacore", "cutile"] :
    String.(split(contender_arg, ","))

##### Preflight

function preflight(; need_cutile)
    VERSION >= v"1.11" || error(
        "cuTile.jl requires Julia >= 1.11 (this is $VERSION); use `julia +1.11`.",
    )
    CUDA.functional() || error(
        "CUDA is not functional on this node. Run on a GPU node " *
        "(e.g. `srun --gres=gpu:a100:1 ...`) with CLIMACOMMS_DEVICE=CUDA.",
    )
    CUDA.versioninfo()
    println()
    if need_cutile
        drv = CUDA.driver_version()
        drv >= v"13" || error(
            "cuTile requires an NVIDIA driver supporting CUDA 13 (driver >= 580); " *
            "this node reports CUDA $drv. Try `export JULIA_CUDA_USE_COMPAT=true` " *
            "(CUDA.jl forward-compat libcuda) or request a node with a newer driver.",
        )
        cap = CUDA.capability(CUDA.device())
        cap >= v"8.0" || error(
            "cuTile requires compute capability >= 8.0 (Ampere+); " *
            "this GPU ($(CUDA.name(CUDA.device()))) has $cap.",
        )
        ispow2(Nq * Nq) || error(
            "the cuTile contender requires power-of-two tile extents; " *
            "Nq² = $(Nq * Nq) is not one. Use --nq 4 or drop `cutile` from --contenders.",
        )
    end
end

preflight(; need_cutile = "cutile" in contenders)
if "cutile" in contenders
    include(joinpath(@__DIR__, "gradient_kernels_cutile.jl"))
end

##### Spaces and fields

function create_space(context; float_type, h_elem, z_elem, n_quad)
    radius = float_type(6.37122e6)
    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, h_elem)
    htopology = Topologies.Topology2D(context, hmesh)
    quad = Quadratures.GLL{n_quad}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{float_type}(0),
        Geometry.ZPoint{float_type}(30e3);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = z_elem)
    vtopology = Topologies.IntervalTopology(context, vertmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    return Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
end

# Deterministic smooth initial condition (reproducible across runs/devices).
function init_field!(f)
    space = axes(f)
    coords = Fields.coordinate_field(space)
    @. f =
        FT(2) +
        sind(coords.long) * cosd(coords.lat) * (1 + coords.z / FT(30e3))
    return f
end

device = ClimaComms.CUDADevice()
gpu_space = create_space(
    ClimaComms.context(device);
    float_type = FT,
    h_elem = helem,
    z_elem = zelem,
    n_quad = Nq,
)
cpu_space = create_space(
    ClimaComms.context(ClimaComms.CPUSingleThreaded());
    float_type = FT,
    h_elem = helem,
    z_elem = zelem,
    n_quad = Nq,
)

grad = Operators.Gradient()

# Initialize on the CPU and copy those nodal values to the device. Re-evaluating
# sind/cosd on the GPU disagrees by a few ulps; the GLL differentiation matrix
# (row 1-norm ≈ 9 at Nq = 4) turns that into an O(1e-4) covariant-component
# error in Float32, which fails the 1e-5 gate even when the contraction matches.
f_cpu = init_field!(zeros(cpu_space))
f_gpu = zeros(gpu_space)
parent(Fields.field_values(f_gpu)) .= parent(Fields.field_values(f_cpu))
∇f_cpu = @. grad(f_cpu) # CPU oracle on the same nodal values
∇f_gpu = @. grad(f_gpu) # materializes the output field; reused in-place below

fv = Fields.field_values(f_gpu)
fv isa DataLayouts.VIJFH || error(
    "expected the default field layout to be VIJFH (parent dims (Nv, Ni, Nj, F, Nh)); " *
    "got $(typeof(fv)). The array contenders assume this axis order.",
)

pf = parent(fv)                                 # (Nv, Nq, Nq, 1, Nh)
p∇_oracle = parent(Fields.field_values(∇f_cpu)) # CPU (Nv, Nq, Nq, 2, Nh)
(Nv, _, _, _, Nh) = size(pf)
@assert size(pf) == (Nv, Nq, Nq, 1, Nh)

D = Quadratures.differentiation_matrix(FT, Quadratures.GLL{Nq}())
K = Nq * Nq
C = 2 * K

out_ref = CUDA.zeros(FT, Nv, Nq, Nq, 2, Nh)
out_floor = CUDA.zeros(FT, Nv, Nq, Nq, 2, Nh)
out_cutile = CUDA.zeros(FT, Nv, Nq, Nq, 2, Nh)
F3 = reshape(pf, Nv, K, Nh)           # contiguous, no copy
O3 = reshape(out_cutile, Nv, C, Nh)   # contiguous, no copy
Wt = CUDA.CuArray(gradient_weight(D)) # K × C

##### Contender closures (zero-arg; launches only, sync happens in the timers)

runners = Dict{String, Function}()
outputs = Dict{String, Function}()
if "climacore" in contenders
    runners["climacore"] = () -> begin
        @. ∇f_gpu = grad(f_gpu)
        nothing
    end
    outputs["climacore"] = () -> parent(Fields.field_values(∇f_gpu))
end
if "cuda_ref" in contenders
    runners["cuda_ref"] = () -> launch_grad_cuda_ref!(out_ref, pf, D)
    outputs["cuda_ref"] = () -> out_ref
end
if "cutile" in contenders
    runners["cutile"] = () -> launch_grad_cutile!(F3, Wt, O3; tv, K, C)
    outputs["cutile"] = () -> out_cutile
end
if "copy_floor" in contenders
    runners["copy_floor"] = () -> launch_copy_floor!(out_floor, pf, Val(Nq))
    # copy_floor is traffic-matched, not math-matched: no correctness gate
end

##### Correctness gate (before any timing)

rtol = FT == Float64 ? 1e-12 : 1e-5
println("Correctness gate (rtol = $rtol, vs ClimaCore CPU oracle):")
for name in filter(in(keys(runners)), ["climacore", "cuda_ref", "cutile"])
    runners[name]()
    CUDA.synchronize()
    result = Array(outputs[name]())
    ok = isapprox(result, p∇_oracle; rtol)
    maxrel =
        maximum(abs.(result .- p∇_oracle)) /
        max(maximum(abs.(p∇_oracle)), eps(FT))
    @printf(
        "  %-10s %s (max rel-scale error %.3e)\n",
        name,
        ok ? "PASS" : "FAIL",
        maxrel,
    )
    ok || error(
        "$name does not match the CPU oracle — aborting before timing. " *
        "The oracle is Gradient on the same nodal values. A `cutile` failure " *
        "can indicate KronGEMM weight ordering or a masked out-of-bounds " *
        "store; a Float32 `cutile` failure can indicate TF32 demotion in the " *
        "tile matmul.",
    )
end

##### Timing

function time_contender!(run!, device; ntrials = 100, ninner = 10)
    run!() # compile
    CUDA.synchronize()
    trial = BenchmarkTools.@benchmark ClimaComms.@cuda_sync $device $(run!)()
    t_bt = minimum(trial.times) * 1e-9
    t_loop = Inf
    for _ in 1:ntrials
        t = CUDA.@elapsed begin
            for _ in 1:ninner
                run!()
            end
        end
        t_loop = min(t_loop, t / ninner)
    end
    return (; t_bt, t_loop, nsamples = length(trial.times))
end

problem_size = (Nv, Nq, Nq, 1, Nh)
N = prod(problem_size)
traffic_MiB = round(3 * N * sizeof(FT) / 1024^2, digits = 1)
println("\nProblem: helem = $helem (Nh = $Nh), zelem = $Nv, Nq = $Nq, N = $N, $FT")
println("Traffic per call (1R + 2W): $traffic_MiB MiB\n")

bm = Benchmark(; float_type = FT, device_name = CUDA.name(CUDA.device()))
times = Dict{String, Float64}()
order = filter(in(keys(runners)), ["copy_floor", "cuda_ref", "climacore", "cutile"])
for name in order
    (; t_bt, t_loop, nsamples) = time_contender!(runners[name], device)
    times[name] = t_bt
    disagreement = abs(t_bt - t_loop) / min(t_bt, t_loop)
    @printf(
        "  %-10s  BenchmarkTools min %10.2f µs | event-loop min %10.2f µs%s\n",
        name,
        t_bt * 1e6,
        t_loop * 1e6,
        disagreement > 0.1 ? "  (WARNING: >10% disagreement)" : "",
    )
    push_info(
        bm;
        kernel_time_s = t_bt,
        nreps = nsamples,
        caller = name,
        problem_size,
        n_reads_writes = 3,
    )
end

println()
tabulate_benchmark(bm)

if haskey(times, "climacore") && haskey(times, "cutile")
    @printf(
        "\nSpeedup cuTile vs ClimaCore: %.2fx  (ClimaCore %.2f µs, cuTile %.2f µs)\n",
        times["climacore"] / times["cutile"],
        times["climacore"] * 1e6,
        times["cutile"] * 1e6,
    )
end
if haskey(times, "copy_floor")
    for name in order
        name == "copy_floor" && continue
        times[name] < times["copy_floor"] && @printf(
            "WARNING: %s (%.2f µs) beat the copy floor (%.2f µs) — timing artifact?\n",
            name,
            times[name] * 1e6,
            times["copy_floor"] * 1e6,
        )
    end
end
