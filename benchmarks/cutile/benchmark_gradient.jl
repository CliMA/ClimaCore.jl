#=
1v1 GPU benchmark: ClimaCore's spectral-element Gradient vs a cuTile.jl
implementation of the same tensor contraction.

Contenders (all produce identical covariant components; correctness is gated
against the ClimaCore CPU result before any timing):
  - climacore:  `@. ∇f = grad(f)` — the production fused-slab kernel
  - cutile:     one batched GEMM per element (see gradient_kernels_cutile.jl)
  - cuda_ref:   hand-written flop-matched CUDA.jl kernel
  - copy_floor: one read and (n - 1) writes per point, zero flops. The
    gradient itself is always 1R+2W. Extra counts (6, 9, …) are store-heavy
    copies used as bandwidth references, not extra gradient math.

With no list flags, one process sweeps Float32 and Float64, h_elem = 30, 60,
90, and copy-floor traffic counts 3, 6, 9. A final table reports cuTile
speedup against ClimaCore at each float type and resolution.

Usage (GPU node; see README.md for environment setup):
    export CLIMACOMMS_DEVICE=CUDA
    julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_gradient.jl

Narrow one axis with a comma-separated list, or a single value:
    ... benchmark_gradient.jl --float-type Float64 --helem 30,60

Tiny-config correctness check:
    ... benchmark_gradient.jl --float-type Float64 --helem 2 --zelem 4 --n-reads-writes 3
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
import PrettyTables
using Printf: @printf

include(joinpath(@__DIR__, "gradient_kernels.jl"))
include(joinpath(@__DIR__, "..", "scripts", "benchmark_utils.jl"))

##### CLI

function getarg(flag)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return nothing
    i < length(ARGS) || error("missing value after $flag")
    return ARGS[i + 1]
end

function csv_arg(flag, default, parse_one)
    raw = getarg(flag)
    raw === nothing && return default
    return parse_one.(String.(split(raw, ",")))
end

const FLOAT_TYPES = Dict("Float64" => Float64, "Float32" => Float32)
const float_types = csv_arg("--float-type", [Float32, Float64], name -> begin
    haskey(FLOAT_TYPES, name) ||
        error("unknown --float-type $name; expected Float32 or Float64")
    return FLOAT_TYPES[name]
end)
function int_arg(flag, default)
    raw = getarg(flag)
    raw === nothing && return default
    return parse(Int, raw)
end

const helems = csv_arg("--helem", [30, 60, 90], s -> parse(Int, s))
const n_rws = csv_arg("--n-reads-writes", [3, 6, 9], s -> parse(Int, s))
const zelem = int_arg("--zelem", 63)
const Nq = int_arg("--nq", 4)
const tv = int_arg("--tv", 64) # cuTile tile size along v (power of two)
const contender_arg = something(getarg("--contenders"), "all")
const contenders =
    contender_arg == "all" ? ["copy_floor", "cuda_ref", "climacore", "cutile"] :
    String.(split(contender_arg, ","))

all(>=(2), n_rws) ||
    error("--n-reads-writes counts one read plus at least one write; got $n_rws")
copy_label(n_rw) = "1R+$(n_rw - 1)W"

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
function init_field!(f, ::Type{FT}) where {FT}
    space = axes(f)
    coords = Fields.coordinate_field(space)
    @. f =
        FT(2) +
        sind(coords.long) * cosd(coords.lat) * (1 + coords.z / FT(30e3))
    return f
end

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

function record_time!(bm, times, name, run!, device, problem_size, n_rw)
    (; t_bt, t_loop, nsamples) = time_contender!(run!, device)
    times[name] = t_bt
    disagreement = abs(t_bt - t_loop) / min(t_bt, t_loop)
    @printf(
        "  %-12s  BenchmarkTools min %10.2f µs | event-loop min %10.2f µs%s\n",
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
        n_reads_writes = n_rw,
    )
    return nothing
end

function release_device_memory()
    GC.gc()
    isdefined(CUDA, :reclaim) && CUDA.reclaim()
    return nothing
end

# One (float type, horizontal resolution) case. Gradient kernels always move
# 1 read + 2 writes. `n_rws` only changes the copy floor: count `n` is one
# read and `n - 1` writes of the same nodal value.
function run_case(::Type{FT}, helem, device) where {FT}
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

    # Re-evaluating sind/cosd on the GPU disagrees by a few ulps, and the
    # differentiation matrix turns that into an O(1e-4) Float32 gradient error.
    # memcpy, not a broadcast: a CPU Array is not a bitstype and cannot be
    # captured by the GPU broadcast kernel.
    f_cpu = init_field!(zeros(cpu_space), FT)
    f_gpu = zeros(gpu_space)
    copyto!(parent(Fields.field_values(f_gpu)), parent(Fields.field_values(f_cpu)))
    ∇f_cpu = @. grad(f_cpu)
    ∇f_gpu = @. grad(f_gpu)

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
    out_cutile = CUDA.zeros(FT, Nv, Nq, Nq, 2, Nh)
    F3 = reshape(pf, Nv, K, Nh)
    O3 = reshape(out_cutile, Nv, C, Nh)
    Wt = CUDA.CuArray(gradient_weight(D))

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

    # Rows of D sum to 0, so each derivative entry cancels O(1) products
    # Dᵢⱼ fⱼ. GPU FFMA and CPU mul/add round that cancellation differently:
    # with identical nodal values the Float32 peak-relative gap is ~5e-5.
    # atol covers that floor. A TF32 matmul is ~1e-3 of the peak and still fails.
    grad_scale = max(maximum(abs, p∇_oracle), eps(FT))
    rtol = FT == Float64 ? 1e-12 : 1e-4
    atol = FT == Float64 ? zero(FT) : 2e-4 * grad_scale
    println(
        "Correctness gate (rtol = $rtol, atol = $atol, vs ClimaCore CPU oracle):",
    )
    for name in filter(in(keys(runners)), ["climacore", "cuda_ref", "cutile"])
        runners[name]()
        CUDA.synchronize()
        result = Array(outputs[name]())
        ok = isapprox(result, p∇_oracle; rtol, atol)
        maxrel =
            maximum(abs.(result .- p∇_oracle)) /
            max(maximum(abs.(p∇_oracle)), eps(FT))
        @printf(
            "  %-12s %s (max rel-scale error %.3e)\n",
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

    problem_size = (Nv, Nq, Nq, 1, Nh)
    N = prod(problem_size)
    traffic_MiB = round(3 * N * sizeof(FT) / 1024^2, digits = 1)
    println(
        "\nProblem: helem = $helem (Nh = $Nh), zelem = $Nv, Nq = $Nq, N = $N, $FT",
    )
    println("Gradient traffic per call (1R + 2W): $traffic_MiB MiB")
    println("Copy-floor counts: ", join(copy_label.(n_rws), ", "), "\n")

    bm = Benchmark(; float_type = FT, device_name = CUDA.name(CUDA.device()))
    times = Dict{String, Float64}()
    if "copy_floor" in contenders && 3 in n_rws
        nwrites = 2
        out_floor = CUDA.zeros(FT, Nv, Nq, Nq, nwrites, Nh)
        record_time!(
            bm, times, "copy_floor",
            () -> launch_copy_floor!(out_floor, pf, Val(Nq), nwrites),
            device, problem_size, 3,
        )
        CUDA.unsafe_free!(out_floor)
    end
    for name in filter(in(keys(runners)), ["cuda_ref", "climacore", "cutile"])
        record_time!(bm, times, name, runners[name], device, problem_size, 3)
    end
    if "copy_floor" in contenders
        for n_rw in n_rws
            n_rw == 3 && continue
            nwrites = n_rw - 1
            out_floor = CUDA.zeros(FT, Nv, Nq, Nq, nwrites, Nh)
            record_time!(
                bm, times, "copy_$(copy_label(n_rw))",
                () -> launch_copy_floor!(out_floor, pf, Val(Nq), nwrites),
                device, problem_size, n_rw,
            )
            times["copy_rw$n_rw"] = times["copy_$(copy_label(n_rw))"]
            CUDA.unsafe_free!(out_floor)
        end
        3 in n_rws && (times["copy_rw3"] = times["copy_floor"])
    end

    println()
    tabulate_benchmark(bm)
    if haskey(times, "copy_floor")
        for name in ("cuda_ref", "climacore", "cutile")
            haskey(times, name) || continue
            times[name] < times["copy_floor"] && @printf(
                "WARNING: %s (%.2f µs) beat the 1R+2W copy floor (%.2f µs) — timing artifact?\n",
                name,
                times[name] * 1e6,
                times["copy_floor"] * 1e6,
            )
        end
    end

    CUDA.unsafe_free!(pf)
    CUDA.unsafe_free!(out_ref)
    CUDA.unsafe_free!(out_cutile)
    CUDA.unsafe_free!(Wt)
    CUDA.unsafe_free!(parent(Fields.field_values(∇f_gpu)))
    return (; FT, helem, Nh, times)
end

function print_speedup_summary(rows)
    isempty(rows) && return nothing
    header = ["float", "helem", "Nh", "climacore µs", "cutile µs", "speedup"]
    for n_rw in n_rws
        push!(header, "copy $(copy_label(n_rw)) µs")
    end
    data = Matrix{Any}(undef, length(rows), length(header))
    for (i, row) in pairs(rows)
        cc = get(row.times, "climacore", nothing)
        ct = get(row.times, "cutile", nothing)
        data[i, 1] = string(row.FT)
        data[i, 2] = row.helem
        data[i, 3] = row.Nh
        data[i, 4] = cc === nothing ? "—" : round(cc * 1e6; digits = 2)
        data[i, 5] = ct === nothing ? "—" : round(ct * 1e6; digits = 2)
        data[i, 6] =
            (cc === nothing || ct === nothing) ? "—" : round(cc / ct; digits = 2)
        for (j, n_rw) in pairs(n_rws)
            t = get(row.times, "copy_rw$n_rw", nothing)
            data[i, 6 + j] = t === nothing ? "—" : round(t * 1e6; digits = 2)
        end
    end
    println()
    PrettyTables.pretty_table(
        data;
        title = "cuTile speedup vs ClimaCore (speedup = t_climacore / t_cutile)",
        column_labels = header,
        alignment = :l,
    )
    return nothing
end

device = ClimaComms.CUDADevice()
println(
    "Sweep: float = ",
    join(string.(float_types), ", "),
    ", helem = ",
    join(helems, ", "),
    ", copy-floor counts = ",
    join(n_rws, ", "),
    " (gradient kernels stay at 1R+2W)",
)
rows = NamedTuple[]
for FT in float_types, helem in helems
    println("\n======== $FT  helem = $helem ========\n")
    push!(rows, run_case(FT, helem, device))
    release_device_memory()
end
print_speedup_summary(rows)
