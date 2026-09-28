#=
Production-shaped expressions, not the isolated gradient.

Two ClimaAtmos tendencies that contain the strong scalar gradient:

  pressure:  duₕ = C12(grad(p) / ρ + grad(K + Φ))
             horizontal_advection_tendency! in ClimaAtmos
  laplacian: ∇²χ = wdiv(grad(χ))
             the scalar hyperdiffusion atom (energy, tracers, TKE)

Contenders, timed as a whole:
  fused:   the expression as ClimaAtmos writes it, one broadcast
  split:   materialize each strong gradient with Operators.Gradient, then
           the pointwise / weak-divergence remainder
  cutile:  the same split, with each strong gradient replaced by the
           KronGEMM kernel from gradient_kernels_cutile.jl

Momentum hyperdiffusion, wgrad(div(u)) − wcurl(curl(u)), is not here. This
cuTile kernel only implements the strong scalar gradient.

`weighted_dss!` between ∇² and ∇⁴ passes is not timed. Every contender needs
the same exchange afterward.

Usage (GPU node; see README.md):
    export CLIMACOMMS_DEVICE=CUDA
    julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_fused.jl

    ... benchmark_fused.jl --float-type Float64 --helem 30,60 --zelem 63
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
include(joinpath(@__DIR__, "gradient_kernels_cutile.jl"))

const C12 = Geometry.Covariant12Vector
const grad = Operators.Gradient()
const wdiv = Operators.Divergence{Operators.WeakForm}()

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

function int_arg(flag, default)
    raw = getarg(flag)
    raw === nothing && return default
    return parse(Int, raw)
end

const FLOAT_TYPES = Dict("Float64" => Float64, "Float32" => Float32)
const float_types = csv_arg("--float-type", [Float32, Float64], name -> begin
    haskey(FLOAT_TYPES, name) ||
        error("unknown --float-type $name; expected Float32 or Float64")
    return FLOAT_TYPES[name]
end)
const helems = csv_arg("--helem", [30], s -> parse(Int, s))
const zelem = int_arg("--zelem", 63)
const Nq = int_arg("--nq", 4)
const tv = int_arg("--tv", 64)

##### Preflight

VERSION >= v"1.11" ||
    error("cuTile.jl requires Julia >= 1.11 (this is $VERSION); use `julia +1.11`.")
CUDA.functional() || error(
    "CUDA is not functional on this node. Run on a GPU node with CLIMACOMMS_DEVICE=CUDA.",
)
CUDA.versioninfo()
println()
drv = CUDA.driver_version()
drv >= v"13" || error(
    "cuTile requires an NVIDIA driver supporting CUDA 13 (driver >= 580); " *
    "this node reports CUDA $drv.",
)
cap = CUDA.capability(CUDA.device())
cap >= v"8.0" || error(
    "cuTile requires compute capability >= 8.0 (Ampere+); " *
    "this GPU ($(CUDA.name(CUDA.device()))) has $cap.",
)
ispow2(Nq * Nq) || error(
    "the cuTile gradient requires power-of-two tile extents; Nq² = $(Nq * Nq).",
)

##### Setup

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

function scalar_state(space, ::Type{FT}) where {FT}
    coords = Fields.coordinate_field(space)
    p = zeros(space)
    ρ = zeros(space)
    K = zeros(space)
    Φ = zeros(space)
    χ = zeros(space)
    @. p = FT(1e5) * (FT(1) + FT(0.1) * sind(coords.long) * cosd(coords.lat))
    @. ρ = FT(1) + FT(0.05) * cosd(coords.lat)
    @. K = FT(100) * sind(coords.long) * sind(coords.long)
    @. Φ = FT(9.81) * coords.z
    @. χ =
        sind(coords.long) * cosd(coords.lat) * (FT(1) + coords.z / FT(30e3))
    return (; p, ρ, K, Φ, χ)
end

function copy_nodal!(dest, src)
    copyto!(parent(Fields.field_values(dest)), parent(Fields.field_values(src)))
    return dest
end

function to_device(cpu_field, gpu_space)
    return copy_nodal!(zeros(gpu_space), cpu_field)
end

# Strong scalar gradient via the KronGEMM kernel. `dest` is the
# Covariant12Vector field, `src` the scalar field.
function launch_scalar_grad!(dest, src, Wt; tv, K, C)
    pf = parent(Fields.field_values(src))
    po = parent(Fields.field_values(dest))
    (Nv, nqi, nqj, Fs, Nh) = size(pf)
    (_, _, _, Fd, _) = size(po)
    (nqi == Nq && nqj == Nq && Fs == 1 && Fd == 2) || error(
        "expected VIJFH parents (Nv, Nq, Nq, 1, Nh) and (Nv, Nq, Nq, 2, Nh); " *
        "got $(size(pf)) and $(size(po))",
    )
    launch_grad_cutile!(
        reshape(pf, Nv, K, Nh),
        Wt,
        reshape(po, Nv, C, Nh);
        tv,
        K,
        C,
    )
    return nothing
end

function time_kernel!(run!; ntrials = 100, ninner = 10)
    run!()
    CUDA.synchronize()
    device = ClimaComms.CUDADevice()
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
    return (; t_bt, t_loop)
end

function check_against_oracle!(name, result, oracle, FT)
    rtol = FT == Float64 ? 1e-12 : 1e-4
    scale = max(maximum(abs, oracle), eps(FT))
    atol = FT == Float64 ? zero(FT) : 2e-4 * scale
    ok = isapprox(result, oracle; rtol, atol)
    maxrel = maximum(abs.(result .- oracle)) / scale
    @printf(
        "  %-16s %s (max rel-scale error %.3e)\n",
        name,
        ok ? "PASS" : "FAIL",
        maxrel,
    )
    ok || error("$name does not match the CPU fused oracle")
    return nothing
end

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
    cpu = scalar_state(cpu_space, FT)
    p = to_device(cpu.p, gpu_space)
    ρ = to_device(cpu.ρ, gpu_space)
    K = to_device(cpu.K, gpu_space)
    Φ = to_device(cpu.Φ, gpu_space)
    χ = to_device(cpu.χ, gpu_space)

    du_cpu = @. C12(grad(cpu.p) / cpu.ρ + grad(cpu.K + cpu.Φ))
    lap_cpu = @. wdiv(grad(cpu.χ))
    p_du = parent(Fields.field_values(du_cpu))
    p_lap = parent(Fields.field_values(lap_cpu))

    bernoulli = zeros(gpu_space)
    ∇p = @. grad(p)
    ∇b = similar(∇p)
    gχ = similar(∇p)
    du_fused = similar(∇p)
    du_split = similar(∇p)
    du_cutile = similar(∇p)
    lap_fused = similar(χ)
    lap_split = similar(χ)
    lap_cutile = similar(χ)

    D = Quadratures.differentiation_matrix(FT, Quadratures.GLL{Nq}())
    Kq = Nq * Nq
    C = 2 * Kq
    Wt = CUDA.CuArray(gradient_weight(D))
    launch! = (dest, src) -> launch_scalar_grad!(dest, src, Wt; tv, K = Kq, C)

    runners = Dict(
        "pressure_fused" => () -> begin
            @. du_fused = C12(grad(p) / ρ + grad(K + Φ))
            nothing
        end,
        "pressure_split" => () -> begin
            @. bernoulli = K + Φ
            @. ∇p = grad(p)
            @. ∇b = grad(bernoulli)
            @. du_split = ∇p / ρ + ∇b
            nothing
        end,
        "pressure_cutile" => () -> begin
            @. bernoulli = K + Φ
            launch!(∇p, p)
            launch!(∇b, bernoulli)
            @. du_cutile = ∇p / ρ + ∇b
            nothing
        end,
        "lap_fused" => () -> begin
            @. lap_fused = wdiv(grad(χ))
            nothing
        end,
        "lap_split" => () -> begin
            @. gχ = grad(χ)
            @. lap_split = wdiv(gχ)
            nothing
        end,
        "lap_cutile" => () -> begin
            launch!(gχ, χ)
            @. lap_cutile = wdiv(gχ)
            nothing
        end,
    )
    outputs = Dict(
        "pressure_fused" => () -> parent(Fields.field_values(du_fused)),
        "pressure_split" => () -> parent(Fields.field_values(du_split)),
        "pressure_cutile" => () -> parent(Fields.field_values(du_cutile)),
        "lap_fused" => () -> parent(Fields.field_values(lap_fused)),
        "lap_split" => () -> parent(Fields.field_values(lap_split)),
        "lap_cutile" => () -> parent(Fields.field_values(lap_cutile)),
    )
    oracles = Dict(
        "pressure_fused" => p_du,
        "pressure_split" => p_du,
        "pressure_cutile" => p_du,
        "lap_fused" => p_lap,
        "lap_split" => p_lap,
        "lap_cutile" => p_lap,
    )
    order = [
        "pressure_fused",
        "pressure_split",
        "pressure_cutile",
        "lap_fused",
        "lap_split",
        "lap_cutile",
    ]

    Nh = size(parent(Fields.field_values(p)), 5)
    println("Correctness gate vs CPU fused expression:")
    for name in order
        runners[name]()
        CUDA.synchronize()
        check_against_oracle!(name, Array(outputs[name]()), oracles[name], FT)
    end

    println("\nTiming:")
    times = Dict{String, Float64}()
    for name in order
        (; t_bt, t_loop) = time_kernel!(runners[name])
        times[name] = t_bt
        disagreement = abs(t_bt - t_loop) / min(t_bt, t_loop)
        @printf(
            "  %-16s  BenchmarkTools min %10.2f µs | event-loop min %10.2f µs%s\n",
            name,
            t_bt * 1e6,
            t_loop * 1e6,
            disagreement > 0.1 ? "  (WARNING: >10% disagreement)" : "",
        )
    end
    @printf(
        "\npressure   fused %8.2f µs | split %8.2f µs (split/fused %.2f) | cutile %8.2f µs (fused/cutile %.2f)\n",
        times["pressure_fused"] * 1e6,
        times["pressure_split"] * 1e6,
        times["pressure_split"] / times["pressure_fused"],
        times["pressure_cutile"] * 1e6,
        times["pressure_fused"] / times["pressure_cutile"],
    )
    @printf(
        "laplacian  fused %8.2f µs | split %8.2f µs (split/fused %.2f) | cutile %8.2f µs (fused/cutile %.2f)\n",
        times["lap_fused"] * 1e6,
        times["lap_split"] * 1e6,
        times["lap_split"] / times["lap_fused"],
        times["lap_cutile"] * 1e6,
        times["lap_fused"] / times["lap_cutile"],
    )
    return (; FT, helem, Nh, times)
end

function µs(times, name)
    return round(times[name] * 1e6; digits = 2)
end

function print_summary(rows)
    isempty(rows) && return nothing
    header = [
        "float",
        "helem",
        "Nh",
        "p fused µs",
        "p split µs",
        "p cutile µs",
        "p speedup",
        "lap fused µs",
        "lap split µs",
        "lap cutile µs",
        "lap speedup",
    ]
    data = Matrix{Any}(undef, length(rows), length(header))
    for (i, row) in pairs(rows)
        t = row.times
        data[i, 1] = string(row.FT)
        data[i, 2] = row.helem
        data[i, 3] = row.Nh
        data[i, 4] = µs(t, "pressure_fused")
        data[i, 5] = µs(t, "pressure_split")
        data[i, 6] = µs(t, "pressure_cutile")
        data[i, 7] = round(t["pressure_fused"] / t["pressure_cutile"]; digits = 2)
        data[i, 8] = µs(t, "lap_fused")
        data[i, 9] = µs(t, "lap_split")
        data[i, 10] = µs(t, "lap_cutile")
        data[i, 11] = round(t["lap_fused"] / t["lap_cutile"]; digits = 2)
    end
    println()
    PrettyTables.pretty_table(
        data;
        title = "speedup = t_fused / t_cutile  (>1 means the cuTile split is faster than the fused ClimaCore expression)",
        column_labels = header,
        alignment = :l,
    )
    return nothing
end

device = ClimaComms.CUDADevice()
println(
    "Cases: float = ",
    join(string.(float_types), ", "),
    ", helem = ",
    join(helems, ", "),
    ", zelem = $zelem",
)
rows = NamedTuple[]
for FT in float_types, helem in helems
    println("\n======== $FT  helem = $helem ========\n")
    push!(rows, run_case(FT, helem, device))
    GC.gc()
    isdefined(CUDA, :reclaim) && CUDA.reclaim()
end
print_summary(rows)
