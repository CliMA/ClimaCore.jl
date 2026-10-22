#! format: off
using Test
using ClimaComms
ClimaComms.@import_required_backends
using StaticArrays, IntervalSets, LinearAlgebra
import BenchmarkTools
import StatsBase
import DataStructures
using ClimaCore.Geometry: ⊗
import ClimaCore.DataLayouts

import ClimaCore
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

@show ClimaComms.device() isa ClimaComms.CUDADevice
if ClimaComms.device() isa ClimaComms.CUDADevice
    import CUDA
    device_name = CUDA.name(CUDA.device()) # Move to ClimaComms
    include("benchmark_stencils_device_time.jl")
else
    device_name = "CPU"
end

include(joinpath(pkgdir(ClimaCore), "benchmarks/scripts/benchmark_utils.jl"))

# Benchmark `f()`, returning a `BenchmarkTools.Trial` on CPU and the mean
# device time (ns) on GPU, so that GPU timings exclude host-side overhead.
benchmark_trial(f, device) =
    device isa ClimaComms.CUDADevice ? device_time_ns(f) : BenchmarkTools.@benchmark $f()
# The benchmark time (ns): the minimum wall time on CPU, and the mean device
# time on GPU.
trial_time(trial::BenchmarkTools.Trial) = minimum(trial.times)
trial_time(time_ns::Real) = time_ns
show_trial(trial::BenchmarkTools.Trial) = show(stdout, MIME("text/plain"), trial)
show_trial(time_ns::Real) = println("Mean device time: $(round(time_ns; digits = 1)) ns")

import ClimaCore: Domains, Meshes, Spaces, Fields, Operators, Topologies
import ClimaCore.Domains: Geometry

field_vars(::Type{FT}) where {FT} = (;
    x = FT(0),
    uₕ = Geometry.Covariant12Vector(FT(0), FT(0)),
    uₕ2 = Geometry.Covariant12Vector(FT(0), FT(0)),
    curluₕ = Geometry.Contravariant12Vector(FT(0), FT(0)),
    w = Geometry.Covariant3Vector(FT(0)),
    contra3 = Geometry.Contravariant3Vector(FT(0)),
    y = FT(0),
    D = FT(0),
    U = FT(0),
    ∇x = Geometry.Covariant3Vector(FT(0)),
    ᶠu³ = Geometry.Contravariant3Vector(FT(0)),
    ᶠuₕ³ = Geometry.Contravariant3Vector(FT(0)),
    ᶠw = Geometry.Covariant3Vector(FT(0)),
    s1 = FT(0),
    s2 = FT(0),
    s3 = FT(0),
    s4 = FT(0),
)

function set_value_bcs(c)
    FT = Spaces.undertype(axes(c))
    return (;bottom = Operators.SetValue(FT(0)),
             top = Operators.SetValue(FT(0)))
end

function extrapolate_bcs(c)
    return (;bottom = Operators.Extrapolate(),
             top = Operators.Extrapolate())
end

function set_value_contra3_bcs(c)
    FT = Spaces.undertype(axes(c))
    contra3 = Geometry.Contravariant3Vector
    return (;bottom = Operators.SetValue(contra3(FT(0.0))),
             top = Operators.SetValue(contra3(FT(0.0))))
end

function set_value_divgrad_uₕ_maybe_field_bcs(c) # real-world example
    FT = Spaces.undertype(axes(c))
    top_val = Geometry.Contravariant3Vector(FT(0)) ⊗
            Geometry.Covariant12Vector(FT(0), FT(0))
    if hasproperty(axes(c), :horizontal_space)
        z_bottom = Spaces.level(Fields.coordinate_field(c).z, 1)
        bottom_val =
            Geometry.Contravariant3Vector.(zeros(axes(z_bottom))) .⊗
            Geometry.Covariant12Vector.(
                zeros(axes(z_bottom)),
                zeros(axes(z_bottom)),
            )
        return (;top = Operators.SetValue(top_val),
                 bottom = Operators.SetValue(.-bottom_val))
    else
        return (;top = Operators.SetValue(top_val),
                 bottom = Operators.SetValue(top_val))
    end
end

function set_top_value_bc(c)
    FT = Spaces.undertype(axes(c))
    return (;top = Operators.SetValue(FT(0)))
end

function set_bot_value_bc(c)
    FT = Spaces.undertype(axes(c))
    return (;bottom = Operators.SetValue(FT(0)))
end

function set_divergence_bcs(c)
    FT = Spaces.undertype(axes(c))
    return (;bottom = Operators.SetDivergence(FT(0)),
             top = Operators.SetDivergence(FT(0)))
end

function set_gradient_value_bcs(c)
    FT = Spaces.undertype(axes(c))
    wvec = Geometry.WVector
    return (;bottom = Operators.SetGradient(wvec(FT(0))),
             top = Operators.SetGradient(wvec(FT(0))))
end

function set_curl_bcs(c)
    FT = Spaces.undertype(axes(c))
    contra12 = Geometry.Contravariant12Vector
    return (;bottom = Operators.SetCurl(contra12(FT(0), FT(0))),
             top = Operators.SetCurl(contra12(FT(0), FT(0))))
end

function bc_name(bcs::NamedTuple)
    if haskey(bcs, :inner) && haskey(bcs, :outer)
        return (bc_name_base(bcs.inner)..., bc_name_base(bcs.outer)...)
    else
        return bc_name_base(bcs)
    end
end
bc_name_base(bcs::NamedTuple) = (
    (haskey(bcs, :top) ? (nameof(typeof(bcs.top)),) : ())...,
    (haskey(bcs, :bottom) ? (nameof(typeof(bcs.bottom)), ) : ())...,
)
bc_name_base(bcs::Tuple) = (:none,)
bc_name(bcs::Tuple) = (:none,)
bc_name_base(bcs::@NamedTuple{}) = (:none,)
bc_name(bcs::@NamedTuple{}) = (:none,)

include("benchmark_stencils_array_kernels.jl")
include("benchmark_stencils_climacore_kernels.jl")

uses_bycolumn(::typeof(op_broadcast_example0!)) = true
uses_bycolumn(::typeof(op_broadcast_example1!)) = true
uses_bycolumn(::typeof(op_broadcast_example2!)) = false
uses_bycolumn(::Any) = false

bcs_tested(c, ::typeof(op_broadcast_example0!)) = ((;), )
bcs_tested(c, ::typeof(op_broadcast_example1!)) = ((;), )
bcs_tested(c, ::typeof(op_broadcast_example2!)) = ((;), )

bcs_tested(c, ::typeof(op_GradientF2C!)) = ((;), set_value_bcs(c))
bcs_tested(c, ::typeof(op_GradientC2F!)) = (set_gradient_value_bcs(c),)
bcs_tested(c, ::typeof(op_DivergenceF2C!)) =
    ((;), set_value_contra3_bcs(c), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_DivergenceC2F!)) = (set_divergence_bcs(c), )
bcs_tested(c, ::typeof(op_InterpolateF2C!)) = ((;), )
bcs_tested(c, ::typeof(op_InterpolateC2F!)) = (set_value_bcs(c), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_BottomBiasedC2F!)) = (set_bot_value_bc(c),)
bcs_tested(c, ::typeof(op_BottomBiasedF2C!)) = ((;), set_bot_value_bc(c))
bcs_tested(c, ::typeof(op_TopBiasedC2F!)) = (set_top_value_bc(c),)
bcs_tested(c, ::typeof(op_TopBiasedF2C!)) = ((;), set_top_value_bc(c))
bcs_tested(c, ::typeof(op_CurlC2F!)) = (set_curl_bcs(c),)
# The advection operators are benchmarked with no boundary conditions, i.e. with
# their default `Extrapolate{0}` ghost-point extrapolation.
bcs_tested(c, ::typeof(op_UpwindBiasedProductC2F!)) = ((;),)
bcs_tested(c, ::typeof(op_Upwind3rdOrderBiasedProductC2F!)) = ((;),)

# Composed operators (bcs handled case-by-case)
bcs_tested(c, ::typeof(op_divUpwind3rdOrderBiasedProductC2F!)) =
    ((; inner = (;), outer = set_value_contra3_bcs(c)), )
# The Laplacian of a center field (the most common composed vertical operator
# downstream); the inner gradient carries the SetGradient boundary conditions
# used downstream.
bcs_tested(c, ::typeof(op_divgrad_CC!)) =
    ((; inner = set_gradient_value_bcs(c), outer = (;)), )
bcs_tested(c, ::typeof(op_divgrad_FF!)) =
    ((; inner = (;), outer = set_divergence_bcs(c)), )
bcs_tested(c, ::typeof(op_div_interp_CC!)) =
    ((; inner = set_value_contra3_bcs(c), outer = (;)), )
bcs_tested(c, ::typeof(op_div_interp_FF!)) =
    ((; inner = (;), outer = set_divergence_bcs(c)), )
bcs_tested(c, ::typeof(op_divgrad_uₕ!)) =
    ((; inner = (;), outer = set_value_divgrad_uₕ_maybe_field_bcs(c)),)
# Nested and fused expressions fix their boundary conditions internally.
bcs_tested(c, ::typeof(op_nest_interp_4!)) = ((;),)
bcs_tested(c, ::typeof(op_nest_interp_8!)) = ((;),)
bcs_tested(c, ::typeof(op_nest_costly_8!)) = ((;),)
bcs_tested(c, ::typeof(op_biharmonic!)) = ((;),)
bcs_tested(c, ::typeof(op_wide_sum_8!)) = ((;),)
bcs_tested(c, ::typeof(op_diffusion_like!)) = ((;),)
bcs_tested(c, ::typeof(op_advection_like!)) = ((;),)
bcs_tested(c, ::typeof(op_atmos_like!)) = ((;),)

function short_name(key)
    to_short = (
        "op_divUpwind3rdOrderBiasedProductC2F" => "op_divO3UBPC2F",
        "op_UpwindBiasedProductC2F" => "op_UBPC2F",
    )
    replace(string(key), to_short...)
end
function benchmark_func!(bm, t_min, trials, fun, c, f, verbose = false; compile::Bool, baselines = nothing)
    device = ClimaComms.device(c)
    all_bcs = bcs_tested(c, fun)
    for (ibc,bcs) in enumerate(bcs_tested(c, fun))
        key = (fun, bc_name(bcs)...)
        baseline_time_s = isnothing(baselines) ? nothing : get(baselines, key, nothing)
        if compile
            fun(c, f, bcs)
        else
            verbose && @info "\n@benchmarking $key"
            trials[key] = benchmark_trial(() -> fun(c, f, bcs), device)
            cf = Fields.field_values(Fields.coordinate_field(axes(c)))
            kernel_time_s = trial_time(trials[key]) * 1e-9 # to seconds
            push_info(bm;
                kernel_time_s,
                nreps=1,
                caller=short_name(key),
                n_reads_writes=n_reads_writes(typeof(fun)),
                problem_size=size(cf),
                baseline_time_s,
            )
        end
        if haskey(trials, key)
            verbose && show_trial(trials[key])

            t_min[key] = trial_time(trials[key]) # nano seconds
            if !verbose
                ibc == 1 && print("Benchmarking $fun: ")
                print("($ibc/$(length(all_bcs)))")
                is_last = (ibc == length(all_bcs) || length(all_bcs) == 0)
                is_last || print(", ")
                is_last && println()
            end
        end
    end
end

function column_benchmark_arrays(device, z_elems, ::Type{FT}; compile::Bool = false) where {FT}
    ArrayType = ClimaComms.array_type(device)
    L = ArrayType(zeros(FT, z_elems))
    D = ArrayType(zeros(FT, z_elems))
    U = ArrayType(zeros(FT, z_elems))
    xarr = ArrayType(rand(FT, z_elems))
    uₕ_x = ArrayType(rand(FT, z_elems))
    uₕ_y = ArrayType(rand(FT, z_elems))
    yarr = ArrayType(rand(FT, z_elems + 1))
    if compile
        if device isa ClimaComms.CUDADevice
            column_op_2mul_1add_cuda!(xarr, yarr, D, U)
        else
            column_op_2mul_1add!(xarr, yarr, D, U)
            column_op_3mul_2add!(xarr, yarr, L, D, U)
            column_curl_like!(xarr, uₕ_x, uₕ_y, D, U)
        end
        return nothing
    end

    if device isa ClimaComms.CUDADevice
        println("\n############################ column 2-point stencil")
        show_trial(benchmark_trial(() -> column_op_2mul_1add_cuda!(xarr, yarr, D, U), device))
        println()
    else
        println("\n############################ column 2-point stencil")
        trial = BenchmarkTools.@benchmark column_op_2mul_1add!($xarr, $yarr, $D, $U)
        show(stdout, MIME("text/plain"), trial)
        println()
        println("\n############################ column 3-point stencil")
        trial = BenchmarkTools.@benchmark column_op_3mul_2add!($xarr, $yarr, $L, $D, $U)
        show(stdout, MIME("text/plain"), trial)
        println()
        println("\n############################ column curl-like stencil")
        trial = BenchmarkTools.@benchmark column_curl_like!($xarr, $uₕ_x, $uₕ_y, $D, $U)
        show(stdout, MIME("text/plain"), trial)
        println()
    end
end

function sphere_benchmark_arrays(device, z_elems, helem, Nq, ::Type{FT}; compile::Bool = false) where {FT}
    ArrayType = ClimaComms.array_type(device)
    # VIJFH
    Nh = helem * helem * 6
    cdims = (z_elems  , Nq, Nq, 1, Nh)
    fdims = (z_elems+1, Nq, Nq, 1, Nh)
    D = ArrayType(zeros(FT, cdims...))
    U = ArrayType(zeros(FT, cdims...))
    xarr = ArrayType(rand(FT, cdims...))
    yarr = ArrayType(rand(FT, fdims...))

    if device isa ClimaComms.CUDADevice
        if compile
            sphere_op_2mul_1add_cuda!(xarr, yarr, D, U)
        else
            println("\n############################ sphere 2-point stencil")
            show_trial(benchmark_trial(() -> sphere_op_2mul_1add_cuda!(xarr, yarr, D, U), device))
            println()
        end
    else
        @info "Sphere CPU kernels have not been added yet."
    end
end

function benchmark_operators_column(bm; z_elems, helem, Nq, compile::Bool = false)
    FT = bm.float_type
    device = ClimaComms.device()
    @show device
    trials = DataStructures.OrderedDict()
    t_min = DataStructures.OrderedDict()

    cspace = TU.ColumnCenterFiniteDifferenceSpace(FT; zelem=z_elems)
    fspace = Spaces.FaceFiniteDifferenceSpace(cspace)
    cfield = fill(field_vars(FT), cspace)
    ffield = fill(field_vars(FT), fspace)
    baselines = baseline_lookup(reference_times_column(device))
    benchmark_operators_base(bm, trials, t_min, cfield, ffield, "column"; compile, baselines)

    # The regression checks live in `test_results_column`; the table shows
    # the percent change from the reference timings when they exist.
    tabulate_benchmark(bm)
    return (; bm, trials, t_min)
end

function benchmark_operators_sphere(bm; z_elems, helem, Nq, compile::Bool = false, VIJH)
    FT = bm.float_type
    device = ClimaComms.device()
    @show device
    trials = DataStructures.OrderedDict()
    t_min = DataStructures.OrderedDict()

    cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem=z_elems, helem, Nq, VIJH)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    cfield = fill(field_vars(FT), cspace)
    ffield = fill(field_vars(FT), fspace)
    baselines = baseline_lookup(reference_times_sphere(device; VIJH, float_type = FT))
    benchmark_operators_base(bm, trials, t_min, cfield, ffield, "sphere"; compile, baselines)

    # The regression checks live in `test_results_sphere`; the table shows
    # the percent change from the reference timings when they exist.
    tabulate_benchmark(bm)
    return (; trials, t_min)
end

function benchmark_operators_base(bm, trials, t_min, cfield, ffield, name; compile::Bool, baselines = nothing)
    ops = [
        #### Core discrete operators
        op_GradientF2C!,
        op_GradientC2F!,
        op_DivergenceF2C!,
        op_DivergenceC2F!,
        op_InterpolateF2C!,
        op_InterpolateC2F!,
        op_broadcast_example0!,
        op_broadcast_example1!,
        op_broadcast_example2!,
        op_BottomBiasedC2F!,
        op_BottomBiasedF2C!,
        op_TopBiasedC2F!,
        op_TopBiasedF2C!,
        op_CurlC2F!,
        #### Mixed / adaptive
        op_UpwindBiasedProductC2F!, # TODO: do we need to test this for different w values?
        # TODO: This is throwing an index error (bad/incompatible BCs)
        #       Is it possible to unit test this?
        # op_Upwind3rdOrderBiasedProductC2F!, # TODO: do we need to test this for different w values?
        #### Composed
        op_divUpwind3rdOrderBiasedProductC2F!,
        op_divgrad_CC!,
        op_divgrad_FF!,
        op_div_interp_CC!,
        op_div_interp_FF!,
        op_divgrad_uₕ!,
        #### Nested and fused expressions
        op_nest_interp_4!,
        op_nest_interp_8!,
        op_nest_costly_8!,
        op_biharmonic!,
        op_wide_sum_8!,
        op_diffusion_like!,
        op_advection_like!,
        op_atmos_like!,
    ]

    @info "Benchmarking $name operators, this may take a minute or two..."
    for op in ops
        if uses_bycolumn(op) && axes(cfield) isa Spaces.FiniteDifferenceSpace
            continue
        end
        benchmark_func!(bm, t_min, trials, op, cfield, ffield, #= verbose = =# false; compile, baselines)
    end

    return nothing
end

const GPU_BUFFER = 1.15
const CPU_BUFFER = 2

# The reference tables hold, per device:
# - A100: mean device (kernel) times from `device_time_ns`, which is what
#   `benchmark_func!` measures on GPU, as the median over 5 runs. Most entries
#   vary by about 1% between runs, but some kernels (e.g., `op_nest_interp_8!`)
#   occasionally run up to ~20% slower for a whole run;
# - P100: mean device times from `device_time_ns`, from a single CI run
#   (climacore-ci build 8048);
# - CPU: minimum wall times of `@benchmark`.
#
# Reference timings are keyed by GPU model; CI pins the hardware with a `gres`
# entry in .buildkite/pipeline.yml. Returns `nothing` for GPUs without
# reference timings.
function gpu_model(device_name)
    occursin("P100", device_name) && return :P100
    occursin("A100", device_name) && return :A100
    return nothing
end

# Reference tables map a benchmark key to the measured minimum time (ns) on
# the reference hardware. Returns a `Dict` of key => time in seconds for the
# "% vs baseline" column of `tabulate_benchmark`, or `nothing`.
baseline_lookup(::Nothing) = nothing
baseline_lookup(results) = Dict(key => ref_time * 1e-9 for (key, ref_time) in results)

# Select the `dev` column (e.g. `:P100`, `:A100`, `:CPU`) out of a reference
# table of `[key, (; P100, A100, CPU)]` rows with times in μs. Rows without a
# time for `dev` are skipped. Returns `[key, time_ns]` pairs, or `nothing` if
# there are none.
function select_device(table, dev)
    isnothing(dev) && return nothing
    μs = 10^3
    results = [[key, getproperty(times, dev) * μs] for (key, times) in table if haskey(times, dev)]
    return isempty(results) ? nothing : results
end

# Regressions fail the test on GPU (where CI pins the hardware) and only warn
# on CPU. `buffer` is the tolerance factor applied to the reference times.
function check_regressions(t_min, results, device, buffer)
    if isnothing(results)
        @warn "No reference timings for $device_name; skipping regression checks."
        return nothing
    end
    for (params, ref_time) in results
        passed = t_min[params] ≤ ref_time * buffer
        if !passed
            @warn "Possible regression: $params, time=$(t_min[params]), ref_time=$ref_time, buffer=$buffer"
        end
        if device isa ClimaComms.CUDADevice
            @test passed
        end
    end
end

function test_results_column(
    t_min,
    device = ClimaComms.device();
    gpu = gpu_model(device_name),
)
    # If these tests fail, just update the numbers (or the
    # buffer) so long its not an egregious regression.
    buffer = device isa ClimaComms.CUDADevice ? GPU_BUFFER : CPU_BUFFER
    results = reference_times_column(device; gpu)
    check_regressions(t_min, results, device, buffer)
end

# Reference times (ns) for the column benchmarks, or `nothing` when there are
# no reference timings for this device.
function reference_times_column(
    device = ClimaComms.device();
    gpu = gpu_model(device_name),
)
    # Reference times in μs per device; rows without an entry for a device have
    # no reference on it. See the note above `gpu_model`
    # for what each column holds.
    table = [
        [(op_GradientF2C!, :none), (P100 = 1.866, A100 = 2.155, CPU = 0.2531)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 1.847, A100 = 2.078, CPU = 0.270448)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 2.374, A100 = 2.629, CPU = 0.242053)],
        [(op_DivergenceF2C!, :none), (P100 = 3.542, A100 = 3.4, CPU = 1.005)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 3.628, A100 = 3.614, CPU = 1.076)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 3.549, A100 = 3.39, CPU = 1.076)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 3.626, A100 = 3.595, CPU = 0.878028)],
        [(op_InterpolateF2C!, :none), (P100 = 1.832, A100 = 2.134, CPU = 0.254523)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 2.115, A100 = 2.37, CPU = 0.254241)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 2.014, A100 = 2.119, CPU = 0.241308)],
        [(op_broadcast_example2!, :none), (P100 = 2.047, A100 = 2.757, CPU = 0.555039)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 2.05, A100 = 2.09, CPU = 0.207264)],
        [(op_BottomBiasedF2C!, :none), (P100 = 1.812, A100 = 2.131, CPU = 0.137031)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 2.065, A100 = 2.32, CPU = 0.185135)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 1.827, A100 = 2.093, CPU = 0.129971)],
        [(op_TopBiasedF2C!, :none), (P100 = 1.829, A100 = 2.136, CPU = 0.14212)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 1.901, A100 = 2.175, CPU = 0.141446)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 2.51, A100 = 2.66, CPU = 1.692)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 2.157, A100 = 2.543, CPU = 0.765401)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 4.837, A100 = 4.786, CPU = 2.54)],
        # The CPU reference times of entries whose boundary conditions changed in
        # the operator refactor (op_DivergenceF2C! above, op_divgrad_CC!,
        # op_div_interp_FF!) were carried over from the closest pre-refactor
        # entries rather than re-measured; they are within-buffer tripwires, not
        # measurements.
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 3.403, A100 = 3.404, CPU = 0.924147)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 3.17, A100 = 3.168, CPU = 0.87651)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 2.904, A100 = 3.037, CPU = 0.721119)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 2.939, A100 = 3.024, CPU = 0.686581)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 3.378, A100 = 3.362, CPU = 5.047)],
        [(op_nest_interp_4!, :none), (P100 = 2.914, A100 = 2.566)],
        [(op_nest_interp_8!, :none), (P100 = 3.962, A100 = 2.943)],
        [(op_nest_costly_8!, :none), (P100 = 8.344, A100 = 6.643)],
        [(op_biharmonic!, :none), (P100 = 4.769, A100 = 4.11)],
        [(op_wide_sum_8!, :none), (P100 = 3.324, A100 = 3.764)],
        [(op_diffusion_like!, :none), (P100 = 3.592, A100 = 3.393)],
        [(op_advection_like!, :none), (P100 = 3.317, A100 = 3.212)],
        [(op_atmos_like!, :none), (P100 = 8.063, A100 = 7.396)],
    ]
    return select_device(table, device isa ClimaComms.CUDADevice ? gpu : :CPU)
end

function test_results_sphere(
    t_min,
    device = ClimaComms.device();
    VIJH = DataLayouts.VIJFH,
    float_type = Float64,
    gpu = gpu_model(device_name),
)
    # If these tests fail, just update the numbers (or the
    # buffer) so long its not an egregious regression.
    buffer = device isa ClimaComms.CUDADevice ? GPU_BUFFER : CPU_BUFFER
    results = reference_times_sphere(device; VIJH, float_type, gpu)
    check_regressions(t_min, results, device, buffer)
end

# Reference times (ns) for the sphere benchmarks, or `nothing` when there are
# no reference timings for this device / layout / float type.
function reference_times_sphere(
    device = ClimaComms.device();
    VIJH = DataLayouts.VIJFH,
    float_type = Float64,
    gpu = gpu_model(device_name),
)
    # Reference times in μs per device; rows without an entry for a device have
    # no reference on it. See the note above `gpu_model`
    # for what each column holds.
    # The CPU times are layout-agnostic tripwires (`CPU_BUFFER`) and are stored
    # once, under VIJFH.
    dev = device isa ClimaComms.CUDADevice ? gpu : :CPU
    dev === :CPU && (VIJH = DataLayouts.VIJFH)
    tables = Dict(
        (DataLayouts.VIJFH, Float64) => [
        [(op_GradientF2C!, :none), (P100 = 217.9, A100 = 65.5, CPU = 1746.0)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 218.2, A100 = 65.49, CPU = 1754.0)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 238.2, A100 = 81.53, CPU = 1899.0)],
        [(op_DivergenceF2C!, :none), (P100 = 1120.0, A100 = 361.7, CPU = 6792.0)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 1119.0, A100 = 365.8, CPU = 6776.0)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 1119.0, A100 = 365.6, CPU = 6776.0)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 1132.0, A100 = 371.6, CPU = 6720.0)],
        [(op_InterpolateF2C!, :none), (P100 = 217.6, A100 = 65.29, CPU = 1701.0)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 197.1, A100 = 63.91, CPU = 1713.0)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 196.8, A100 = 63.76, CPU = 1698.0)],
        [(op_broadcast_example0!, :none), (P100 = 177.6, A100 = 64.22, CPU = 1059.0)],
        [(op_broadcast_example1!, :none), (P100 = 321.3, A100 = 115.8, CPU = 154330.0)],
        [(op_broadcast_example2!, :none), (P100 = 321.4, A100 = 115.7, CPU = 152689.0)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 195.5, A100 = 63.47, CPU = 1758.0)],
        [(op_BottomBiasedF2C!, :none), (P100 = 219.5, A100 = 65.08, CPU = 1711.0)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 218.6, A100 = 65.28, CPU = 1754.0)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 196.8, A100 = 63.59, CPU = 1847.0)],
        [(op_TopBiasedF2C!, :none), (P100 = 223.1, A100 = 65.29, CPU = 1582.0)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 222.1, A100 = 65.4, CPU = 1551.0)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 459.5, A100 = 146.7, CPU = 4669.0)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 334.2, A100 = 113.1, CPU = 3432.0)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 796.6, A100 = 368.7, CPU = 5650.0)],
        # See the note in reference_times_column: the changed-BC entries' CPU
        # reference times were carried over, not re-measured.
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 467.8, A100 = 161.7, CPU = 4474.0)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 428.2, A100 = 148.8, CPU = 4470.0)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 344.9, A100 = 121.2, CPU = 3566.0)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 337.5, A100 = 123.1, CPU = 3663.0)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 692.3, A100 = 230.2, CPU = 7251.0)],
        [(op_nest_interp_4!, :none), (P100 = 235.0, A100 = 65.64)],
        [(op_nest_interp_8!, :none), (P100 = 316.3, A100 = 104.3)],
        [(op_nest_costly_8!, :none), (P100 = 1302.0, A100 = 532.3)],
        [(op_biharmonic!, :none), (P100 = 668.5, A100 = 196.4)],
        [(op_wide_sum_8!, :none), (P100 = 857.3, A100 = 275.9)],
        [(op_diffusion_like!, :none), (P100 = 559.0, A100 = 191.4)],
        [(op_advection_like!, :none), (P100 = 520.5, A100 = 175.0)],
        [(op_atmos_like!, :none), (P100 = 1322.0, A100 = 481.5)],
        ],
        (DataLayouts.VIJFH, Float32) => [
        [(op_GradientF2C!, :none), (P100 = 131.2, A100 = 41)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 132.1, A100 = 37.87)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 168.1, A100 = 58.69)],
        [(op_DivergenceF2C!, :none), (P100 = 613.0, A100 = 194.8)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 621.5, A100 = 194.4)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 614.4, A100 = 194.1)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 621.1, A100 = 196.4)],
        [(op_InterpolateF2C!, :none), (P100 = 134.4, A100 = 37.42)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 120.2, A100 = 37.99)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 115.2, A100 = 37.01)],
        [(op_broadcast_example0!, :none), (P100 = 95.87, A100 = 22.97)],
        [(op_broadcast_example1!, :none), (P100 = 174.0, A100 = 63.54)],
        [(op_broadcast_example2!, :none), (P100 = 174.0, A100 = 63.47)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 109.8, A100 = 36.42)],
        [(op_BottomBiasedF2C!, :none), (P100 = 131.6, A100 = 37.37)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 132.1, A100 = 37.44)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 110.3, A100 = 36.55)],
        [(op_TopBiasedF2C!, :none), (P100 = 132.8, A100 = 37.56)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 133.3, A100 = 37.51)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 259.5, A100 = 83.54)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 203.5, A100 = 63.24)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 555.6, A100 = 157.7)],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 313.9, A100 = 117.8)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 259.3, A100 = 88.75)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 221.2, A100 = 79.3)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 219.5, A100 = 78.31)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 395.7, A100 = 130.8)],
        [(op_nest_interp_4!, :none), (P100 = 175.5, A100 = 43.28)],
        [(op_nest_interp_8!, :none), (P100 = 295.0, A100 = 74.6)],
        [(op_nest_costly_8!, :none), (P100 = 716.0, A100 = 197.8)],
        [(op_biharmonic!, :none), (P100 = 434.4, A100 = 138.5)],
        [(op_wide_sum_8!, :none), (P100 = 541.7, A100 = 147.4)],
        [(op_diffusion_like!, :none), (P100 = 359.6, A100 = 117.9)],
        [(op_advection_like!, :none), (P100 = 321.1, A100 = 104.2)],
        [(op_atmos_like!, :none), (P100 = 854.3, A100 = 269.5)],
        ],
        (DataLayouts.VIJHF, Float64) => [
        [(op_GradientF2C!, :none), (P100 = 187.0, A100 = 60.56)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 187.7, A100 = 60.72)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 222.5, A100 = 77.83)],
        [(op_DivergenceF2C!, :none), (P100 = 1073.0, A100 = 361.6)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 1074.0, A100 = 361.4)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 1072.0, A100 = 360.7)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 1096.0, A100 = 364.5)],
        [(op_InterpolateF2C!, :none), (P100 = 187.3, A100 = 60.53)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 180.2, A100 = 60.44)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 190.7, A100 = 60.17)],
        [(op_broadcast_example0!, :none), (P100 = 178.3, A100 = 62.17)],
        [(op_broadcast_example1!, :none), (P100 = 326.4, A100 = 108.3)],
        [(op_broadcast_example2!, :none), (P100 = 326.5, A100 = 108.3)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 181.0, A100 = 59.97)],
        [(op_BottomBiasedF2C!, :none), (P100 = 191.0, A100 = 60.55)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 190.5, A100 = 60.47)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 181.9, A100 = 60.09)],
        [(op_TopBiasedF2C!, :none), (P100 = 193.0, A100 = 60.69)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 192.6, A100 = 60.61)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 434.8, A100 = 139.8)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 326.1, A100 = 108.9)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 754.0, A100 = 347.8)],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 450.5, A100 = 156.5)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 420.5, A100 = 145.2)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 333.3, A100 = 118.4)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 332.4, A100 = 118.3)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 660.5, A100 = 229.9)],
        [(op_nest_interp_4!, :none), (P100 = 211.0, A100 = 64.02)],
        [(op_nest_interp_8!, :none), (P100 = 314.3, A100 = 104.3)],
        [(op_nest_costly_8!, :none), (P100 = 1295.0, A100 = 529.2)],
        [(op_biharmonic!, :none), (P100 = 611.7, A100 = 217.9)],
        [(op_wide_sum_8!, :none), (P100 = 855.4, A100 = 260.8)],
        [(op_diffusion_like!, :none), (P100 = 537.6, A100 = 186.2)],
        [(op_advection_like!, :none), (P100 = 511.9, A100 = 172.7)],
        [(op_atmos_like!, :none), (P100 = 1345.0, A100 = 471.3)],
        ],
        (DataLayouts.VIJHF, Float32) => [
        [(op_GradientF2C!, :none), (P100 = 101.5, A100 = 37.19)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 103.2, A100 = 35.47)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 146.7, A100 = 53.68)],
        [(op_DivergenceF2C!, :none), (P100 = 601.9, A100 = 194.6)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 591.4, A100 = 194.3)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 606.4, A100 = 193.9)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 608.0, A100 = 193.3)],
        [(op_InterpolateF2C!, :none), (P100 = 101.3, A100 = 35.24)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 99.41, A100 = 35.85)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 98.47, A100 = 35.38)],
        [(op_broadcast_example0!, :none), (P100 = 113.8, A100 = 24.14)],
        [(op_broadcast_example1!, :none), (P100 = 175.3, A100 = 61.38)],
        [(op_broadcast_example2!, :none), (P100 = 174.8, A100 = 61.45)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 93.08, A100 = 34.98)],
        [(op_BottomBiasedF2C!, :none), (P100 = 98.88, A100 = 35.15)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 99.02, A100 = 35.05)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 93.11, A100 = 35.13)],
        [(op_TopBiasedF2C!, :none), (P100 = 99.66, A100 = 35.22)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 99.92, A100 = 35.21)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 235.6, A100 = 79.7)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 196.1, A100 = 61.6)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 555.8, A100 = 157.4)],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 294.3, A100 = 112.2)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 251.6, A100 = 83.65)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 213.7, A100 = 76.07)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 211.9, A100 = 75.57)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 398.4, A100 = 126.4)],
        [(op_nest_interp_4!, :none), (P100 = 170.6, A100 = 42.5)],
        [(op_nest_interp_8!, :none), (P100 = 287.1, A100 = 79.9)],
        [(op_nest_costly_8!, :none), (P100 = 708.2, A100 = 196.9)],
        [(op_biharmonic!, :none), (P100 = 463.3, A100 = 156.1)],
        [(op_wide_sum_8!, :none), (P100 = 522.4, A100 = 148.7)],
        [(op_diffusion_like!, :none), (P100 = 334.4, A100 = 112.6)],
        [(op_advection_like!, :none), (P100 = 320.0, A100 = 96.95)],
        [(op_atmos_like!, :none), (P100 = 872.2, A100 = 289.3)],
        ],
    )
    table = get(tables, (VIJH, float_type), nothing)
    isnothing(table) && return nothing
    return select_device(table, dev)
end

#! format: on
