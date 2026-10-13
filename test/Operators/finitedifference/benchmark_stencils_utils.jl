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
else
    device_name = "CPU"
end

include(joinpath(pkgdir(ClimaCore), "benchmarks/scripts/benchmark_utils.jl"))

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
            trials[key] = BenchmarkTools.@benchmark ClimaComms.@cuda_sync $device $fun($c, $f, $bcs)
            cf = Fields.field_values(Fields.coordinate_field(axes(c)))
            kernel_time_s = minimum(trials[key].times) * 1e-9 # to seconds
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
            verbose && show(stdout, MIME("text/plain"), trials[key])

            t_min[key] = minimum(trials[key].times) # nano seconds
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
        trial = BenchmarkTools.@benchmark ClimaComms.@cuda_sync $device column_op_2mul_1add_cuda!($xarr, $yarr, $D, $U)
        show(stdout, MIME("text/plain"), trial)
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
            trial = BenchmarkTools.@benchmark ClimaComms.@cuda_sync $device sphere_op_2mul_1add_cuda!($xarr, $yarr, $D, $U)
            show(stdout, MIME("text/plain"), trial)
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

const GPU_BUFFER_COLUMN = 1.3
const GPU_BUFFER_SPHERE = 1.25

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
    buffer = device isa ClimaComms.CUDADevice ? GPU_BUFFER_COLUMN : 2
    results = reference_times_column(device; gpu)
    check_regressions(t_min, results, device, buffer)
end

# Measured minimum times (ns) for the column benchmarks, or `nothing` when
# there are no reference timings for this device.
function reference_times_column(
    device = ClimaComms.device();
    gpu = gpu_model(device_name),
)
    # Measured minimum times in μs per device; rows without an entry for a
    # device have no reference on it.
    table = [
        [(op_GradientF2C!, :none), (P100 = 16.051, A100 = 15.32, CPU = 0.2531)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 16.349, A100 = 15.31, CPU = 0.270448)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 16.743, A100 = 15.81, CPU = 0.242053)],
        [(op_DivergenceF2C!, :none), (P100 = 18.231, A100 = 16.91, CPU = 1.005)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 18.444, A100 = 17.2, CPU = 1.076)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 19.164, A100 = 18.02, CPU = 1.076)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 18.834, A100 = 17.52, CPU = 0.878028)],
        [(op_InterpolateF2C!, :none), (P100 = 16.215, A100 = 15.25, CPU = 0.254523)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 16.507, A100 = 15.13, CPU = 0.254241)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 16.316, A100 = 15.73, CPU = 0.241308)],
        [(op_broadcast_example2!, :none), (P100 = 19.808, A100 = 18.96, CPU = 0.555039)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 16.748, A100 = 15.66, CPU = 0.207264)],
        [(op_BottomBiasedF2C!, :none), (P100 = 16.273, A100 = 15.17, CPU = 0.137031)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 16.758, A100 = 15.91, CPU = 0.185135)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 16.78, A100 = 16.08, CPU = 0.129971)],
        [(op_TopBiasedF2C!, :none), (P100 = 16.462, A100 = 15.48, CPU = 0.14212)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 16.366, A100 = 15.44, CPU = 0.141446)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 17.082, A100 = 15.71, CPU = 1.692)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 17.632, A100 = 16.6, CPU = 0.765401)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 22.962, A100 = 22.03, CPU = 2.54)],
        # The CPU reference times of entries whose boundary conditions changed in
        # the operator refactor (op_DivergenceF2C! above, op_divgrad_CC!,
        # op_div_interp_FF!) were carried over from the closest pre-refactor
        # entries rather than re-measured; they are within-buffer tripwires, not
        # measurements.
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 18.584, A100 = 17.85, CPU = 0.924147)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 18.396, A100 = 16.91, CPU = 0.87651)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 18.087, A100 = 16.61, CPU = 0.721119)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 17.956, A100 = 17.57, CPU = 0.686581)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 19.73, A100 = 17.61, CPU = 5.047)],
        [(op_nest_interp_4!, :none), (P100 = 18.851, A100 = 16.83)],
        [(op_nest_interp_8!, :none), (P100 = 22.703, A100 = 19.41)],
        [(op_nest_costly_8!, :none), (P100 = 205.037, A100 = 134.49)],
        [(op_biharmonic!, :none), (P100 = 21.262, A100 = 19.14)],
        [(op_wide_sum_8!, :none), (P100 = 30.356, A100 = 25.279)],
        [(op_diffusion_like!, :none), (P100 = 20.566, A100 = 19.81)],
        [(op_advection_like!, :none), (P100 = 21.363, A100 = 19.17)],
        [(op_atmos_like!, :none), (P100 = 38.506, A100 = 31.29)],
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
    buffer = device isa ClimaComms.CUDADevice ? GPU_BUFFER_SPHERE : 2
    results = reference_times_sphere(device; VIJH, float_type, gpu)
    check_regressions(t_min, results, device, buffer)
end

# Measured minimum times (ns) for the sphere benchmarks, or `nothing` when
# there are no reference timings for this device / layout / float type.
function reference_times_sphere(
    device = ClimaComms.device();
    VIJH = DataLayouts.VIJFH,
    float_type = Float64,
    gpu = gpu_model(device_name),
)
    # Measured minimum times in μs per device; rows without an entry for a
    # device have no reference on it. The CPU times are layout-agnostic
    # tripwires (buffer 2) and are stored once, under VIJFH.
    dev = device isa ClimaComms.CUDADevice ? gpu : :CPU
    dev === :CPU && (VIJH = DataLayouts.VIJFH)
    tables = Dict(
        (DataLayouts.VIJFH, Float64) => [
        [(op_GradientF2C!, :none), (P100 = 357.7, A100 = 74.03, CPU = 1746.0)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 372.8, A100 = 80.46, CPU = 1754.0)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 487.9, A100 = 92.17, CPU = 1899.0)],
        [(op_DivergenceF2C!, :none), (P100 = 1104.0, A100 = 377.1, CPU = 6792.0)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 1092.0, A100 = 374.9, CPU = 6776.0)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 1384.0, A100 = 397.5, CPU = 6776.0)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 1144.0, A100 = 392.9, CPU = 6720.0)],
        [(op_InterpolateF2C!, :none), (P100 = 359.5, A100 = 73.36, CPU = 1701.0)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 372.1, A100 = 78.09, CPU = 1713.0)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 367.4, A100 = 73.18, CPU = 1698.0)],
        [(op_broadcast_example0!, :none), (P100 = 196.3, A100 = 79.41, CPU = 1059.0)],
        [(op_broadcast_example1!, :none), (P100 = 352.0, A100 = 129.3, CPU = 154330.0)],
        [(op_broadcast_example2!, :none), (P100 = 352.7, A100 = 129.8, CPU = 152689.0)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 368.7, A100 = 73.29, CPU = 1758.0)],
        [(op_BottomBiasedF2C!, :none), (P100 = 358.2, A100 = 73.57, CPU = 1711.0)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 366.5, A100 = 74.41, CPU = 1754.0)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 369.1, A100 = 79.47, CPU = 1847.0)],
        [(op_TopBiasedF2C!, :none), (P100 = 357.6, A100 = 73.06, CPU = 1582.0)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 362.1, A100 = 77.17, CPU = 1551.0)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 649.0, A100 = 148.4, CPU = 4669.0)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 887.2, A100 = 140.5, CPU = 3432.0)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 1893.0, A100 = 423.2, CPU = 5650.0)],
        # See the note in reference_times_column: the changed-BC entries' CPU
        # reference times were carried over, not re-measured.
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 962.3, A100 = 221.9, CPU = 4474.0)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 824.8, A100 = 205.3, CPU = 4470.0)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 735.3, A100 = 157.9, CPU = 3566.0)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 718.8, A100 = 172.2, CPU = 3663.0)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 1116.0, A100 = 255.9, CPU = 7251.0)],
        [(op_nest_interp_4!, :none), (P100 = 460.9, A100 = 86.52)],
        [(op_nest_interp_8!, :none), (P100 = 597.2, A100 = 107.3)],
        [(op_nest_costly_8!, :none), (P100 = 1733.0, A100 = 568.5)],
        [(op_biharmonic!, :none), (P100 = 1443.0, A100 = 331.7)],
        [(op_wide_sum_8!, :none), (P100 = 1483.0, A100 = 329.6)],
        [(op_diffusion_like!, :none), (P100 = 1126.0, A100 = 270.5)],
        [(op_advection_like!, :none), (P100 = 1019.0, A100 = 219.1)],
        [(op_atmos_like!, :none), (P100 = 3607.0, A100 = 841.5)],
        ],
        (DataLayouts.VIJFH, Float32) => [
        [(op_GradientF2C!, :none), (P100 = 354.9, A100 = 65.39)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 369.2, A100 = 73.96)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 479.3, A100 = 87.3)],
        [(op_DivergenceF2C!, :none), (P100 = 849.2, A100 = 217.8)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 889.2, A100 = 216.5)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 1058.0, A100 = 273.2)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 957.7, A100 = 233.9)],
        [(op_InterpolateF2C!, :none), (P100 = 352.7, A100 = 65.36)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 361.2, A100 = 71.6)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 360.0, A100 = 68.56)],
        [(op_broadcast_example0!, :none), (P100 = 126.9, A100 = 43.59)],
        [(op_broadcast_example1!, :none), (P100 = 238.5, A100 = 79.14)],
        [(op_broadcast_example2!, :none), (P100 = 238.4, A100 = 78.85)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 361.5, A100 = 66.56)],
        [(op_BottomBiasedF2C!, :none), (P100 = 350.7, A100 = 65.06)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 357.2, A100 = 65.69)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 362.8, A100 = 69.47)],
        [(op_TopBiasedF2C!, :none), (P100 = 349.5, A100 = 64.81)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 352.5, A100 = 70.16)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 625.4, A100 = 109.9)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 869.7, A100 = 138.9)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 1482.0, A100 = 317.9)],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 926.4, A100 = 202.2)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 802.5, A100 = 174.6)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 706.8, A100 = 151.5)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 702.9, A100 = 151.6)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 1061.0, A100 = 220.9)],
        [(op_nest_interp_4!, :none), (P100 = 439.4, A100 = 82.34)],
        [(op_nest_interp_8!, :none), (P100 = 565.2, A100 = 99.67)],
        [(op_nest_costly_8!, :none), (P100 = 1053.0, A100 = 426.1)],
        [(op_biharmonic!, :none), (P100 = 1336.0, A100 = 297.0)],
        [(op_wide_sum_8!, :none), (P100 = 1424.0, A100 = 279.6)],
        [(op_diffusion_like!, :none), (P100 = 1086.0, A100 = 257.6)],
        [(op_advection_like!, :none), (P100 = 982.3, A100 = 204.6)],
        [(op_atmos_like!, :none), (P100 = 2916.0, A100 = 658.1)],
        ],
        (DataLayouts.VIJHF, Float64) => [
        [(op_GradientF2C!, :none), (P100 = 356.2, A100 = 67.5)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 373.6, A100 = 75.81)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 478.7, A100 = 89.72)],
        [(op_DivergenceF2C!, :none), (P100 = 1100.0, A100 = 370.0)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 1078.0, A100 = 366.5)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 1402.0, A100 = 374.2)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 1164.0, A100 = 370.2)],
        [(op_InterpolateF2C!, :none), (P100 = 355.1, A100 = 67.52)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 367.7, A100 = 73.7)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 362.8, A100 = 68.85)],
        [(op_broadcast_example0!, :none), (P100 = 361.3, A100 = 88.9)],
        [(op_broadcast_example1!, :none), (P100 = 393.5, A100 = 122.4)],
        [(op_broadcast_example2!, :none), (P100 = 392.9, A100 = 122.7)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 366.5, A100 = 69.6)],
        [(op_BottomBiasedF2C!, :none), (P100 = 355.0, A100 = 67.67)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 362.3, A100 = 69.02)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 366.7, A100 = 74.27)],
        [(op_TopBiasedF2C!, :none), (P100 = 357.7, A100 = 66.54)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 362.6, A100 = 74.94)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 635.9, A100 = 142.0)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 867.8, A100 = 137.7)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 1870.0, A100 = 417.9)],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 943.6, A100 = 217.1)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 807.4, A100 = 197.1)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 732.1, A100 = 152.6)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 710.4, A100 = 166.5)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 1101.0, A100 = 244.2)],
        [(op_nest_interp_4!, :none), (P100 = 455.7, A100 = 83.83)],
        [(op_nest_interp_8!, :none), (P100 = 590.0, A100 = 111.2)],
        [(op_nest_costly_8!, :none), (P100 = 1724.0, A100 = 578.2)],
        [(op_biharmonic!, :none), (P100 = 1430.0, A100 = 300.0)],
        [(op_wide_sum_8!, :none), (P100 = 1484.0, A100 = 314.3)],
        [(op_diffusion_like!, :none), (P100 = 1112.0, A100 = 287.6)],
        [(op_advection_like!, :none), (P100 = 1012.0, A100 = 219.8)],
        [(op_atmos_like!, :none), (P100 = 3616.0, A100 = 935.6)],
        ],
        (DataLayouts.VIJHF, Float32) => [
        [(op_GradientF2C!, :none), (P100 = 349.1, A100 = 63.88)],
        [(op_GradientF2C!, :SetValue, :SetValue), (P100 = 363.4, A100 = 72.99)],
        [(op_GradientC2F!, :SetGradient, :SetGradient), (P100 = 470.3, A100 = 86.89)],
        [(op_DivergenceF2C!, :none), (P100 = 840.1, A100 = 208.9)],
        [(op_DivergenceF2C!, :SetValue, :SetValue), (P100 = 883.3, A100 = 209.6)],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), (P100 = 1060.0, A100 = 269.9)],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), (P100 = 933.8, A100 = 228.2)],
        [(op_InterpolateF2C!, :none), (P100 = 347.5, A100 = 63.05)],
        [(op_InterpolateC2F!, :SetValue, :SetValue), (P100 = 356.1, A100 = 70.24)],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), (P100 = 355.8, A100 = 70.48)],
        [(op_broadcast_example0!, :none), (P100 = 356.3, A100 = 80.13)],
        [(op_broadcast_example1!, :none), (P100 = 379.8, A100 = 91.6)],
        [(op_broadcast_example2!, :none), (P100 = 381.1, A100 = 91.08)],
        [(op_BottomBiasedC2F!, :SetValue), (P100 = 356.5, A100 = 65.56)],
        [(op_BottomBiasedF2C!, :none), (P100 = 346.2, A100 = 63.04)],
        [(op_BottomBiasedF2C!, :SetValue), (P100 = 350.5, A100 = 64.6)],
        [(op_TopBiasedC2F!, :SetValue), (P100 = 361.8, A100 = 68.54)],
        [(op_TopBiasedF2C!, :none), (P100 = 345.6, A100 = 62.94)],
        [(op_TopBiasedF2C!, :SetValue), (P100 = 353.6, A100 = 70.96)],
        [(op_CurlC2F!, :SetCurl, :SetCurl), (P100 = 614.1, A100 = 108.7)],
        [(op_UpwindBiasedProductC2F!, :none), (P100 = 862.7, A100 = 135.5)],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), (P100 = 1484.0, A100 = 316.0)],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), (P100 = 914.8, A100 = 199.6)],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 785.0, A100 = 172.4)],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), (P100 = 698.4, A100 = 157.9)],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), (P100 = 693.0, A100 = 148.9)],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), (P100 = 1049.0, A100 = 216.2)],
        [(op_nest_interp_4!, :none), (P100 = 438.6, A100 = 81.4)],
        [(op_nest_interp_8!, :none), (P100 = 557.8, A100 = 98.0)],
        [(op_nest_costly_8!, :none), (P100 = 1048.0, A100 = 458.7)],
        [(op_biharmonic!, :none), (P100 = 1260.0, A100 = 276.2)],
        [(op_wide_sum_8!, :none), (P100 = 1412.0, A100 = 272.4)],
        [(op_diffusion_like!, :none), (P100 = 1076.0, A100 = 254.4)],
        [(op_advection_like!, :none), (P100 = 973.3, A100 = 201.7)],
        [(op_atmos_like!, :none), (P100 = 2930.0, A100 = 663.9)],
        ],
    )
    table = get(tables, (VIJH, float_type), nothing)
    isnothing(table) && return nothing
    return select_device(table, dev)
end

#! format: on
