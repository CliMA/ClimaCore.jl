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
# downstream); the inner gradient carries the SetGradient boundary conditions,
# since GradientC2F no longer accepts SetValue.
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
function benchmark_func!(bm, t_min, trials, fun, c, f, verbose = false; compile::Bool)
    device = ClimaComms.device(c)
    all_bcs = bcs_tested(c, fun)
    for (ibc,bcs) in enumerate(bcs_tested(c, fun))
        key = (fun, bc_name(bcs)...)
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
                problem_size=size(cf)
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
    benchmark_operators_base(bm, trials, t_min, cfield, ffield, "column"; compile)

    # Results are reported without assertions because the measurements are not yet
    # stable enough for maintainable baselines.
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
    benchmark_operators_base(bm, trials, t_min, cfield, ffield, "sphere"; compile)

    # Tests are removed since they're flaky. And maintaining
    # them before they're converged is a bit of work..
    tabulate_benchmark(bm)
    return (; trials, t_min)
end

function benchmark_operators_base(bm, trials, t_min, cfield, ffield, name; compile::Bool)
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
        benchmark_func!(bm, t_min, trials, op, cfield, ffield, #= verbose = =# false; compile)
    end

    return nothing
end

const GPU_BUFFER_COLUMN = 1.3
const GPU_BUFFER_SPHERE = 1.25

# Regressions fail the test on GPU (where CI pins the hardware) and only warn
# on CPU.
function check_regressions(t_min, results, device)
    for (params, ref_time) in results
        passed = t_min[params] ≤ ref_time
        if !passed
            @warn "Possible regression: $params, time=$(t_min[params]), ref_time=$ref_time"
        end
        if device isa ClimaComms.CUDADevice
            @test passed
        end
    end
end

function test_results_column(t_min, device = ClimaComms.device())
    # If these tests fail, just update the numbers (or the
    # buffer) so long its not an egregious regression.
    buffer = device isa ClimaComms.CUDADevice ? GPU_BUFFER_COLUMN : 2
    ns = 1
    μs = 10^3
    results_gpu = [
    [(op_GradientF2C!, :none), 17.634*μs*buffer],
    [(op_GradientF2C!, :SetValue, :SetValue), 17.288*μs*buffer],
    [(op_GradientC2F!, :SetGradient, :SetGradient), 17.764*μs*buffer],
    [(op_DivergenceF2C!, :none), 18.716*μs*buffer],
    [(op_DivergenceF2C!, :SetValue, :SetValue), 19.195*μs*buffer],
    [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), 20.848*μs*buffer],
    [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), 19.584*μs*buffer],
    [(op_InterpolateF2C!, :none), 16.874*μs*buffer],
    [(op_InterpolateC2F!, :SetValue, :SetValue), 17.699*μs*buffer],
    [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), 17.200*μs*buffer],
    [(op_broadcast_example2!, :none), 20.828*μs*buffer],
    [(op_BottomBiasedC2F!, :SetValue), 17.169*μs*buffer],
    [(op_BottomBiasedF2C!, :none), 17.240*μs*buffer],
    [(op_BottomBiasedF2C!, :SetValue), 17.155*μs*buffer],
    [(op_TopBiasedC2F!, :SetValue), 18.111*μs*buffer],
    [(op_TopBiasedF2C!, :none), 17.523*μs*buffer],
    [(op_TopBiasedF2C!, :SetValue), 17.974*μs*buffer],
    [(op_CurlC2F!, :SetCurl, :SetCurl), 17.262*μs*buffer],
    [(op_UpwindBiasedProductC2F!, :none), 18.260*μs*buffer],
    [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), 23.883*μs*buffer],
    [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), 19.380*μs*buffer],
    [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 19.888*μs*buffer],
    [(op_div_interp_CC!, :SetValue, :SetValue, :none), 18.918*μs*buffer],
    [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), 19.106*μs*buffer],
    [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), 20.160*μs*buffer],
    [(op_nest_interp_4!, :none), 19.652*μs*buffer],
    [(op_nest_interp_8!, :none), 22.908*μs*buffer],
    [(op_nest_costly_8!, :none), 209.583*μs*buffer],
    [(op_biharmonic!, :none), 22.044*μs*buffer],
    [(op_wide_sum_8!, :none), 30.696*μs*buffer],
    [(op_diffusion_like!, :none), 21.361*μs*buffer],
    [(op_advection_like!, :none), 21.209*μs*buffer],
    [(op_atmos_like!, :none), 38.611*μs*buffer],
    ]
    results_cpu = [
    [(op_GradientF2C!, :none), 253.100*ns*buffer],
    [(op_GradientF2C!, :SetValue, :SetValue), 270.448*ns*buffer],
    [(op_GradientC2F!, :SetGradient, :SetGradient), 242.053*ns*buffer],
    [(op_DivergenceF2C!, :none), 1.005*μs*buffer],
    [(op_DivergenceF2C!, :SetValue, :SetValue), 1.076*μs*buffer],
    [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), 1.076*μs*buffer],
    [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), 878.028*ns*buffer],
    [(op_InterpolateF2C!, :none), 254.523*ns*buffer],
    [(op_InterpolateC2F!, :SetValue, :SetValue), 254.241*ns*buffer],
    [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), 241.308*ns*buffer],
    [(op_broadcast_example2!, :none), 555.039*ns*buffer],
    [(op_BottomBiasedC2F!, :SetValue), 207.264*ns*buffer],
    [(op_BottomBiasedF2C!, :none), 137.031*ns*buffer],
    [(op_BottomBiasedF2C!, :SetValue), 185.135*ns*buffer],
    [(op_TopBiasedC2F!, :SetValue), 129.971*ns*buffer],
    [(op_TopBiasedF2C!, :none), 142.120*ns*buffer],
    [(op_TopBiasedF2C!, :SetValue), 141.446*ns*buffer],
    [(op_CurlC2F!, :SetCurl, :SetCurl), 1.692*μs*buffer],
    [(op_UpwindBiasedProductC2F!, :none), 765.401*ns*buffer],
    [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), 2.540*μs*buffer],
    # The reference times of entries whose boundary conditions changed in the
    # operator refactor (op_DivergenceF2C! above, op_divgrad_CC!,
    # op_div_interp_FF!) were carried over from the closest pre-refactor
    # entries rather than re-measured; they are within-buffer tripwires, not
    # measurements.
    [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), 924.147*ns*buffer],
    [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 876.510*ns*buffer],
    [(op_div_interp_CC!, :SetValue, :SetValue, :none), 721.119*ns*buffer],
    [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), 686.581*ns*buffer],
    [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), 5.047*μs*buffer],
    ]
    results = device isa ClimaComms.CUDADevice ? results_gpu : results_cpu
    check_regressions(t_min, results, device)
end

function test_results_sphere(
    t_min,
    device = ClimaComms.device();
    VIJH = DataLayouts.VIJFH,
    float_type = Float64,
)
    # If these tests fail, just update the numbers (or the
    # buffer) so long its not an egregious regression.
    buffer = device isa ClimaComms.CUDADevice ? GPU_BUFFER_SPHERE : 2
    μs = 10^3
    ms = 10^6
    results_gpu = Dict(
        (DataLayouts.VIJFH, Float64) => [
        [(op_GradientF2C!, :none), 499.0*μs*buffer],
        [(op_GradientF2C!, :SetValue, :SetValue), 515.7*μs*buffer],
        [(op_GradientC2F!, :SetGradient, :SetGradient), 641.2*μs*buffer],
        [(op_DivergenceF2C!, :none), 1212.0*μs*buffer],
        [(op_DivergenceF2C!, :SetValue, :SetValue), 1192.0*μs*buffer],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), 1490.0*μs*buffer],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), 1296.0*μs*buffer],
        [(op_InterpolateF2C!, :none), 496.4*μs*buffer],
        [(op_InterpolateC2F!, :SetValue, :SetValue), 510.1*μs*buffer],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), 504.2*μs*buffer],
        [(op_broadcast_example0!, :none), 197.9*μs*buffer],
        [(op_broadcast_example1!, :none), 354.5*μs*buffer],
        [(op_broadcast_example2!, :none), 351.7*μs*buffer],
        [(op_BottomBiasedC2F!, :SetValue), 471.5*μs*buffer],
        [(op_BottomBiasedF2C!, :none), 496.9*μs*buffer],
        [(op_BottomBiasedF2C!, :SetValue), 507.2*μs*buffer],
        [(op_TopBiasedC2F!, :SetValue), 508.6*μs*buffer],
        [(op_TopBiasedF2C!, :none), 496.4*μs*buffer],
        [(op_TopBiasedF2C!, :SetValue), 500.3*μs*buffer],
        [(op_CurlC2F!, :SetCurl, :SetCurl), 817.6*μs*buffer],
        [(op_UpwindBiasedProductC2F!, :none), 1081.0*μs*buffer],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), 1931.0*μs*buffer],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), 1162.0*μs*buffer],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 1012.0*μs*buffer],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), 913.9*μs*buffer],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), 895.2*μs*buffer],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), 1296.0*μs*buffer],
        [(op_nest_interp_4!, :none), 593.2*μs*buffer],
        [(op_nest_interp_8!, :none), 738.5*μs*buffer],
        [(op_nest_costly_8!, :none), 1944.0*μs*buffer],
        [(op_biharmonic!, :none), 1542.0*μs*buffer],
        [(op_wide_sum_8!, :none), 1584.0*μs*buffer],
        [(op_diffusion_like!, :none), 1226.0*μs*buffer],
        [(op_advection_like!, :none), 1227.0*μs*buffer],
        [(op_atmos_like!, :none), 3872.0*μs*buffer],
        ],
        (DataLayouts.VIJHF, Float64) => [
        [(op_GradientF2C!, :none), 425.4*μs*buffer],
        [(op_GradientF2C!, :SetValue, :SetValue), 511.9*μs*buffer],
        [(op_GradientC2F!, :SetGradient, :SetGradient), 629.2*μs*buffer],
        [(op_DivergenceF2C!, :none), 1213.0*μs*buffer],
        [(op_DivergenceF2C!, :SetValue, :SetValue), 1185.0*μs*buffer],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), 1517.0*μs*buffer],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), 1274.0*μs*buffer],
        [(op_InterpolateF2C!, :none), 430.7*μs*buffer],
        [(op_InterpolateC2F!, :SetValue, :SetValue), 474.8*μs*buffer],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), 500.4*μs*buffer],
        [(op_broadcast_example0!, :none), 498.9*μs*buffer],
        [(op_broadcast_example1!, :none), 528.3*μs*buffer],
        [(op_broadcast_example2!, :none), 528.1*μs*buffer],
        [(op_BottomBiasedC2F!, :SetValue), 505.1*μs*buffer],
        [(op_BottomBiasedF2C!, :none), 494.1*μs*buffer],
        [(op_BottomBiasedF2C!, :SetValue), 500.3*μs*buffer],
        [(op_TopBiasedC2F!, :SetValue), 505.2*μs*buffer],
        [(op_TopBiasedF2C!, :none), 494.6*μs*buffer],
        [(op_TopBiasedF2C!, :SetValue), 500.5*μs*buffer],
        [(op_CurlC2F!, :SetCurl, :SetCurl), 803.5*μs*buffer],
        [(op_UpwindBiasedProductC2F!, :none), 1062.0*μs*buffer],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), 1966.0*μs*buffer],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), 1141.0*μs*buffer],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 990.1*μs*buffer],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), 910.0*μs*buffer],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), 886.3*μs*buffer],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), 1203.0*μs*buffer],
        [(op_nest_interp_4!, :none), 602.5*μs*buffer],
        [(op_nest_interp_8!, :none), 751.7*μs*buffer],
        [(op_nest_costly_8!, :none), 1982.0*μs*buffer],
        [(op_biharmonic!, :none), 1528.0*μs*buffer],
        [(op_wide_sum_8!, :none), 1581.0*μs*buffer],
        [(op_diffusion_like!, :none), 1212.0*μs*buffer],
        [(op_advection_like!, :none), 1112.0*μs*buffer],
        [(op_atmos_like!, :none), 4024.0*μs*buffer],
        ],
        (DataLayouts.VIJFH, Float32) => [
        [(op_GradientF2C!, :none), 491.9*μs*buffer],
        [(op_GradientF2C!, :SetValue, :SetValue), 507.3*μs*buffer],
        [(op_GradientC2F!, :SetGradient, :SetGradient), 629.7*μs*buffer],
        [(op_DivergenceF2C!, :none), 1037.0*μs*buffer],
        [(op_DivergenceF2C!, :SetValue, :SetValue), 1082.0*μs*buffer],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), 1267.0*μs*buffer],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), 1156.0*μs*buffer],
        [(op_InterpolateF2C!, :none), 386.5*μs*buffer],
        [(op_InterpolateC2F!, :SetValue, :SetValue), 499.0*μs*buffer],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), 496.9*μs*buffer],
        [(op_broadcast_example0!, :none), 127.1*μs*buffer],
        [(op_broadcast_example1!, :none), 238.9*μs*buffer],
        [(op_broadcast_example2!, :none), 238.8*μs*buffer],
        [(op_BottomBiasedC2F!, :SetValue), 498.8*μs*buffer],
        [(op_BottomBiasedF2C!, :none), 488.4*μs*buffer],
        [(op_BottomBiasedF2C!, :SetValue), 494.9*μs*buffer],
        [(op_TopBiasedC2F!, :SetValue), 501.1*μs*buffer],
        [(op_TopBiasedF2C!, :none), 488.4*μs*buffer],
        [(op_TopBiasedF2C!, :SetValue), 491.5*μs*buffer],
        [(op_CurlC2F!, :SetCurl, :SetCurl), 790.9*μs*buffer],
        [(op_UpwindBiasedProductC2F!, :none), 1063.0*μs*buffer],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), 1580.0*μs*buffer],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), 1121.0*μs*buffer],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 984.1*μs*buffer],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), 881.2*μs*buffer],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), 877.3*μs*buffer],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), 1275.0*μs*buffer],
        [(op_nest_interp_4!, :none), 585.4*μs*buffer],
        [(op_nest_interp_8!, :none), 724.1*μs*buffer],
        [(op_nest_costly_8!, :none), 1714.0*μs*buffer],
        [(op_biharmonic!, :none), 1435.0*μs*buffer],
        [(op_wide_sum_8!, :none), 1524.0*μs*buffer],
        [(op_diffusion_like!, :none), 1187.0*μs*buffer],
        [(op_advection_like!, :none), 1083.0*μs*buffer],
        [(op_atmos_like!, :none), 3002.0*μs*buffer],
        ],
        (DataLayouts.VIJHF, Float32) => [
        [(op_GradientF2C!, :none), 449.3*μs*buffer],
        [(op_GradientF2C!, :SetValue, :SetValue), 503.0*μs*buffer],
        [(op_GradientC2F!, :SetGradient, :SetGradient), 619.3*μs*buffer],
        [(op_DivergenceF2C!, :none), 1029.0*μs*buffer],
        [(op_DivergenceF2C!, :SetValue, :SetValue), 1074.0*μs*buffer],
        [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), 1269.0*μs*buffer],
        [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), 1131.0*μs*buffer],
        [(op_InterpolateF2C!, :none), 484.0*μs*buffer],
        [(op_InterpolateC2F!, :SetValue, :SetValue), 495.1*μs*buffer],
        [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), 495.3*μs*buffer],
        [(op_broadcast_example0!, :none), 496.8*μs*buffer],
        [(op_broadcast_example1!, :none), 521.7*μs*buffer],
        [(op_broadcast_example2!, :none), 521.3*μs*buffer],
        [(op_BottomBiasedC2F!, :SetValue), 494.3*μs*buffer],
        [(op_BottomBiasedF2C!, :none), 485.0*μs*buffer],
        [(op_BottomBiasedF2C!, :SetValue), 490.1*μs*buffer],
        [(op_TopBiasedC2F!, :SetValue), 498.7*μs*buffer],
        [(op_TopBiasedF2C!, :none), 484.1*μs*buffer],
        [(op_TopBiasedF2C!, :SetValue), 490.3*μs*buffer],
        [(op_CurlC2F!, :SetCurl, :SetCurl), 778.7*μs*buffer],
        [(op_UpwindBiasedProductC2F!, :none), 1054.0*μs*buffer],
        [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), 1583.0*μs*buffer],
        [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), 1112.0*μs*buffer],
        [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 968.5*μs*buffer],
        [(op_div_interp_CC!, :SetValue, :SetValue, :none), 872.3*μs*buffer],
        [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), 866.2*μs*buffer],
        [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), 1257.0*μs*buffer],
        [(op_nest_interp_4!, :none), 583.6*μs*buffer],
        [(op_nest_interp_8!, :none), 716.2*μs*buffer],
        [(op_nest_costly_8!, :none), 1705.0*μs*buffer],
        [(op_biharmonic!, :none), 1362.0*μs*buffer],
        [(op_wide_sum_8!, :none), 1510.0*μs*buffer],
        [(op_diffusion_like!, :none), 1177.0*μs*buffer],
        [(op_advection_like!, :none), 1072.0*μs*buffer],
        [(op_atmos_like!, :none), 3007.0*μs*buffer],
        ],
    )
    results_cpu = [
    [(op_GradientF2C!, :none), 1.746*ms*buffer],
    [(op_GradientF2C!, :SetValue, :SetValue), 1.754*ms*buffer],
    [(op_GradientC2F!, :SetGradient, :SetGradient), 1.899*ms*buffer],
    [(op_DivergenceF2C!, :none), 6.792*ms*buffer],
    [(op_DivergenceF2C!, :SetValue, :SetValue), 6.776*ms*buffer],
    [(op_DivergenceF2C!, :Extrapolate, :Extrapolate), 6.776*ms*buffer],
    [(op_DivergenceC2F!, :SetDivergence, :SetDivergence), 6.720*ms*buffer],
    [(op_InterpolateF2C!, :none), 1.701*ms*buffer],
    [(op_InterpolateC2F!, :SetValue, :SetValue), 1.713*ms*buffer],
    [(op_InterpolateC2F!, :Extrapolate, :Extrapolate), 1.698*ms*buffer],
    [(op_broadcast_example0!, :none), 1.059*ms*buffer],
    [(op_broadcast_example1!, :none), 154.330*ms*buffer],
    [(op_broadcast_example2!, :none), 152.689*ms*buffer],
    [(op_BottomBiasedC2F!, :SetValue), 1.758*ms*buffer],
    [(op_BottomBiasedF2C!, :none), 1.711*ms*buffer],
    [(op_BottomBiasedF2C!, :SetValue), 1.754*ms*buffer],
    [(op_TopBiasedC2F!, :SetValue), 1.847*ms*buffer],
    [(op_TopBiasedF2C!, :none), 1.582*ms*buffer],
    [(op_TopBiasedF2C!, :SetValue), 1.551*ms*buffer],
    [(op_CurlC2F!, :SetCurl, :SetCurl), 4.669*ms*buffer],
    [(op_UpwindBiasedProductC2F!, :none), 3.432*ms*buffer],
    [(op_divUpwind3rdOrderBiasedProductC2F!, :none, :SetValue, :SetValue), 5.650*ms*buffer],
    # See the note in test_results_column: the changed-BC entries' reference
    # times were carried over, not re-measured.
    [(op_divgrad_CC!, :SetGradient, :SetGradient, :none), 4.474*ms*buffer],
    [(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 4.470*ms*buffer],
    [(op_div_interp_CC!, :SetValue, :SetValue, :none), 3.566*ms*buffer],
    [(op_div_interp_FF!, :none, :SetDivergence, :SetDivergence), 3.663*ms*buffer],
    [(op_divgrad_uₕ!, :none, :SetValue, :SetValue), 7.251*ms*buffer],
    ]
    results = if device isa ClimaComms.CUDADevice
        results_gpu[(VIJH, float_type)]
    else
        if float_type === Float32
            @info "No CPU Float32 reference times yet; skipping the checks."
            return nothing
        end
        results_cpu
    end
    check_regressions(t_min, results, device)
end

#! format: on
