#! format: off
using Test
using ClimaComms
ClimaComms.@import_required_backends
using StaticArrays, IntervalSets, LinearAlgebra
import BenchmarkTools
import StatsBase
import OrderedCollections
using ClimaCore.Geometry: ⊗

import ClimaCore
include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

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

function set_value_divgrad_uₕ_bcs(c) # real-world example
    FT = Spaces.undertype(axes(c))
    top_val = Geometry.Contravariant3Vector(FT(0)) ⊗
            Geometry.Covariant12Vector(FT(0), FT(0))
    bottom_val = Geometry.Contravariant3Vector(FT(0)) ⊗
            Geometry.Covariant12Vector(FT(0), FT(0))
    return (;top = Operators.SetValue(top_val),
             bottom = Operators.Extrapolate())
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

function set_vec_value_bcs(c)
    FT = Spaces.undertype(axes(c))
    wvec = Geometry.WVector
    return (;bottom = Operators.SetValue(wvec(FT(0))),
             top = Operators.SetValue(wvec(FT(0))))
end

function set_upwind_biased_bcs(c)
    return (;bottom = Operators.FirstOrderOneSided(),
             top = Operators.FirstOrderOneSided())
end

function set_upwind_biased_bcs(c)
    return (;bottom = Operators.ThirdOrderOneSided(),
             top = Operators.ThirdOrderOneSided())
end

function set_upwind_biased_3_bcs(c)
    return (;bottom = Operators.ThirdOrderOneSided(),
             top = Operators.ThirdOrderOneSided())
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

function set_divergence_contra3_bcs(c)
    FT = Spaces.undertype(axes(c))
    contra3 = Geometry.Contravariant3Vector
    return (;bottom = Operators.SetDivergence(contra3(FT(0))),
             top = Operators.SetDivergence(contra3(FT(0))))
end

function set_gradient_value_bcs(c)
    FT = Spaces.undertype(axes(c))
    wvec = Geometry.WVector
    return (;bottom = Operators.SetGradient(wvec(FT(0))),
             top = Operators.SetGradient(wvec(FT(0))))
end

function set_gradient_bcs(c)
    FT = Spaces.undertype(axes(c))
    return (;bottom = Operators.SetGradient(FT(0)),
             top = Operators.SetGradient(FT(0)))
end

function set_gradient_contra3_bcs(c)
    FT = Spaces.undertype(axes(c))
    contra3 = Geometry.Contravariant3Vector
    return (;bottom = Operators.SetGradient(FT(0)),
             top = Operators.SetGradient(FT(0)))
end

function set_curl_bcs(c)
    FT = Spaces.undertype(axes(c))
    contra12 = Geometry.Contravariant12Vector
    return (;bottom = Operators.SetCurl(contra12(FT(0), FT(0))),
             top = Operators.SetCurl(contra12(FT(0), FT(0))))
end

function set_curl_value_bcs(c)
    FT = Spaces.undertype(axes(c))
    cov12 = Geometry.Covariant12Vector
    return (;bottom = Operators.SetValue(cov12(FT(0), FT(0))),
             top = Operators.SetValue(cov12(FT(0), FT(0))))
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
bcs_tested(c, ::typeof(op_GradientC2F!)) = (set_gradient_value_bcs(c), set_value_bcs(c))
bcs_tested(c, ::typeof(op_DivergenceF2C!)) = ((;), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_DivergenceC2F!)) = (set_divergence_bcs(c), )
bcs_tested(c, ::typeof(op_InterpolateF2C!)) = ((;), )
bcs_tested(c, ::typeof(op_InterpolateC2F!)) = (set_value_bcs(c), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_LeftBiasedC2F!)) = (set_bot_value_bc(c),)
bcs_tested(c, ::typeof(op_LeftBiasedF2C!)) = ((;), set_bot_value_bc(c))
bcs_tested(c, ::typeof(op_RightBiasedC2F!)) = (set_top_value_bc(c),)
bcs_tested(c, ::typeof(op_RightBiasedF2C!)) = ((;), set_top_value_bc(c))
bcs_tested(c, ::typeof(op_CurlC2F!)) = (set_curl_bcs(c), set_curl_value_bcs(c))
bcs_tested(c, ::typeof(op_UpwindBiasedProductC2F!)) = (set_value_bcs(c), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_Upwind3rdOrderBiasedProductC2F!)) = (set_upwind_biased_3_bcs(c), extrapolate_bcs(c))

# Composed operators (bcs handled case-by-case)
bcs_tested(c, ::typeof(op_divUpwind3rdOrderBiasedProductC2F!)) =
    ((; inner = set_upwind_biased_3_bcs(c), outer = set_value_contra3_bcs(c)), )
bcs_tested(c, ::typeof(op_divgrad_CC!)) =
    ((; inner = set_value_bcs(c), outer = (;)), )
bcs_tested(c, ::typeof(op_divgrad_FF!)) =
    ((; inner = (;), outer = set_divergence_bcs(c)), )
bcs_tested(c, ::typeof(op_div_interp_CC!)) =
    ((; inner = set_value_contra3_bcs(c), outer = (;)), )
bcs_tested(c, ::typeof(op_div_interp_FF!)) =
    ((; inner = (;), outer = set_value_contra3_bcs(c)), )
bcs_tested(c, ::typeof(op_divgrad_uₕ!)) =
    (
        (; inner = (;), outer = set_value_divgrad_uₕ_bcs(c)),
        (; inner = (;), outer = set_value_divgrad_uₕ_maybe_field_bcs(c)),
    )

function benchmark_func!(t_min, trials, fun, c, f, verbose = false; compile::Bool)
    device = ClimaComms.device(c)
    for bcs in bcs_tested(c, fun)
        h_space = nameof(typeof(axes(c)))
        key = (h_space, fun, bc_name(bcs)...)
        if compile
            fun(c, f, bcs)
        else
            verbose && @info "\n@benchmarking $key"
            trials[key] = BenchmarkTools.@benchmark ClimaComms.@cuda_sync $device $fun($c, $f, $bcs)
        end
        if haskey(trials, key)
            verbose && show(stdout, MIME("text/plain"), trials[key])

            t_min[key] = minimum(trials[key].times) # nano seconds
            t_pretty = BenchmarkTools.prettytime(t_min[key])
            verbose || @info "$t_pretty <=> t_min[$key]"
        end
    end
end

function column_benchmark_arrays(device, z_elems, ::Type{FT}; compile::Bool) where {FT}
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

function sphere_benchmark_arrays(device, z_elems, helem, Nq, ::Type{FT}; compile::Bool) where {FT}
    ArrayType = ClimaComms.array_type(device)
    # VIJFH
    Nh = helem * helem * 6
    cdims = (z_elems  , Nq, Nq, 1, Nh)
    fdims = (z_elems+1, Nq, Nq, 1, Nh)
    L = ArrayType(zeros(FT, cdims...))
    D = ArrayType(zeros(FT, cdims...))
    U = ArrayType(zeros(FT, cdims...))
    xarr = ArrayType(rand(FT, cdims...))
    uₕ_x = ArrayType(rand(FT, cdims...))
    uₕ_y = ArrayType(rand(FT, cdims...))
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

function benchmark_operators_column(::Type{FT}; z_elems, helem, Nq, compile::Bool = false) where {FT}
    device = ClimaComms.device()
    @show device
    trials = OrderedCollections.OrderedDict()
    t_min = OrderedCollections.OrderedDict()
    column_benchmark_arrays(device, z_elems, FT; compile)

    cspace = TU.ColumnCenterFiniteDifferenceSpace(FT; zelem=z_elems)
    fspace = Spaces.FaceFiniteDifferenceSpace(cspace)
    cfield = fill(field_vars(FT), cspace)
    ffield = fill(field_vars(FT), fspace)
    benchmark_operators_base(trials, t_min, cfield, ffield, "column"; compile)

    # Tests are removed since they're flakey. And maintaining
    # them before they're converged is a bit of work..
    compile || test_results_column(t_min)
    return (; trials, t_min)
end

function benchmark_operators_sphere(::Type{FT}; z_elems, helem, Nq, compile::Bool = false) where {FT}
    device = ClimaComms.device()
    @show device
    trials = OrderedCollections.OrderedDict()
    t_min = OrderedCollections.OrderedDict()
    sphere_benchmark_arrays(device, z_elems, helem, Nq, FT; compile)

    cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem=z_elems, helem, Nq)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    cfield = fill(field_vars(FT), cspace)
    ffield = fill(field_vars(FT), fspace)
    benchmark_operators_base(trials, t_min, cfield, ffield, "sphere"; compile)

    # Tests are removed since they're flakey. And maintaining
    # them before they're converged is a bit of work..
    compile || test_results_sphere(t_min)
    return (; trials, t_min)
end

function benchmark_operators_base(trials, t_min, cfield, ffield, name; compile::Bool)
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
        op_LeftBiasedC2F!,
        op_LeftBiasedF2C!,
        op_RightBiasedC2F!,
        op_RightBiasedF2C!,
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
    ]

    @info "Benchmarking $name operators, this may take a minute or two..."
    for op in ops
        if uses_bycolumn(op) && axes(cfield) isa Spaces.FiniteDifferenceSpace
            continue
        end
        benchmark_func!(t_min, trials, op, cfield, ffield, #= verbose = =# false; compile)
    end

    return nothing
end

function test_results_column(t_min)
    # If these tests fail, just update the numbers (or the
    # buffer) so long its not an egregious regression.
    buffer = 2
    ns = 1
    μs = 10^3
    ms = 10^6
    results = [
    [(:FiniteDifferenceSpace, op_GradientF2C!, :none), 253.100*ns*buffer],
    [(:FiniteDifferenceSpace, op_GradientF2C!, :SetValue, :SetValue), 270.448*ns*buffer],
    [(:FiniteDifferenceSpace, op_GradientC2F!, :SetGradient, :SetGradient), 242.053*ns*buffer],
    [(:FiniteDifferenceSpace, op_GradientC2F!, :SetValue, :SetValue), 241.647*ns*buffer],
    [(:FiniteDifferenceSpace, op_DivergenceF2C!, :none), 1.005*μs*buffer],
    [(:FiniteDifferenceSpace, op_DivergenceF2C!, :Extrapolate, :Extrapolate), 1.076*μs*buffer],
    [(:FiniteDifferenceSpace, op_DivergenceC2F!, :SetDivergence, :SetDivergence), 878.028*ns*buffer],
    [(:FiniteDifferenceSpace, op_InterpolateF2C!, :none), 254.523*ns*buffer],
    [(:FiniteDifferenceSpace, op_InterpolateC2F!, :SetValue, :SetValue), 254.241*ns*buffer],
    [(:FiniteDifferenceSpace, op_InterpolateC2F!, :Extrapolate, :Extrapolate), 241.308*ns*buffer],
    [(:FiniteDifferenceSpace, op_broadcast_example2!, :none), 555.039*ns*buffer],
    [(:FiniteDifferenceSpace, op_LeftBiasedC2F!, :SetValue), 207.264*ns*buffer],
    [(:FiniteDifferenceSpace, op_LeftBiasedF2C!, :none), 137.031*ns*buffer],
    [(:FiniteDifferenceSpace, op_LeftBiasedF2C!, :SetValue), 185.135*ns*buffer],
    [(:FiniteDifferenceSpace, op_RightBiasedC2F!, :SetValue), 129.971*ns*buffer],
    [(:FiniteDifferenceSpace, op_RightBiasedF2C!, :none), 142.120*ns*buffer],
    [(:FiniteDifferenceSpace, op_RightBiasedF2C!, :SetValue), 141.446*ns*buffer],
    [(:FiniteDifferenceSpace, op_CurlC2F!, :SetCurl, :SetCurl), 1.692*μs*buffer],
    [(:FiniteDifferenceSpace, op_CurlC2F!, :SetValue, :SetValue), 1.616*μs*buffer],
    [(:FiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :SetValue, :SetValue), 754.856*ns*buffer],
    [(:FiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :Extrapolate, :Extrapolate), 765.401*ns*buffer],
    [(:FiniteDifferenceSpace, op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue), 2.540*μs*buffer],
    [(:FiniteDifferenceSpace, op_divgrad_CC!, :SetValue, :SetValue, :none), 924.147*ns*buffer],
    [(:FiniteDifferenceSpace, op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 876.510*ns*buffer],
    [(:FiniteDifferenceSpace, op_div_interp_CC!, :SetValue, :SetValue, :none), 721.119*ns*buffer],
    [(:FiniteDifferenceSpace, op_div_interp_FF!, :none, :SetValue, :SetValue), 686.581*ns*buffer],
    [(:FiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :Extrapolate), 4.960*μs*buffer],
    [(:FiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :SetValue), 5.047*μs*buffer],
    ]
    for (params, ref_time) in results
        if !(t_min[params] ≤ ref_time)
            @warn "Possible regression: $params, time=$(t_min[params]), ref_time=$ref_time"
        end
    end
end

function test_results_sphere(t_min)
    # If these tests fail, just update the numbers (or the
    # buffer) so long its not an egregious regression.
    buffer = 2
    ns = 1
    μs = 10^3
    ms = 10^6
    results = [
    [(:ExtrudedFiniteDifferenceSpace, op_GradientF2C!, :none), 1.746*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_GradientF2C!, :SetValue, :SetValue), 1.754*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_GradientC2F!, :SetGradient, :SetGradient), 1.899*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_GradientC2F!, :SetValue, :SetValue), 1.782*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_DivergenceF2C!, :none), 6.792*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_DivergenceF2C!, :Extrapolate, :Extrapolate), 6.776*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_DivergenceC2F!, :SetDivergence, :SetDivergence), 6.720*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_InterpolateF2C!, :none), 1.701*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_InterpolateC2F!, :SetValue, :SetValue), 1.713*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_InterpolateC2F!, :Extrapolate, :Extrapolate), 1.698*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_broadcast_example0!, :none), 1.059*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_broadcast_example1!, :none), 154.330*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_broadcast_example2!, :none), 152.689*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_LeftBiasedC2F!, :SetValue), 1.758*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_LeftBiasedF2C!, :none), 1.711*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_LeftBiasedF2C!, :SetValue), 1.754*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_RightBiasedC2F!, :SetValue), 1.847*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_RightBiasedF2C!, :none), 1.582*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_RightBiasedF2C!, :SetValue), 1.551*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_CurlC2F!, :SetCurl, :SetCurl), 4.669*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_CurlC2F!, :SetValue, :SetValue), 4.568*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :SetValue, :SetValue), 3.444*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :Extrapolate, :Extrapolate), 3.432*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue), 5.650*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_divgrad_CC!, :SetValue, :SetValue, :none), 4.474*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_divgrad_FF!, :none, :SetDivergence, :SetDivergence), 4.470*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_div_interp_CC!, :SetValue, :SetValue, :none), 3.566*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_div_interp_FF!, :none, :SetValue, :SetValue), 3.663*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :Extrapolate), 7.470*ms*buffer],
    [(:ExtrudedFiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :SetValue), 7.251*ms*buffer],
    ]
    for (params, ref_time) in results
        if !(t_min[params] ≤ ref_time)
            @warn "Possible regression: $params, time=$(t_min[params]), ref_time=$ref_time"
        end
    end
end

#! format: on
