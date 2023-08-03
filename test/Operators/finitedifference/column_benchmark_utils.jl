#! format: off
using Test
using ClimaComms
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

#####
##### Second order interpolation / derivatives
#####

#= e.g., any 2nd order interpolation / derivative operator =#
function op_2mul_1add!(x, y, D, U)
    y1 = @view y[1:(end - 1)]
    y2 = @view y[2:end]
    @inbounds for i in eachindex(x)
        x[i] = D[i] * y1[i] + U[i] * y2[i]
    end
    return nothing
end

#= e.g., div(grad(scalar)), div(interp(vec)) =#
function op_3mul_2add!(x, y, L, D, U)
    y1 = @view y[1:(end - 1)]
    y2 = @view y[2:(end - 1)]
    y3 = @view y[2:end]
    @inbounds for i in eachindex(x)
        i==1 && continue
        i==length(x) && continue
        x[i] = L[i] * y1[i] + D[i] * y2[i] + U[i] * y3[i]
    end
    return nothing
end

#= e.g., curlC2F =#
function curl_like!(curluₕ, uₕ_x, uₕ_y, D, U)
    @inbounds for i in eachindex(curluₕ)
        curluₕ[i] = D[i] * uₕ_x[i] + U[i] * uₕ_y[i]
    end
    return nothing
end

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

include("column_benchmark_kernels.jl")

uses_bycolumn(::typeof(op_broadcast_example0!)) = true
uses_bycolumn(::typeof(op_broadcast_example1!)) = true
uses_bycolumn(::typeof(op_broadcast_example2!)) = false
uses_bycolumn(::Any) = false

bcs_tested(c, ::typeof(op_broadcast_example0!)) = ((), )
bcs_tested(c, ::typeof(op_broadcast_example1!)) = ((), )
bcs_tested(c, ::typeof(op_broadcast_example2!)) = ((), )

bcs_tested(c, ::typeof(op_GradientF2C!)) = ((), set_value_bcs(c))
bcs_tested(c, ::typeof(op_GradientC2F!)) = (set_gradient_value_bcs(c), set_value_bcs(c))
bcs_tested(c, ::typeof(op_DivergenceF2C!)) = ((), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_DivergenceC2F!)) = (set_divergence_bcs(c), )
bcs_tested(c, ::typeof(op_InterpolateF2C!)) = ((), )
bcs_tested(c, ::typeof(op_InterpolateC2F!)) = (set_value_bcs(c), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_LeftBiasedC2F!)) = (set_bot_value_bc(c),)
bcs_tested(c, ::typeof(op_LeftBiasedF2C!)) = ((), set_bot_value_bc(c))
bcs_tested(c, ::typeof(op_RightBiasedC2F!)) = (set_top_value_bc(c),)
bcs_tested(c, ::typeof(op_RightBiasedF2C!)) = ((), set_top_value_bc(c))
bcs_tested(c, ::typeof(op_CurlC2F!)) = (set_curl_bcs(c), set_curl_value_bcs(c))
bcs_tested(c, ::typeof(op_UpwindBiasedProductC2F!)) = (set_value_bcs(c), extrapolate_bcs(c))
bcs_tested(c, ::typeof(op_Upwind3rdOrderBiasedProductC2F!)) = (set_upwind_biased_3_bcs(c), extrapolate_bcs(c))

# Composed operators (bcs handled case-by-case)
bcs_tested(c, ::typeof(op_divUpwind3rdOrderBiasedProductC2F!)) =
    ((; inner = set_upwind_biased_3_bcs(c), outer = set_value_contra3_bcs(c)), )
bcs_tested(c, ::typeof(op_divgrad_CC!)) =
    ((; inner = set_value_bcs(c), outer = ()), )
bcs_tested(c, ::typeof(op_divgrad_FF!)) =
    ((; inner = (), outer = set_divergence_bcs(c)), )
bcs_tested(c, ::typeof(op_div_interp_CC!)) =
    ((; inner = set_value_contra3_bcs(c), outer = ()), )
bcs_tested(c, ::typeof(op_div_interp_FF!)) =
    ((; inner = (), outer = set_value_contra3_bcs(c)), )
bcs_tested(c, ::typeof(op_divgrad_uₕ!)) =
    (
        (; inner = (), outer = set_value_divgrad_uₕ_bcs(c)),
        (; inner = (), outer = set_value_divgrad_uₕ_maybe_field_bcs(c)),
    )

function benchmark_func!(t_ave, trials, fun, c, f, verbose = false)
    for bcs in bcs_tested(c, fun)
        h_space = nameof(typeof(axes(c)))
        key = (h_space, fun, bc_name(bcs)...)
        verbose && @info "\n@benchmarking $key"
        trials[key] = BenchmarkTools.@benchmark $fun($c, $f, $bcs)
        verbose && show(stdout, MIME("text/plain"), trials[key])

        t_ave[key] = StatsBase.mean(trials[key].times) # nano seconds
        t_pretty = BenchmarkTools.prettytime(t_ave[key])
        verbose || @info "$t_pretty <=> t_ave[$key]"
    end
end

function benchmark_arrays(z_elems, ::Type{FT}) where {FT}
    L = zeros(FT, z_elems)
    D = zeros(FT, z_elems)
    U = zeros(FT, z_elems)
    xarr = rand(FT, z_elems)
    uₕ_x = rand(FT, z_elems)
    uₕ_y = rand(FT, z_elems)
    yarr = rand(FT, z_elems + 1)

    println("\n############################ 2-point stencil")
    trial = BenchmarkTools.@benchmark op_2mul_1add!($xarr, $yarr, $D, $U)
    show(stdout, MIME("text/plain"), trial)
    println()
    println("\n############################ 3-point stencil")
    trial = BenchmarkTools.@benchmark op_3mul_2add!($xarr, $yarr, $L, $D, $U)
    show(stdout, MIME("text/plain"), trial)
    println()
    println("\n############################ curl-like stencil")
    trial = BenchmarkTools.@benchmark curl_like!($xarr, $uₕ_x, $uₕ_y, $D, $U)
    show(stdout, MIME("text/plain"), trial)
    println()
end

function benchmark_operators(z_elems, ::Type{FT}) where {FT}
    trials = OrderedCollections.OrderedDict()
    t_ave = OrderedCollections.OrderedDict()
    benchmark_arrays(z_elems, FT)

    cspace = TU.ColumnCenterFiniteDifferenceSpace(FT; zelem=z_elems)
    fspace = Spaces.FaceFiniteDifferenceSpace(cspace)
    cfield = fill(field_vars(FT), cspace)
    ffield = fill(field_vars(FT), fspace)
    benchmark_operators_base(trials, t_ave, cfield, ffield)

    cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem=z_elems)
    fspace = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace}(cspace)
    cfield = fill(field_vars(FT), cspace)
    ffield = fill(field_vars(FT), fspace)
    benchmark_operators_base(trials, t_ave, cfield, ffield)

    # Tests are removed since they're flakey. And maintaining
    # them before they're converged is a bit of work..
    # test_results(t_ave)
    return (; trials, t_ave)
end

function benchmark_operators_base(trials, t_ave, cfield, ffield)
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

    @info "Benchmarking operators, this may take a minute or two..."
    for op in ops
        if uses_bycolumn(op) && axes(cfield) isa Spaces.FiniteDifferenceSpace
            continue
        end
        benchmark_func!(t_ave, trials, op, cfield, ffield, #= verbose = =# false)
    end

    return nothing
end

function test_results(t_ave)
    # If these tests fail, just update the numbers (or the
    # buffer) so long its not an egregious regression.
    buffer = 2
    ns = 1
    μs = 10^3
    @test t_ave[(:FiniteDifferenceSpace, op_GradientF2C!, :none)] < 559.8423*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_GradientF2C!, :SetValue, :SetValue)] < 569.8995*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_GradientC2F!, :SetGradient, :SetGradient)] < 250.761*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_GradientC2F!, :SetValue, :SetValue)] < 248.521*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_DivergenceF2C!, :none)] < 1.550*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_DivergenceF2C!, :Extrapolate, :Extrapolate)] < 1.587*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_DivergenceC2F!, :SetDivergence, :SetDivergence)] < 1.565*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_InterpolateF2C!, :none)] < 332.432*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_InterpolateC2F!, :SetValue, :SetValue)] < 404.5473*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_InterpolateC2F!, :Extrapolate, :Extrapolate)] < 400.4656*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_broadcast_example2!, :none)] < 600*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_LeftBiasedC2F!, :SetValue)] < 365.2909*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_LeftBiasedF2C!, :none)] < 185.358*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_LeftBiasedF2C!, :SetValue)] < 221.175*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_RightBiasedC2F!, :SetValue)] < 138.649*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_RightBiasedF2C!, :none)] < 186.417*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_RightBiasedF2C!, :SetValue)] < 189.139*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_CurlC2F!, :SetCurl, :SetCurl)] < 2.884*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_CurlC2F!, :SetValue, :SetValue)] < 2.926*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :SetValue, :SetValue)] < 697.341*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :Extrapolate, :Extrapolate)] < 659.267*ns*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue)] < 4.483*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_divgrad_CC!, :SetValue, :SetValue, :none)] < 1.607*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_divgrad_FF!, :none, :SetDivergence, :SetDivergence)] < 1.529*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_div_interp_CC!, :SetValue, :SetValue, :none)] < 1.510*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_div_interp_FF!, :none, :SetValue, :SetValue)] < 1.523*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :Extrapolate)] < 4.637*μs*buffer
    @test t_ave[(:FiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :SetValue)] < 4.618*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_GradientF2C!, :none)] < 441.097*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_GradientF2C!, :SetValue, :SetValue)] < 724.818*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_GradientC2F!, :SetGradient, :SetGradient)] < 346.544*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_GradientC2F!, :SetValue, :SetValue)] < 327.835*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_DivergenceF2C!, :none)] < 1.884*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_DivergenceF2C!, :Extrapolate, :Extrapolate)] < 1.953*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_DivergenceC2F!, :SetDivergence, :SetDivergence)] < 1.858*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_InterpolateF2C!, :none)] < 436.229*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_InterpolateC2F!, :SetValue, :SetValue)] < 713.735*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_InterpolateC2F!, :Extrapolate, :Extrapolate)] < 808.127*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_broadcast_example0!, :none)] < 800*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_broadcast_example1!, :none)] < 39*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_broadcast_example2!, :none)] < 35*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_LeftBiasedC2F!, :SetValue)] < 619.749*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_LeftBiasedF2C!, :none)] < 276.520*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_LeftBiasedF2C!, :SetValue)] < 333.901*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_RightBiasedC2F!, :SetValue)] < 245.966*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_RightBiasedF2C!, :none)] < 277.616*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_RightBiasedF2C!, :SetValue)] < 280.969*ns*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_CurlC2F!, :SetCurl, :SetCurl)] < 3.078*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_CurlC2F!, :SetValue, :SetValue)] < 3.159*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :SetValue, :SetValue)] < 5.197*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :Extrapolate, :Extrapolate)] < 5.304*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue)] < 14.304*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_CC!, :SetValue, :SetValue, :none)] < 8.593*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_FF!, :none, :SetDivergence, :SetDivergence)] < 8.597*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_div_interp_CC!, :SetValue, :SetValue, :none)] < 1.735*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_div_interp_FF!, :none, :SetValue, :SetValue)] < 1.744*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :Extrapolate)] < 70.819*μs*buffer
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :SetValue)] < 72.569*μs*buffer

    # Broken tests
    @test_broken t_ave[(:FiniteDifferenceSpace, op_CurlC2F!, :SetCurl, :SetCurl)] < 500
    @test_broken t_ave[(:FiniteDifferenceSpace, op_CurlC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[(:FiniteDifferenceSpace, op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue)] < 500
    @test_broken t_ave[(:FiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :Extrapolate)] < 500
    @test_broken t_ave[(:FiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :SetValue)] < 500 # different with/without h_space
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_DivergenceF2C!, :none)] < 500
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_DivergenceF2C!, :Extrapolate, :Extrapolate)] < 500
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_DivergenceC2F!, :SetDivergence, :SetDivergence)] < 500
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_CurlC2F!, :SetCurl, :SetCurl)] < 500
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_CurlC2F!, :SetValue, :SetValue)] < 500
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :SetValue, :SetValue)] < 800
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_UpwindBiasedProductC2F!, :Extrapolate, :Extrapolate)] < 800
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue)] < 500
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_CC!, :SetValue, :SetValue, :none)] < 500
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_FF!, :none, :SetDivergence, :SetDivergence)] < 500
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_div_interp_CC!, :SetValue, :SetValue, :none)] < 800
    @test t_ave[(:ExtrudedFiniteDifferenceSpace, op_div_interp_FF!, :none, :SetValue, :SetValue)] < 800
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :Extrapolate)] < 500
    @test_broken t_ave[(:ExtrudedFiniteDifferenceSpace, op_divgrad_uₕ!, :none, :SetValue, :SetValue)] < 500 # different with/without h_space
end

#! format: on
