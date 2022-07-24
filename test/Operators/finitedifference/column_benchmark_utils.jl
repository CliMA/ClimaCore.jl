#! format: off
using Test
using StaticArrays, IntervalSets, LinearAlgebra
import BenchmarkTools
import StatsBase

import ClimaCore
ClimaCore.enable_threading() = false

import ClimaCore: Domains, Meshes, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

field_vars(::Type{FT}) where {FT} = (;
    x = FT(0),
    uₕ = Geometry.Covariant12Vector(FT(0), FT(0)),
    curluₕ = Geometry.Contravariant12Vector(FT(0), FT(0)),
    w = Geometry.Covariant3Vector(FT(0)),
    contra3 = Geometry.Contravariant3Vector(FT(0)),
    y = FT(0),
    D = FT(0),
    U = FT(0),
    ∇x = Geometry.Covariant3Vector(FT(0)),
)

function get_fields(n_elems, ::Type{FT}) where {FT}
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(pi);
        boundary_tags = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = n_elems)
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = getproperty(Fields.coordinate_field(cs), :z)
    zf = getproperty(Fields.coordinate_field(fs), :z)
    cfield = field_wrapper(cs, field_vars(FT))
    ffield = field_wrapper(fs, field_vars(FT))

    L = zeros(FT, n_elems)
    D = zeros(FT, n_elems)
    U = zeros(FT, n_elems)
    xarr = rand(FT, n_elems)
    uₕ_x = rand(FT, n_elems)
    uₕ_y = rand(FT, n_elems)
    yarr = rand(FT, n_elems + 1)
    vars_contig = (; L, D, U, xarr, yarr, uₕ_x, uₕ_y)

    return (; cfield, ffield, vars_contig)
end

function field_wrapper(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

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

function set_value_contra3_bcs(c)
    FT = Spaces.undertype(axes(c))
    contra3 = Geometry.Contravariant3Vector
    return (;bottom = Operators.SetValue(contra3(FT(0.0))),
             top = Operators.SetValue(contra3(FT(0.0))))
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
    cov12 = Geometry.Contravariant12Vector
    return (;bottom = Operators.SetCurl(cov12(FT(0), FT(0))),
             top = Operators.SetCurl(cov12(FT(0), FT(0))))
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

bcs_tested(c, ::typeof(op_GradientF2C!)) = ((), set_value_bcs(c))
bcs_tested(c, ::typeof(op_GradientC2F!)) = (set_gradient_value_bcs(c), set_value_bcs(c))
bcs_tested(c, ::typeof(op_DivergenceF2C!)) = ((), )
bcs_tested(c, ::typeof(op_DivergenceC2F!)) = (set_divergence_bcs(c),)
bcs_tested(c, ::typeof(op_InterpolateF2C!)) = ((), )
bcs_tested(c, ::typeof(op_InterpolateC2F!)) = (set_value_bcs(c),)
bcs_tested(c, ::typeof(op_LeftBiasedC2F!)) = (set_bot_value_bc(c),)
bcs_tested(c, ::typeof(op_LeftBiasedF2C!)) = ((), set_bot_value_bc(c))
bcs_tested(c, ::typeof(op_RightBiasedC2F!)) = (set_top_value_bc(c),)
bcs_tested(c, ::typeof(op_RightBiasedF2C!)) = ((), set_top_value_bc(c))
bcs_tested(c, ::typeof(op_CurlC2F!)) = (set_curl_bcs(c), )
bcs_tested(c, ::typeof(op_UpwindBiasedProductC2F!)) = (set_value_bcs(c), )
bcs_tested(c, ::typeof(op_Upwind3rdOrderBiasedProductC2F!)) = (set_upwind_biased_3_bcs(c), )

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

function benchmark_func!(trials, fun, c, f, show_bm = false)
    for bcs in bcs_tested(c, fun)
        key = (fun, bc_name(bcs)...)
        show_bm && @info "\n@benchmarking $key"
        trials[key] = BenchmarkTools.@benchmark $fun($c, $f, $bcs)
        show_bm && show(stdout, MIME("text/plain"), trials[key])
    end
end

function benchmark_cases(vars_contig, cfield, ffield)
    println("\n############################ 2-point stencil")
    (; L, D, U, xarr, yarr, uₕ_x, uₕ_y) = vars_contig
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

    ops = [
        #### Core discrete operators
        op_GradientF2C!,
        op_GradientC2F!,
        op_DivergenceF2C!,
        op_DivergenceC2F!,
        op_InterpolateF2C!,
        op_InterpolateC2F!,
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
    ]

    trials = Dict()
    @info "Benchmarking operators, this may take a minute or two..."
    for op in ops
        benchmark_func!(trials, op, cfield, ffield, #= show_bm = =# false)
    end
    # For configuring tests:
    t_ave = Dict()
    for key in keys(trials)
        trial = trials[key]
        t_ave[key] = BenchmarkTools.prettytime(StatsBase.mean(trial.times))
        @info "$(t_ave[key]) <=> t_ave[$key]"
    end

    @test_broken t_ave[(op_LeftBiasedF2C!, :none)] < 5e-7
    @test_broken t_ave[(op_DivergenceF2C!, :none)] < 5e-7
    @test_broken t_ave[(op_LeftBiasedF2C!, :SetValue)] < 5e-7
    @test_broken t_ave[(op_RightBiasedF2C!, :SetValue)] < 5e-7
    @test_broken t_ave[(op_RightBiasedC2F!, :SetValue)] < 5e-7
    @test_broken t_ave[(op_divgrad_FF!, :none, :SetDivergence, :SetDivergence)] < 5e-7
    @test_broken t_ave[(op_CurlC2F!, :SetCurl, :SetCurl)] < 5e-7
    @test_broken t_ave[(op_UpwindBiasedProductC2F!, :SetValue, :SetValue)] < 5e-7
    @test_broken t_ave[(op_GradientF2C!, :SetValue, :SetValue)] < 5e-7
    @test_broken t_ave[(op_InterpolateF2C!, :none)] < 5e-7
    @test_broken t_ave[(op_RightBiasedF2C!, :none)] < 5e-7
    @test_broken t_ave[(op_LeftBiasedC2F!, :SetValue)] < 5e-7
    @test_broken t_ave[(op_GradientF2C!, :none)] < 5e-7
    @test_broken t_ave[(op_divgrad_CC!, :SetValue, :SetValue, :none)] < 5e-7
    @test_broken t_ave[(op_div_interp_FF!, :none, :SetValue, :SetValue)] < 5e-7
    @test_broken t_ave[(op_InterpolateC2F!, :SetValue, :SetValue)] < 5e-7
    @test_broken t_ave[(op_GradientC2F!, :SetValue, :SetValue)] < 5e-7
    @test_broken t_ave[(op_GradientC2F!, :SetGradient, :SetGradient)] < 5e-7
    @test_broken t_ave[(op_div_interp_CC!, :SetValue, :SetValue, :none)] < 5e-7
    @test_broken t_ave[(op_DivergenceC2F!, :SetDivergence, :SetDivergence)] < 5e-7
    @test_broken t_ave[(op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue)] < 5e-7

    return nothing
end

#! format: on
