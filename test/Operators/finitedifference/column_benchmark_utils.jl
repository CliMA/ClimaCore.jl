#! format: off
using Test
using StaticArrays, IntervalSets, LinearAlgebra
import BenchmarkTools
import StatsBase
import OrderedCollections
using ClimaCore.Geometry: ⊗

import ClimaCore
ClimaCore.enable_threading() = false

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
)

function get_spaces(z_elems, ::Type{FT}) where {FT}
    quad = Spaces.Quadratures.GL{1}()
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(FT(0)),
        Geometry.XPoint(FT(1));
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(FT(0)),
        Geometry.YPoint(FT(1));
        periodic = true,
    )
    h_domain = Domains.RectangleDomain(x_domain, y_domain)
    h_mesh = Meshes.RectilinearMesh(h_domain, #=x_elem=# 1, #=y_elem=# 1)
    topology = Topologies.Topology2D(h_mesh)
    h_space = Spaces.SpectralElementSpace2D(topology, quad)

    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(pi);
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems = z_elems)
    z_topology = Topologies.IntervalTopology(z_mesh)
    z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    cs = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    fs = Spaces.FaceExtrudedFiniteDifferenceSpace(cs)
    return (;cs, fs)
end

function get_column_spaces(z_elems, ::Type{FT}) where {FT}
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(pi);
        boundary_tags = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = z_elems)
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = getproperty(Fields.coordinate_field(cs), :z)
    zf = getproperty(Fields.coordinate_field(fs), :z)
    cfield = field_wrapper(cs, field_vars(FT))
    ffield = field_wrapper(fs, field_vars(FT))

    return (;cs, fs)
end

function get_fields(z_elems, ::Type{FT}, has_h_space = false) where {FT}

    (; cs, fs) = if has_h_space
        get_spaces(z_elems, FT)
    else
        get_column_spaces(z_elems, FT)
    end
    zc = getproperty(Fields.coordinate_field(cs), :z)
    zf = getproperty(Fields.coordinate_field(fs), :z)
    cfield = field_wrapper(cs, field_vars(FT))
    ffield = field_wrapper(fs, field_vars(FT))
    return (; cfield, ffield)
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

function benchmark_func!(t_ave, trials, fun, c, f, has_h_space, verbose = false)
    for bcs in bcs_tested(c, fun)
        key = (has_h_space, fun, bc_name(bcs)...)
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

    @warn string(
        "The `set_value_divgrad_uₕ_maybe_field_bcs` bcs are different",
        "between `has_h_space = true` and `has_h_space = false`."
    )

    has_h_space = false
    (; cfield, ffield) = get_fields(1000, Float64, true)
    (; cfield, ffield) = get_fields(z_elems, FT, has_h_space)
    benchmark_operators_base(trials, t_ave, cfield, ffield, has_h_space)

    has_h_space = true
    (; cfield, ffield) = get_fields(z_elems, FT, has_h_space)
    benchmark_operators_base(trials, t_ave, cfield, ffield, has_h_space)
end

function benchmark_operators_base(trials, t_ave, cfield, ffield, has_h_space)
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
        op_divgrad_uₕ!,
    ]

    @info "Benchmarking operators, this may take a minute or two..."
    for op in ops
        benchmark_func!(t_ave, trials, op, cfield, ffield, has_h_space, #= verbose = =# false)
    end

    @test_broken t_ave[true, (op_GradientF2C!, :none)] < 500
    @test_broken t_ave[true, (op_GradientF2C!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[true, (op_GradientC2F!, :SetGradient, :SetGradient)] < 500
    @test_broken t_ave[true, (op_GradientC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[true, (op_DivergenceF2C!, :none)] < 500
    @test_broken t_ave[true, (op_DivergenceF2C!, :Extrapolate, :Extrapolate)] < 500
    @test_broken t_ave[true, (op_DivergenceC2F!, :SetDivergence, :SetDivergence)] < 500
    @test_broken t_ave[true, (op_InterpolateF2C!, :none)] < 500
    @test_broken t_ave[true, (op_InterpolateC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[true, (op_InterpolateC2F!, :Extrapolate, :Extrapolate)] < 500
    @test_broken t_ave[true, (op_LeftBiasedC2F!, :SetValue)] < 500
    @test_broken t_ave[true, (op_LeftBiasedF2C!, :none)] < 500
    @test_broken t_ave[true, (op_LeftBiasedF2C!, :SetValue)] < 500
    @test_broken t_ave[true, (op_RightBiasedC2F!, :SetValue)] < 500
    @test_broken t_ave[true, (op_RightBiasedF2C!, :none)] < 500
    @test_broken t_ave[true, (op_RightBiasedF2C!, :SetValue)] < 500
    @test_broken t_ave[true, (op_CurlC2F!, :SetCurl, :SetCurl)] < 500
    @test_broken t_ave[true, (op_CurlC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[true, (op_UpwindBiasedProductC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[true, (op_UpwindBiasedProductC2F!, :Extrapolate, :Extrapolate)] < 500
    @test_broken t_ave[true, (op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue)] < 500
    @test_broken t_ave[true, (op_divgrad_CC!, :SetValue, :SetValue, :none)] < 500
    @test_broken t_ave[true, (op_divgrad_FF!, :none, :SetDivergence, :SetDivergence)] < 500
    @test_broken t_ave[true, (op_div_interp_CC!, :SetValue, :SetValue, :none)] < 500
    @test_broken t_ave[true, (op_div_interp_FF!, :none, :SetValue, :SetValue)] < 500
    @test_broken t_ave[true, (op_divgrad_uₕ!, :none, :SetValue, :Extrapolate)] < 500
    @test_broken t_ave[true, (op_divgrad_uₕ!, :none, :SetValue, :SetValue)] < 500 # different with/without h_space

    @test_broken t_ave[false, (op_GradientF2C!, :none)] < 500
    @test_broken t_ave[false, (op_GradientF2C!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[false, (op_GradientC2F!, :SetGradient, :SetGradient)] < 500
    @test_broken t_ave[false, (op_GradientC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[false, (op_DivergenceF2C!, :none)] < 500
    @test_broken t_ave[false, (op_DivergenceF2C!, :Extrapolate, :Extrapolate)] < 500
    @test_broken t_ave[false, (op_DivergenceC2F!, :SetDivergence, :SetDivergence)] < 500
    @test_broken t_ave[false, (op_InterpolateF2C!, :none)] < 500
    @test_broken t_ave[false, (op_InterpolateC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[false, (op_InterpolateC2F!, :Extrapolate, :Extrapolate)] < 500
    @test_broken t_ave[false, (op_LeftBiasedC2F!, :SetValue)] < 500
    @test_broken t_ave[false, (op_LeftBiasedF2C!, :none)] < 500
    @test_broken t_ave[false, (op_LeftBiasedF2C!, :SetValue)] < 500
    @test_broken t_ave[false, (op_RightBiasedC2F!, :SetValue)] < 500
    @test_broken t_ave[false, (op_RightBiasedF2C!, :none)] < 500
    @test_broken t_ave[false, (op_RightBiasedF2C!, :SetValue)] < 500
    @test_broken t_ave[false, (op_CurlC2F!, :SetCurl, :SetCurl)] < 500
    @test_broken t_ave[false, (op_CurlC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[false, (op_UpwindBiasedProductC2F!, :SetValue, :SetValue)] < 500
    @test_broken t_ave[false, (op_UpwindBiasedProductC2F!, :Extrapolate, :Extrapolate)] < 500
    @test_broken t_ave[false, (op_divUpwind3rdOrderBiasedProductC2F!, :ThirdOrderOneSided, :ThirdOrderOneSided, :SetValue, :SetValue)] < 500
    @test_broken t_ave[false, (op_divgrad_CC!, :SetValue, :SetValue, :none)] < 500
    @test_broken t_ave[false, (op_divgrad_FF!, :none, :SetDivergence, :SetDivergence)] < 500
    @test_broken t_ave[false, (op_div_interp_CC!, :SetValue, :SetValue, :none)] < 500
    @test_broken t_ave[false, (op_div_interp_FF!, :none, :SetValue, :Extrapolate)] < 500
    @test_broken t_ave[false, (op_divgrad_uₕ!, :none, :SetValue, :SetValue)] < 500
    @test_broken t_ave[false, (op_divgrad_uₕ!, :none, :SetValue, :SetValue)] < 500 # different with/without h_space

    return nothing
end

#! format: on
