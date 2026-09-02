# Divergence of vector spherical harmonics on the cubed sphere.
#
# The spheroidal and toroidal fields built in `utils_vsh_divergence.jl` give
# the divergence a pointwise-exact answer at every wavenumber,
#
#     div ∇Y_lm = -l(l+1)/a² Y_lm,     div (r̂ × ∇Y_lm) = 0,
#
# which resolves two things the single large-scale field in
# `conv_sphere_divergence.jl` cannot:
#  - max-norm accuracy as l rises toward the element resolution limit, and
#  - the inter-element discontinuity of the pre-DSS strong-form divergence,
#    which comparisons made after `weighted_dss!` average away.
# The toroidal cases are the sharpest probe of element-boundary oscillations:
# their divergence is identically zero, so the computed field is pure error.
#
# The split form is checked against the analytic product rule,
#
#     div(ψ ∇Y) = ∇ψ·∇Y - l(l+1)/a² ψ Y,     div(ψ r̂×∇Y) = ∇ψ·(r̂×∇Y),
#
# with ψ a spherical harmonic, so its gradient is analytic too.
#
# The weak- and split-form divergences are only checked after DSS: at
# element-boundary nodes each element holds only its own share of the
# mass-weighted sum (the split form contains weak-form terms), so their
# pre-DSS values there are incomplete by construction. The DG divergence
# completes the same weak volume term with a central interface flux instead
# of DSS, and is checked as assembled — no DSS anywhere.

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Domains, Meshes, Operators, Quadratures, Spaces, Topologies

include("utils_vsh_divergence.jl")
include("utils_dg.jl") # dg_divergence

# Odd Ne and even Nq keep every node off the poles, where the (u, v)
# components of a VSH are singular.
const Ne = 5
const Nq = 4

@testset "VSH divergence on the sphere [$FT]" for FT in (Float32, Float64)
    radius = FT(2)
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    topology = Topologies.Topology2D(ClimaComms.context(), mesh)
    space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{Nq}())
    positions = unit_sphere_positions(space)

    sdiv = Operators.Divergence()
    wdiv = Operators.Divergence{Operators.WeakForm}()
    split_div = Operators.SplitDivergence()
    (lψ, mψ) = (3, 2)
    ψ = ylm_field(lψ, mψ, space)

    # tol bounds the post-DSS max error of each (l, m), measured at this
    # resolution with ~2.5x margin; the pre-DSS jump gets 3 tol, since the
    # measured discontinuity runs ~3x the post-DSS error. Every check is
    # relative to l(l+1)/a² max|Y|, the size of the spheroidal divergence.
    # tol_split does the same for the split form, whose reference div(ψ u)
    # carries wavenumbers up to l + lψ; both split checks are relative to
    # max|div(ψ ∇Y)|, since the toroidal reference ∇ψ·(r̂×∇Y) is the small
    # residual of a computation on fields of that larger size. tol_dg bounds
    # the DG divergence, whose interface error the flux leaves in place
    # rather than averaging away.
    for (l, m, tol, tol_split, tol_dg) in (
        (2, 1, 0.012, 0.03, 0.02),
        (3, 0, 0.018, 0.06, 0.035),
        (5, 3, 0.045, 0.06, 0.1),
        (10, 7, 0.25, 0.25, 0.45),
    )
        tol = FT(tol)
        tol_split = FT(tol_split)
        tol_dg = FT(tol_dg)
        Y = ylm_field(l, m, space)
        exact = @. -l * (l + 1) / radius^2 * Y
        scale = l * (l + 1) / radius^2 * maximum(abs, parent(Y))

        S_uv = vsh_uv_field(spheroidal_uv, l, m, space, radius)
        T_uv = vsh_uv_field(toroidal_uv, l, m, space, radius)
        S = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), S_uv)
        T = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), T_uv)

        @testset "spheroidal (l = $l, m = $m)" begin
            div_s = sdiv.(S)
            @test max_interelement_jump(div_s, positions) ≤ 3 * tol * scale
            Spaces.weighted_dss!(div_s)
            @test maximum(abs, parent(div_s .- exact)) ≤ tol * scale
            div_w = wdiv.(S)
            Spaces.weighted_dss!(div_w)
            @test maximum(abs, parent(div_w .- exact)) ≤ tol * scale
        end

        @testset "toroidal (l = $l, m = $m)" begin
            div_s = sdiv.(T)
            @test max_interelement_jump(div_s, positions) ≤ 3 * tol * scale
            # Pre-DSS: the computed divergence is pure error.
            @test maximum(abs, parent(div_s)) ≤ tol * scale
            Spaces.weighted_dss!(div_s)
            @test maximum(abs, parent(div_s)) ≤ tol * scale
            div_w = wdiv.(T)
            Spaces.weighted_dss!(div_w)
            @test maximum(abs, parent(div_w)) ≤ tol * scale
        end

        @testset "split (l = $l, m = $m; ψ = Y_$(lψ)$(mψ))" begin
            exact_S =
                vsh_grad_dot_field(spheroidal_uv, l, m, lψ, mψ, space, radius) .+
                ψ .* exact
            exact_T =
                vsh_grad_dot_field(toroidal_uv, l, m, lψ, mψ, space, radius)
            scale_split = maximum(abs, parent(exact_S))

            div_S = split_div.(S, ψ)
            Spaces.weighted_dss!(div_S)
            @test maximum(abs, parent(div_S .- exact_S)) ≤ tol_split * scale_split
            div_T = split_div.(T, ψ)
            Spaces.weighted_dss!(div_T)
            @test maximum(abs, parent(div_T .- exact_T)) ≤ tol_split * scale_split
        end

        @testset "DG (l = $l, m = $m)" begin
            div_S = dg_divergence(S_uv)
            @test maximum(abs, parent(div_S .- exact)) ≤ tol_dg * scale
            div_T = dg_divergence(T_uv)
            @test maximum(abs, parent(div_T)) ≤ tol_dg * scale
        end
    end
end
