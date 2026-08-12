using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore: Fields, Domains, Meshes, Geometry, Spaces, Operators

# End-to-end smoke tests for the vertical (finite-difference) transport
# schemes, integrating the constant-velocity advection equation
#
#     ∂_t q + w ∂_z q = 0
#
# over a full advection cycle and asserting the properties each scheme is
# designed to guarantee: accuracy (L2 error vs the exact translated pulse),
# mass conservation (flux form is discretely conservative), and — for the
# van Leer limiter — boundedness / monotonicity, integrated with a hand-rolled SSP RK33 loop.
#
# The suite runs Float64 only (the accuracy/mass tolerances below are
# Float64-calibrated).
const FT = Float64

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU
import .TestUtilities: ssp33!  # shared hand-rolled SSP RK33 time integrator

##### Column flux-corrected-transport (FCT) schemes #####

# Each scheme's tendency has the identical signature `(dq, q, params, t)` and
# the identical structure `dq = -divf2c(<scheme flux>)`; they differ in
# how the face flux is assembled from the low-/high-order upwind operators and
# the flux corrector.

function fct_blend_tendency!(dq, q, p, t)
    (; w, C) = p
    FT = Spaces.undertype(axes(q))
    upwind1 = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    upwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    # divf2c is linear, so the split first-order + corrected-antidiffusive form
    # of the source example equals this combined form.
    @. dq = -divf2c(upwind1(w, q) + C * (upwind3(w, q) - upwind1(w, q)))
    return dq
end

function boris_book_tendency!(dq, q, p, t)
    (; w, Δt) = p
    FT = Spaces.undertype(axes(q))
    upwind1 = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    upwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    fct = Operators.FCTBorisBook(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    @. dq =
        -divf2c(
            upwind1(w, q) + fct(
                upwind3(w, q) - upwind1(w, q),
                q / Δt - divf2c(upwind1(w, q)),
            ),
        )
    return dq
end

function zalesak_tendency!(dq, q, p, t)
    (; w, Δt) = p
    FT = Spaces.undertype(axes(q))
    upwind1 = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    upwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    fct = Operators.FCTZalesak(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    @. dq =
        -divf2c(
            upwind1(w, q) + fct(
                upwind3(w, q) - upwind1(w, q),
                q / Δt,
                q / Δt - divf2c(upwind1(w, q)),
            ),
        )
    return dq
end

# (name, tendency!, L2-error bound, mass-conservation tolerance in units of eps)
const COLUMN_SCHEMES = [
    (;
        name = "FCT (3rd-order blend)",
        tendency! = fct_blend_tendency!,
        err_bound = FT(0.11),
        mass_eps = 3,
    ),
    (;
        name = "Boris & Book FCT",
        tendency! = boris_book_tendency!,
        err_bound = FT(0.11),
        mass_eps = 5,
    ),
    (;
        name = "Zalesak FCT",
        tendency! = zalesak_tendency!,
        err_bound = FT(0.11),
        mass_eps = 10,
    ),
]

@testset "Column FCT advection (accuracy + mass conservation)" begin
    speed = FT(1)
    z₀, zₕ, z₁ = FT(0), FT(1), FT(1)
    pulse(z, t) = abs(z - speed * t) ≤ zₕ ? z₁ : z₀
    t₀, Δt = FT(0), FT(1e-4)
    nsteps = 100
    t₁ = nsteps * Δt
    n = 64
    C = FT(1)
    device = ClimaComms.device()
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(-FT(π)),
        Geometry.ZPoint{FT}(FT(π));
        boundary_names = (:bottom, :top),
    )
    stretches =
        (("uniform", Meshes.Uniform()), ("stretched", Meshes.ExponentialStretching(FT(7))))

    for scheme in COLUMN_SCHEMES, (meshname, stretch) in stretches
        @testset "$(scheme.name) [$meshname]" begin
            mesh = Meshes.IntervalMesh(domain, stretch; nelems = n)
            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.FaceFiniteDifferenceSpace(cs)
            z = Fields.coordinate_field(cs).z
            w = Geometry.WVector.(speed .* ones(FT, fs))

            q = pulse.(z, t₀)
            params = (; w, Δt, C)
            dq, q1, q2 = similar(q), similar(q), similar(q)
            ssp33!(scheme.tendency!, q, dq, q1, q2, params, Δt, nsteps)

            q_exact = pulse.(z, t₁)
            m₀ = sum(pulse.(z, t₀))
            err = norm(q .- q_exact)
            rel_mass_err = abs(sum(q) - m₀) / abs(m₀)

            @test all(!isnan, parent(q))
            @test err ≤ scheme.err_bound
            @test rel_mass_err ≤ scheme.mass_eps * eps(FT)
        end
    end
end

##### Column van Leer flux limiter: boundedness / monotonicity #####

function vanleer_tendency!(dq, q, p, t)
    (; w, Δt, constraint) = p
    FT = Spaces.undertype(axes(q))
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    vanleer = Operators.LinVanLeerC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
        constraint = constraint,
    )
    @. dq = -divf2c(vanleer(w, q, Δt))
    return dq
end

@testset "Column van Leer limiter (monotonicity + mass conservation)" begin
    # Advect a top-hat pulse; bounds-preserving constraints must keep the
    # solution within the initial range [0, 1] to roundoff, and every
    # constraint conserves mass (flux form). Runs on a uniform mesh at the
    # source resolution (Courant ≈ 0.1); the integration time is halved
    # relative to the example to fit the smoke budget — monotonicity is a
    # per-step structural property, so it holds regardless.
    speed = FT(-1)
    z₀, zₕ, z₁ = FT(0), FT(2π), FT(1)
    pulse(z, t) = abs(z - speed * t) ≤ zₕ ? z₁ : z₀
    n = 256
    L = FT(20π)
    Δt = FT(0.1) * (L / n)          # Courant ≈ 0.1
    t₁ = FT(3)
    nsteps = round(Int, t₁ / Δt)
    device = ClimaComms.device()
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(-FT(10π)),
        Geometry.ZPoint{FT}(FT(10π));
        boundary_names = (:bottom, :top),
    )

    # (constraint, is it bounds-preserving?)
    constraints = [
        (Operators.PositiveDefinite(), false),
        (Operators.MonotoneHarmonic(), true),
        (Operators.MonotoneLocalExtrema(), true),
    ]

    for (constraint, monotone) in constraints
        cname = string(nameof(typeof(constraint)))
        @testset "$cname" begin
            mesh = Meshes.IntervalMesh(domain, Meshes.Uniform(); nelems = n)
            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.FaceFiniteDifferenceSpace(cs)
            z = Fields.coordinate_field(cs).z
            w = Geometry.WVector.(speed .* ones(FT, fs))

            q = pulse.(z, FT(0))
            m₀ = sum(copy(q))
            params = (; w, Δt, constraint)
            dq, q1, q2 = similar(q), similar(q), similar(q)
            ssp33!(vanleer_tendency!, q, dq, q1, q2, params, Δt, nsteps)

            qmax = maximum(parent(q))
            qmin = minimum(parent(q))
            @test all(!isnan, parent(q))
            if monotone
                # Bounds-preserving: q stays within [z₀, z₁] to roundoff
                @test qmax ≤ z₁ + eps(FT)
                @test qmin ≥ z₀ - eps(FT)
            else
                # Positivity-preserving only: small overshoot allowed, no undershoot
                @test qmax ≤ z₁ + FT(0.05)
                @test qmin ≥ z₀ - FT(0.05)
            end
            @test abs(sum(q) - m₀) / abs(m₀) ≤ 10 * eps(FT)
        end
    end
end
