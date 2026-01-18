import ClimaComms
ClimaComms.@import_required_backends
using Test
using LinearAlgebra
using OrdinaryDiffEqTsit5: ODEProblem, solve, Tsit5

import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces


convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "1D ∂T/t = α ∇²T ODE solve" begin

    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7)
    err, Δh = zeros(length(n_elems_seq)), zeros(length(n_elems_seq))
    device = ClimaComms.device()

    for (k, n) in enumerate(n_elems_seq)
        z₀ = Geometry.ZPoint(FT(0))
        z₁ = Geometry.ZPoint(FT(10))
        t₀ = FT(0)
        t₁ = FT(10)
        μ = FT(-1 / 2)
        ν = FT(5)
        𝓌 = FT(1)
        δ = FT(1)

        domain =
            Domains.IntervalDomain(z₀, z₁; boundary_names = (:bottom, :top))
        zp = (z₀.z + z₁.z / n / 2):(z₁.z / n):(z₁.z - z₁.z / n / 2)

        function gaussian(z, t; μ = -1 // 2, ν = 1, 𝓌 = 1, δ = 1)
            return exp(-(z - μ - 𝓌 * t)^2 / (4 * ν * (t + δ))) / sqrt(1 + t / δ)
        end

        function ∇gaussian(z, t; μ = -1 // 2, ν = 1, 𝓌 = 1, δ = 1)
            return -2 * (z - μ - 𝓌 * t) / (4 * ν * (δ + t)) *
                   exp(-(z - μ - 𝓌 * t)^2 / (4 * ν * (δ + t))) / sqrt(1 + t / δ)
        end

        z₀ = FT(0)
        z₁ = FT(10)

        mesh = Meshes.IntervalMesh(domain, nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.FaceFiniteDifferenceSpace(cs)
        zc = Fields.coordinate_field(cs)

        T = gaussian.(zc.z, -0; μ = μ, δ = δ, ν = ν, 𝓌 = 𝓌)
        V = ones(FT, fs)

        function ∑tendencies!(dT, T, z, t)
            bc_vb = Operators.SetValue(
                FT(gaussian(z₀, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ)),
            )
            bc_gt = Operators.SetGradient(
                FT(∇gaussian(z₁, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ)),
            )
            A = Operators.AdvectionC2C(
                bottom = bc_vb,
                top = Operators.Extrapolate(),
            )
            gradc2f = Operators.GradientC2F(; bottom = bc_vb, top = bc_gt)
            gradf2c = Operators.GradientF2C()
            return @. dT = gradf2c(ν * gradc2f(T)) - A(V, T)
        end

        # Solve the ODE operator
        Δt = 0.001
        prob = ODEProblem(∑tendencies!, T, (t₀, t₁))
        sol = solve(
            prob,
            Tsit5(),
            reltol = 1e-8,
            abstol = 1e-8,
            #dt = Δt,
        )
        computed_result = sol.u[end]
        analytical_result = gaussian.(zp, t₁; μ = μ, δ = δ, ν = ν, 𝓌 = 𝓌)
        Δh[k] = cs.Δh_c2c[1]
        err[k] =
            norm(parent(computed_result) .- analytical_result) /
            length(analytical_result)
    end
    conv = convergence_rate(err, Δh)
    # conv should be approximately 2 for second order-accurate stencil.
    @test 1.4 ≤ conv[1] ≤ 2.1
    @test 1.4 ≤ conv[2] ≤ 2.1
    @test conv[1] ≤ conv[2]
    @test err[3] ≤ err[2] ≤ err[1] ≤ 1e-2
end
