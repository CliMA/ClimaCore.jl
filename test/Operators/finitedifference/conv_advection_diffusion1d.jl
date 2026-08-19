import ClimaComms
ClimaComms.@import_required_backends
using Test
using LinearAlgebra
import ClimaTimeSteppers as CTS

import ClimaCore:
    ClimaCore,
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces


@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities: convergence_rate

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
        V = Geometry.WVector.(ones(FT, fs))

        function ∑tendencies!(dT, T, z, t)
            bc_gb = Operators.SetGradient(
                Geometry.WVector(FT(∇gaussian(z₀, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ))),
            )
            bc_gt = Operators.SetGradient(
                Geometry.WVector(FT(∇gaussian(z₁, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ))),
            )
            top_center_left_biased_grad =
                Geometry.Covariant3Vector.(
                    Fields.level(T, Fields.nlevels(T)) .-
                    Fields.level(T, Fields.nlevels(T) - 1),
                )

            bc_gt_lb = Operators.SetGradient(top_center_left_biased_grad)
            gradc2f = Operators.GradientC2F(bottom = bc_gb, top = bc_gt)
            gradc2f_advect = Operators.GradientC2F(bottom = bc_gb, top = bc_gt_lb)
            interpf2c = Operators.InterpolateF2C()
            divf2c = Operators.DivergenceF2C()
            return @. dT =
                divf2c(ν * gradc2f(T)) -
                interpf2c(Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f_advect(T)))
        end

        # Solve the ODE operator. The explicit diffusive stability limit is
        # Δz²/(4ν) times SSPRK33's real-axis stability bound of 2.51, or
        # 7.7e-4 s on the finest mesh; the step below sits under that for
        # every mesh in the sequence. Its own third-order temporal error,
        # ~1e-10, is far below the spatial errors measured here, so what the
        # convergence rate reflects is the stencil.
        Δt = 5e-4
        prob = CTS.ODEProblem(
            CTS.ClimaODEFunction(; T_exp! = ∑tendencies!),
            T,
            (t₀, t₁),
            nothing,
        )
        sol = CTS.solve(
            prob,
            CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
            dt = Δt,
        )
        computed_result = sol.u[end]
        analytical_result = gaussian.(zp, t₁; μ = μ, δ = δ, ν = ν, 𝓌 = 𝓌)
        Δh[k] = (z₁ - z₀) / n
        err[k] =
            norm(parent(computed_result) .- analytical_result) /
            length(analytical_result)
    end
    conv = convergence_rate(err, Δh)
    # conv should be approximately 2 for second order-accurate stencil.
    @test 1.4 ≤ conv[1] ≤ 2.6
    @test 1.4 ≤ conv[2] ≤ 2.6
    @test err[3] ≤ err[2] ≤ err[1] ≤ 1e-2
end
