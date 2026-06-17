import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces,
    Utilities

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33
using ClimaCorePlots
using Plots

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const FT = Float64

a = FT(0.0)
b = FT(4pi)
n = 128
α = FT(0.1)

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(a),
    Geometry.ZPoint{FT}(b),
    boundary_names = (:left, :right),
)
mesh_sin = Meshes.IntervalMesh(domain, nelems = n)

a = FT(-20.0)
b = FT(20.0)
n = 64
α = FT(0.1)
domain = Domains.IntervalDomain(
    Geometry.ZPoint(a),
    Geometry.ZPoint(b),
    boundary_names = (:left, :right),
)
mesh_step = Meshes.IntervalMesh(domain, nelems = n)
device = ClimaComms.device()
for (fn, mesh) in zip(("sin", "step"), (mesh_sin, mesh_step))

    cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)

    V = Geometry.WVector.(ones(FT, fs))
    if fn == "sin"
        θ = sin.(Fields.coordinate_field(cs).z)
    elseif fn == "step"
        function heaviside(pt)
            0.5 * (sign(pt.z) + 1)
        end
        θ = heaviside.(Fields.coordinate_field(cs))
    end

    # Solve advection Equation: ∂θ/dt = -∂(vθ)

    # upwinding
    function tendency1!(dθ, θ, _, t)
        lg_field = Fields.local_geometry_field(fs)
        lg_left = Fields.level(lg_field, Utilities.PlusHalf(0))
        lg_right = Fields.level(lg_field, Fields.nlevels(lg_field) - Utilities.PlusHalf(0))
        v_left = Fields.field_values(
            Geometry.contravariant3.(Fields.level(V, Utilities.PlusHalf(0)), lg_left),
        )[]
        aᴸᴮ = sin(a - t)
        aᴸ = Fields.field_values(Fields.level(θ, 1))[]
        left_bc = Operators.SetValue(
            Geometry.Contravariant3Vector(Operators.upwind_biased_product(v_left, aᴸᴮ, aᴸ)),
        )
        v_right = Fields.field_values(
            Geometry.contravariant3.(
                Fields.level(V, Fields.nlevels(V) - Utilities.PlusHalf(0)),
                lg_right,
            ),
        )[]
        aᴿᴮ = sin(b - t)
        aᴿ = Fields.field_values(Fields.level(θ, Fields.nlevels(θ)))[]
        right_bc = Operators.SetValue(
            Geometry.Contravariant3Vector(
                Operators.upwind_biased_product(v_right, aᴿ, aᴿᴮ),
            ),
        )
        set_bcs = Operators.SetBoundaryOperator(; left = left_bc, right = right_bc)
        UB = Operators.UpwindBiasedProductC2F()
        ∂ = Operators.DivergenceF2C()

        return @. dθ = -∂(set_bcs(UB(V, θ)))
    end
    function tendency2!(dθ, θ, _, t)
        lg_field = Fields.local_geometry_field(fs)
        lg_left = Fields.level(lg_field, Utilities.PlusHalf(0))
        lg_right = Fields.level(lg_field, Fields.nlevels(lg_field) - Utilities.PlusHalf(0))
        v_left = Fields.field_values(
            Geometry.contravariant3.(Fields.level(V, Utilities.PlusHalf(0)), lg_left),
        )[]
        aᴸᴮ = sin(a - t)
        aᴸ = Fields.field_values(Fields.level(θ, 1))[]
        left_bc = Operators.SetValue(
            Geometry.Contravariant3Vector(Operators.upwind_biased_product(v_left, aᴸᴮ, aᴸ)),
        )
        v_right = Fields.field_values(
            Geometry.contravariant3.(
                Fields.level(V, Fields.nlevels(V) - Utilities.PlusHalf(0)),
                lg_right,
            ),
        )[]
        aᴿᴮ = sin(b - t)
        aᴿ = Fields.field_values(Fields.level(θ, Fields.nlevels(θ)))[]
        right_bc = Operators.SetValue(
            Geometry.Contravariant3Vector(
                Operators.upwind_biased_product(v_right, aᴿ, aᴿᴮ),
            ),
        )
        set_bcs = Operators.SetBoundaryOperator(; left = left_bc, right = right_bc)
        UB = Operators.UpwindBiasedProductC2F()
        ∂ = Operators.DivergenceF2C()
        left_center = Fields.level(θ, 1)
        right_center_left_biased_grad =
            Geometry.Covariant3Vector.(
                Fields.level(θ, Fields.nlevels(θ)) .-
                Fields.level(θ, Fields.nlevels(θ) - 1)
            )
        right_gradient_extrapolate = Operators.SetGradient(right_center_left_biased_grad)
        left_gradient_extrapolate = Operators.SetGradient(
            Geometry.Covariant3Vector.(Fields.level(θ, 2) .- left_center),
        )
        gradc2f_fcc = Operators.GradientC2F(
            left = left_gradient_extrapolate,
            right = right_gradient_extrapolate,
        )
        gradf2c = Operators.GradientF2C()
        return @. dθ =
            -∂(set_bcs(UB(V, θ))) +
            parent(
                gradf2c(Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f_fcc(θ))),
            ).data.:1
    end
    # use the advection operator
    function tendency3!(dθ, θ, _, t)
        left_center = Fields.level(θ, 1)
        right_center_left_biased_grad =
            Geometry.Covariant3Vector.(
                Fields.level(θ, Fields.nlevels(θ)) .-
                Fields.level(θ, Fields.nlevels(θ) - 1)
            )
        left_gradient =
            Operators.SetGradient(@. Geometry.Covariant3Vector(2 * (left_center - sin(-t))))
        right_gradient = Operators.SetGradient(right_center_left_biased_grad)
        gradc2f = Operators.GradientC2F(left = left_gradient, right = right_gradient)
        interpf2c = Operators.InterpolateF2C()
        return @. dθ =
            -1 * interpf2c(Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f(θ)))
    end
    # use the advection operator
    function tendency4!(dθ, θ, _, t)
        left_center = Fields.level(θ, 1)
        right_center_left_biased_grad =
            Geometry.Covariant3Vector.(
                Fields.level(θ, Fields.nlevels(θ)) .-
                Fields.level(θ, Fields.nlevels(θ) - 1)
            )
        left_gradient =
            Operators.SetGradient(@. Geometry.Covariant3Vector(2 * (left_center - sin(-t))))
        right_gradient_extrapolate = Operators.SetGradient(right_center_left_biased_grad)
        gradc2f =
            Operators.GradientC2F(left = left_gradient, right = right_gradient_extrapolate)
        interpf2c = Operators.InterpolateF2C()
        left_gradient_extrapolate = Operators.SetGradient(
            Geometry.Covariant3Vector.(Fields.level(θ, 2) .- left_center),
        )
        gradc2f_fcc = Operators.GradientC2F(
            left = left_gradient_extrapolate,
            right = right_gradient_extrapolate,
        )
        gradf2c = Operators.GradientF2C()
        return @. dθ =
            -1 * interpf2c(Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f(θ))) +
            parent(
                gradf2c(Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f_fcc(θ))),
            ).data.:1
    end

    # use the advection operator

    @show tendency1!(similar(θ), θ, nothing, 0.0)
    # Solve the ODE operator
    Δt = 0.001
    t_end = fn == "sin" ? 10.0 : 5.0
    prob1 = ODEProblem(tendency1!, θ, (0.0, t_end))
    prob2 = ODEProblem(tendency2!, θ, (0.0, t_end))
    prob3 = ODEProblem(tendency3!, θ, (0.0, t_end))
    prob4 = ODEProblem(tendency4!, θ, (0.0, t_end))
    sol1 = solve(
        prob1,
        SSPRK33(),
        dt = Δt,
        saveat = collect(0.0:(10 * Δt):t_end),
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
    sol2 = solve(
        prob2,
        SSPRK33(),
        dt = Δt,
        saveat = collect(0.0:(10 * Δt):t_end),
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
    sol3 = solve(
        prob3,
        SSPRK33(),
        dt = Δt,
        saveat = collect(0.0:(10 * Δt):t_end),
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
    sol4 = solve(
        prob4,
        SSPRK33(),
        dt = Δt,
        saveat = collect(0.0:(10 * Δt):t_end),
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )

    ENV["GKSwstype"] = "nul"

    Plots.GRBackend()

    sim_type = fn == "sin" ? "advect" : "advect_step_function"
    dir = sim_type
    path = joinpath(@__DIR__, "output", dir)
    mkpath(path)

    anim = Plots.@animate for u in sol1.u
        Plots.plot(u, xlim = (-1, 1))
    end
    Plots.mp4(anim, joinpath(path, "UBP_$(sim_type).mp4"), fps = 10)
    Plots.png(
        Plots.plot(sol1.u[end], xlim = (-1, 1)),
        joinpath(path, "sol1_$(sim_type)_end.png"),
    )

    anim = Plots.@animate for u in sol2.u
        Plots.plot(u, xlim = (-1, 1))
    end
    Plots.mp4(anim, joinpath(path, "UBP_$(sim_type)_fc.mp4"), fps = 10)
    Plots.png(
        Plots.plot(sol2.u[end], xlim = (-1, 1)),
        joinpath(path, "sol2_$(sim_type)_end.png"),
    )

    anim = Plots.@animate for u in sol3.u
        Plots.plot(u, xlim = (-1, 1))
    end
    Plots.mp4(anim, joinpath(path, "C2C_$(sim_type).mp4"), fps = 10)
    Plots.png(
        Plots.plot(sol3.u[end], xlim = (-1, 1)),
        joinpath(path, "sol3_$(sim_type)_end.png"),
    )

    anim = Plots.@animate for u in sol4.u
        Plots.plot(u, xlim = (-1, 1))
    end
    Plots.mp4(anim, joinpath(path, "C2C_$(sim_type)_fc.mp4"), fps = 10)
    Plots.png(
        Plots.plot(sol4.u[end], xlim = (-1, 1)),
        joinpath(path, "sol4_$(sim_type)_end.png"),
    )

    p = Plots.plot(sol1.u[end], xlim = (-1, 1), ls = :dash, label = "UBP")
    p = Plots.plot!(sol2.u[end], xlim = (-1, 1), ls = :dot, label = "UBP_FC")
    p = Plots.plot!(sol3.u[end], xlim = (-1, 1), ls = :solid, label = "C2C")
    p = Plots.plot!(sol4.u[end], xlim = (-1, 1), ls = :dashdot, label = "C2C_FC")
    Plots.png(p, joinpath(path, "all_$(sim_type)_end.png"))

    function linkfig(figpath, alt = "")
        # buildkite-agent upload figpath
        # link figure in logs if we are running on CI
        if get(ENV, "BUILDKITE", "") == "true"
            artifact_url = "artifact://$figpath"
            print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
        end
    end

    linkfig(
        relpath(joinpath(path, "$(sim_type)_end.png"), joinpath(@__DIR__, "../..")),
        "Advect End Simulation",
    )
end
