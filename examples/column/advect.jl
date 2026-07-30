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

import LazyBroadcast: lazy
using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33
using ClimaCorePlots
using Plots

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const FT = Float64

α = FT(0.1)

a_sin = FT(0.0)
b_sin = FT(4pi)
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(a_sin),
    Geometry.ZPoint{FT}(b_sin),
    boundary_names = (:left, :right),
)
mesh_sin = Meshes.IntervalMesh(domain, nelems = 128)

a_step = FT(-20.0)
b_step = FT(20.0)
domain = Domains.IntervalDomain(
    Geometry.ZPoint(a_step),
    Geometry.ZPoint(b_step),
    boundary_names = (:left, :right),
)
mesh_step = Meshes.IntervalMesh(domain, nelems = 64)

# `UpwindBiasedProductC2F` no longer takes a `SetValue` boundary condition, so
# evaluate its stencil on the boundary faces here, using the value of θ outside
# of the domain, and impose the result with a `SetBoundaryOperator`.
function upwind_boundary_operator(V, θ, a, b, t)
    lg_field = Fields.local_geometry_field(axes(V))
    face_bottom = Utilities.PlusHalf(0)
    face_top = Fields.nlevels(V) - Utilities.PlusHalf(0)
    v_left = Fields.field_values(
        Geometry.contravariant3.(
            Fields.level(V, face_bottom),
            Fields.level(lg_field, face_bottom),
        ),
    )[]
    v_right = Fields.field_values(
        Geometry.contravariant3.(
            Fields.level(V, face_top),
            Fields.level(lg_field, face_top),
        ),
    )[]
    θ_left = Fields.field_values(Fields.level(θ, 1))[]
    θ_right = Fields.field_values(Fields.level(θ, Fields.nlevels(θ)))[]
    return Operators.SetBoundaryOperator(;
        left = Operators.SetValue(
            Geometry.Contravariant3Vector(
                Operators.upwind_biased_product(v_left, sin(a - t), θ_left),
            ),
        ),
        right = Operators.SetValue(
            Geometry.Contravariant3Vector(
                Operators.upwind_biased_product(v_right, θ_right, sin(b - t)),
            ),
        ),
    )
end

# The gradient on the left boundary face is the one implied by θ = sin(-t)
# outside of the domain; on the right boundary face it is extrapolated from the
# closest interior faces.
function advection_gradient(θ, t)
    θ_1 = Fields.level(θ, 1)
    θ_n = Fields.level(θ, Fields.nlevels(θ))
    θ_nm1 = Fields.level(θ, Fields.nlevels(θ) - 1)
    return Operators.GradientC2F(
        left = Operators.SetGradient(
            @. lazy(Geometry.Covariant3Vector(2 * (θ_1 - sin(-t))))
        ),
        right = Operators.SetGradient(
            @. lazy(Geometry.Covariant3Vector(θ_n - θ_nm1))
        ),
    )
end

# The `Extrapolate` boundary condition of the removed `FluxCorrectionC2C`
# operator drops the term outside of the boundary, i.e. it sets the flux through
# the boundary face -- and hence the inner gradient there -- to zero.
flux_correction_gradient(::Type{FT}) where {FT} = Operators.GradientC2F(
    left = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    right = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
)

device = ClimaComms.device()
for (fn, mesh, a, b) in zip(
    ("sin", "step"),
    (mesh_sin, mesh_step),
    (a_sin, a_step),
    (b_sin, b_step),
)

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

    lg_field = Fields.local_geometry_field(fs)
    ∂ = Operators.DivergenceF2C()
    UB = Operators.UpwindBiasedProductC2F()
    gradf2c = Operators.GradientF2C()
    interpf2c = Operators.InterpolateF2C()
    gradc2f_fcc = flux_correction_gradient(FT)

    # the flux correction term, equivalent to the removed `FluxCorrectionC2C`
    # operator with `Extrapolate` boundary conditions
    fcc(θ) = @. lazy(
        adjoint(
            gradf2c(
                adjoint(gradc2f_fcc(θ)) * Geometry.Contravariant3Vector(
                    abs(Geometry.contravariant3(V, lg_field)),
                ),
            ),
        ) * Geometry.Contravariant3Vector(1),
    )

    # upwinding
    function tendency1!(dθ, θ, _, t)
        set_bcs = upwind_boundary_operator(V, θ, a, b, t)
        return @. dθ = -∂(set_bcs(UB(V, θ)))
    end
    # upwinding, with flux correction
    function tendency2!(dθ, θ, _, t)
        set_bcs = upwind_boundary_operator(V, θ, a, b, t)
        correction = fcc(θ)
        return @. dθ = -∂(set_bcs(UB(V, θ))) + correction
    end
    # advection, written as an interpolated center-to-face gradient
    function tendency3!(dθ, θ, _, t)
        gradc2f = advection_gradient(θ, t)
        return @. dθ =
            -interpf2c(
                Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f(θ)),
            )
    end
    # advection, with flux correction
    function tendency4!(dθ, θ, _, t)
        gradc2f = advection_gradient(θ, t)
        correction = fcc(θ)
        return @. dθ =
            -interpf2c(
                Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f(θ)),
            ) + correction
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
