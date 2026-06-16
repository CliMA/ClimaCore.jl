import ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
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

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

a = Geometry.ZPoint(FT(-20.0))
b = Geometry.ZPoint(FT(20.0))
n = 64
α = FT(0.1)

function heaviside(pt)
    0.5 * (sign(pt.z) + 1)
end

domain = Domains.IntervalDomain(a, b, boundary_names = (:left, :right))
mesh = Meshes.IntervalMesh(domain, nelems = n)
device = ClimaComms.device()
cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)

V = Geometry.WVector.(ones(FT, fs))
θ = heaviside.(Fields.coordinate_field(cs))

# Solve advection Equation: ∂θ/dt = -∂(vθ)

# upwinding
function tendency1!(dθ, θ, _, t)
    lg_field = Fields.local_geometry_field(fs)
    lg_left = Fields.level(lg_field, Utilities.PlusHalf(0))
    lg_right = Fields.level(lg_field, Fields.nlevels(lg_field) - Utilities.PlusHalf(0))
    v_left = Fields.field_values(
        Geometry.contravariant3.(Fields.level(V, Utilities.PlusHalf(0)), lg_left),
    )[]
    aᴸᴮ = sin(a.z - t)
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
    aᴿᴮ = sin(b.z - t)
    aᴿ = Fields.field_values(Fields.level(θ, Fields.nlevels(θ)))[]
    right_bc = Operators.SetValue(
        Geometry.Contravariant3Vector(Operators.upwind_biased_product(v_right, aᴿ, aᴿᴮ)),
    )
    set_bcs = Operators.SetBoundaryOperator(; left = left_bc, right = right_bc)
    UB = Operators.UpwindBiasedProductC2F()
    ∂ = Operators.DivergenceF2C()

    return @. dθ = -∂(set_bcs(UB(V, θ)))
end
function tendency2!(dθ, θ, _, t)
    fcc = Operators.FluxCorrectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    lg_field = Fields.local_geometry_field(fs)
    lg_left = Fields.level(lg_field, Utilities.PlusHalf(0))
    lg_right = Fields.level(lg_field, Fields.nlevels(lg_field) - Utilities.PlusHalf(0))
    v_left = Fields.field_values(
        Geometry.contravariant3.(Fields.level(V, Utilities.PlusHalf(0)), lg_left),
    )[]
    aᴸᴮ = sin(a.z - t)
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
    aᴿᴮ = sin(b.z - t)
    aᴿ = Fields.field_values(Fields.level(θ, Fields.nlevels(θ)))[]
    right_bc = Operators.SetValue(
        Geometry.Contravariant3Vector(Operators.upwind_biased_product(v_right, aᴿ, aᴿᴮ)),
    )
    set_bcs = Operators.SetBoundaryOperator(; left = left_bc, right = right_bc)
    UB = Operators.UpwindBiasedProductC2F()
    ∂ = Operators.DivergenceF2C()
    return @. dθ = -∂(set_bcs(UB(V, θ))) + fcc(V, θ)
end
# use the advection operator
function tendency3!(dθ, θ, _, t)
    left_center = Fields.level(θ, 1)
    right_center_left_biased_grad =
        Geometry.Covariant3Vector.(
            Fields.level(θ, Fields.nlevels(θ)) .- Fields.level(θ, Fields.nlevels(θ) - 1)
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

    fcc = Operators.FluxCorrectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    left_center = Fields.level(θ, 1)
    right_center_left_biased_grad =
        Geometry.Covariant3Vector.(
            Fields.level(θ, Fields.nlevels(θ)) .- Fields.level(θ, Fields.nlevels(θ) - 1)
        )
    left_gradient =
        Operators.SetGradient(@. Geometry.Covariant3Vector(2 * (left_center - sin(-t))))
    right_gradient = Operators.SetGradient(right_center_left_biased_grad)
    gradc2f = Operators.GradientC2F(left = left_gradient, right = right_gradient)
    interpf2c = Operators.InterpolateF2C()
    return @. dθ =
        -1 * interpf2c(Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f(θ))) +
        fcc(V, θ)
end

# use the advection operator

@show tendency1!(similar(θ), θ, nothing, 0.0)
# Solve the ODE operator
Δt = 0.001
prob1 = ODEProblem(tendency1!, θ, (0.0, 5.0))
prob2 = ODEProblem(tendency2!, θ, (0.0, 5.0))
prob3 = ODEProblem(tendency3!, θ, (0.0, 5.0))
prob4 = ODEProblem(tendency4!, θ, (0.0, 5.0))
sol1 = solve(
    prob1,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):5.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);
sol2 = solve(
    prob2,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):5.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);
sol3 = solve(
    prob3,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):5.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);
sol4 = solve(
    prob4,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):5.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import ClimaCorePlots, Plots
Plots.GRBackend()

dir = "advect_step_function"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol1.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "UBP_advect_step_function.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol1.u[end], xlim = (-1, 1)),
    joinpath(path, "sol1_advect_step_function_end.png"),
)

anim = Plots.@animate for u in sol2.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "UBP_advect_step_function_fc.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol2.u[end], xlim = (-1, 1)),
    joinpath(path, "sol2_advect_step_function_end.png"),
)

anim = Plots.@animate for u in sol3.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "C2C_advect_step_function.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol3.u[end], xlim = (-1, 1)),
    joinpath(path, "sol3_advect_step_function_end.png"),
)

anim = Plots.@animate for u in sol4.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "C2C_advect_step_function_fc.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol4.u[end], xlim = (-1, 1)),
    joinpath(path, "sol4_advect_step_function_end.png"),
)

p = Plots.plot(sol1.u[end], xlim = (-1, 1), label = "UBP")
p = Plots.plot!(sol2.u[end], xlim = (-1, 1), label = "UBP_FC")
p = Plots.plot!(sol3.u[end], xlim = (-1, 1), label = "C2C")
p = Plots.plot!(sol4.u[end], xlim = (-1, 1), label = "C2C_FC")
Plots.png(p, joinpath(path, "all_advect_step_function_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    "examples/column/output/$(dir)/advect_step_function_end.png",
    "Advect End Simulation",
)

linkfig(
    relpath(
        joinpath(path, "advect_step_function_end.png"),
        joinpath(@__DIR__, "../.."),
    ),
    "Advect End Simulation",
)
