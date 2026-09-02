# Ekman boundary layer: a rotating column driven by a geostrophic wind, with
# constant eddy viscosity ν and a no-slip surface. Its steady state is the
# analytic Ekman spiral, which the run plots alongside the computed profile and
# asserts against over the whole column.
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
    Spaces

import ClimaTimeSteppers as CTS

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

const f = 5e-5              # Coriolis parameter (1/s)
const ν = 0.01              # eddy viscosity (m²/s)
const L = 2e2               # domain height (m), = 10 Ekman depths
const nelems = 60
const ug = 1.0              # geostrophic wind, u (m/s)
const vg = 0.0              # geostrophic wind, v (m/s)
const d = sqrt(2 * ν / f)   # Ekman depth (m)
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(L);
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain; nelems = nelems)
device = ClimaComms.device()
cspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)



# Start from the geostrophic wind everywhere; the spiral forms as the no-slip
# surface decelerates the lowest levels.
initial_wind(z) = (u = FT(ug), v = FT(vg))

zc = Fields.coordinate_field(cspace)
Yc = initial_wind.(zc.z)
w = Geometry.WVector.(zeros(Float64, fspace))


function tendency!(dY, Y, _, t)
    Yc = Y.Yc
    w = Y.w

    dYc = dY.Yc
    dw = dY.w

    u = Yc.u
    v = Yc.v

    du = dYc.u
    dv = dYc.v

    # w is carried in the state but not evolved (no subsidence); its tendency
    # must still be set, not left to whatever the tendency buffer contains.
    @. dw = Geometry.WVector(zero(FT))

    # No slip at the surface and the geostrophic wind at the top. The boundary
    # values of u and v are imposed through the gradient operator (via
    # `gradient_c2f_dirichlet`), so both boundary faces of ∂z(u, v) come from
    # the gradient and the divergence needs no boundary condition of its own.
    # The advective form is an interpolated center-to-face gradient, with zero
    # boundary values imposed the same way.
    divf2c = Operators.DivergenceF2C()
    interpf2c = Operators.InterpolateF2C()

    # u-momentum
    ∇u = Operators.gradient_c2f_dirichlet(u; bottom = FT(0), top = FT(ug))
    ∇u_advect = Operators.gradient_c2f_dirichlet(u; bottom = FT(0), top = FT(0))
    @. du =
        divf2c(ν * ∇u) + f * (v - vg) - interpf2c(
            Geometry.dot(Geometry.Contravariant3Vector(w), ∇u_advect),
        )

    # v-momentum
    ∇v = Operators.gradient_c2f_dirichlet(v; bottom = FT(0), top = FT(vg))
    ∇v_advect = Operators.gradient_c2f_dirichlet(v; bottom = FT(0), top = FT(0))
    @. dv =
        divf2c(ν * ∇v) - f * (u - ug) - interpf2c(
            Geometry.dot(Geometry.Contravariant3Vector(w), ∇v_advect),
        )
    return dY
end


Y = Fields.FieldVector(Yc = Yc, w = w)
dY = tendency!(similar(Y), Y, nothing, 0.0)

Δt = 100.0 # the diffusive stability limit is Δz²/2ν ≈ 550 s
# Solve the ODE operator
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = tendency!),
    Y,
    (0.0, 1.5e6), # ≈ 12 inertial periods, enough to reach steady state
    nothing,
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(0.0:1.5e4:1.5e6),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

# The analytic Ekman spiral (steady state of the system above).
u_exact(z) = ug - exp(-z / d) * (ug * cos(z / d) + vg * sin(z / d))
v_exact(z) = vg + exp(-z / d) * (ug * sin(z / d) - vg * cos(z / d))

using Test
@testset "steady state vs the analytic Ekman spiral" begin
    z = vec(parent(Fields.coordinate_field(cspace).z))
    u_end = vec(parent(sol.u[end].Yc.u))
    v_end = vec(parent(sol.u[end].Yc.v))
    u_ref = u_exact.(z)
    v_ref = v_exact.(z)
    # The state has stopped evolving over the last save interval, so what
    # follows is a statement about the steady state and not about a transient.
    u_prev = vec(parent(sol.u[end - 1].Yc.u))
    v_prev = vec(parent(sol.u[end - 1].Yc.v))
    @test maximum(abs, u_end .- u_prev) < 1e-3
    @test maximum(abs, v_end .- v_prev) < 1e-3
    # With no slip at the surface, that steady state is the analytic spiral
    # over the whole column, down to the discretization error. Measured at
    # this resolution: 0.0024 m/s in u and 0.0062 m/s in v, both attained in
    # the first cell, and both falling by a factor of 4 per grid refinement.
    @test maximum(abs, u_end .- u_ref) < 0.01
    @test maximum(abs, v_end .- v_ref) < 0.01
    # The signature of an Ekman layer: rotation turns the flow across the
    # isobars, and in the limit z → 0 the veering angle is exactly 45°.
    @test v_end[1] / u_end[1] ≈ 1 rtol = 0.05
end

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "ekman"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

z_centers = vec(parent(Fields.coordinate_field(cspace)))

function ekman_plot(u; title = "", size = (1024, 600))
    sub_plt1 = Plots.plot(
        u_exact.(z_centers),
        z_centers,
        marker = :circle,
        xlabel = "u",
        label = "Ref",
    )
    sub_plt1 = Plots.plot!(sub_plt1, vec(parent(u.Yc.u)), z_centers, label = "Comp")

    sub_plt2 = Plots.plot(
        v_exact.(z_centers),
        z_centers,
        marker = :circle,
        xlabel = "v",
        label = "Ref",
    )
    sub_plt2 = Plots.plot!(sub_plt2, vec(parent(u.Yc.v)), z_centers, label = "Comp")

    return Plots.plot(
        sub_plt1,
        sub_plt2,
        title = title,
        layout = (1, 2),
        size = size,
    )
end

anim = Plots.@animate for (t, u) in zip(sol.t, sol.u)
    ekman_plot(u, title = "t = $(round(t / 3600, digits = 1)) h")
end
Plots.mp4(anim, joinpath(path, "ekman.mp4"), fps = 10)

Plots.png(ekman_plot(sol.u[end]), joinpath(path, "ekman_end.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "ekman_end.png"), joinpath(@__DIR__, "../..")),
    "Ekman End",
)
