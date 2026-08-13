# Ekman boundary layer: a rotating column driven by a geostrophic wind, with
# constant eddy viscosity ν and a drag law at the surface. Its steady state is
# the analytic Ekman spiral, which the run plots alongside the computed profile
# so the two can be compared.
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
const L = 2e2               # domain height (m)
const nelems = 30
const Cd = ν / (L / nelems) # drag coefficient
const ug = 1.0              # geostrophic wind, u (m/s)
const vg = 0.0              # geostrophic wind, v (m/s)
const d = sqrt(2 * ν / f)   # Ekman depth (m)
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(L);
    boundary_names = (:bottom, :top),
)
#mesh = Meshes.IntervalMesh(domain, Meshes.ExponentialStretching(7.5e3); nelems = 30)
mesh = Meshes.IntervalMesh(domain; nelems = nelems)
device = ClimaComms.device()
cspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)



# Start from the geostrophic wind everywhere; the spiral forms as surface drag
# decelerates the lowest levels.
initial_wind(z) = (u = FT(ug), v = FT(vg))

zc = Fields.coordinate_field(cspace)
Yc = initial_wind.(zc.z)
w = Geometry.WVector.(zeros(Float64, fspace))

Y_init = copy(Yc)
w_init = copy(w)

function tendency!(dY, Y, _, t)
    Yc = Y.Yc
    w = Y.w

    dYc = dY.Yc
    dw = dY.w

    u = Yc.u
    v = Yc.v

    du = dYc.u
    dv = dYc.v

    # S 4.4.1: potential temperature density
    # Mass conservation

    # w is carried in the state but not evolved (no subsidence); its tendency
    # must still be set, not left to whatever the tendency buffer contains.
    @. dw = Geometry.WVector(zero(FT))

    u_1 = parent(u)[1]
    v_1 = parent(v)[1]
    u_wind = sqrt(u_1^2 + v_1^2)
    A = Operators.AdvectionC2C(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )

    # u-momentum
    bcs_bottom = Operators.SetValue(Geometry.WVector(Cd * u_wind * u_1))  # Eq. 4.16
    bcs_top = Operators.SetValue(FT(ug))  # Eq. 4.18
    gradc2f = Operators.GradientC2F(top = bcs_top)
    divf2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. du = divf2c(ν * gradc2f(u)) + f * (v - vg) - A(w, u)   # Eq. 4.8

    # v-momentum
    bcs_bottom = Operators.SetValue(Geometry.WVector(Cd * u_wind * v_1))  # Eq. 4.17
    bcs_top = Operators.SetValue(FT(vg))  # Eq. 4.19
    gradc2f = Operators.GradientC2F(top = bcs_top)
    divf2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. dv = divf2c(ν * gradc2f(v)) - f * (u - ug) - A(w, v)   # Eq. 4.9
    return dY
end


Y = Fields.FieldVector(Yc = Yc, w = w)
dY = tendency!(similar(Y), Y, nothing, 0.0)

Δt = 2.0
ndays = 0
# Solve the ODE operator
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = tendency!),
    Y,
    (0.0, 60 * 60 * 50),
    nothing,
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(0.0:600:(60 * 60 * 50)), # save 10 min
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "ekman"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

z_centers = vec(parent(Fields.coordinate_field(cspace)))
z_faces = vec(parent(Fields.coordinate_field(fspace)))

function ekman_plot(u; title = "", size = (1024, 600))
    u_ref =
        ug .-
        exp.(-z_centers / d) .*
        (ug * cos.(z_centers / d) + vg * sin.(z_centers / d))
    sub_plt1 = Plots.plot(
        u_ref,
        z_centers,
        marker = :circle,
        xlabel = "u",
        label = "Ref",
    )
    sub_plt1 = Plots.plot!(sub_plt1, vec(parent(u.Yc.u)), z_centers, label = "Comp")

    v_ref =
        vg .+
        exp.(-z_centers / d) .*
        (ug * sin.(z_centers / d) - vg * cos.(z_centers / d))
    sub_plt2 = Plots.plot(
        v_ref,
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

anim = Plots.@animate for (i, u) in enumerate(sol.u)
    ekman_plot(u, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "ekman.mp4"), fps = 10)

Plots.png(ekman_plot(sol.u[end]), joinpath(path, "ekman_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "ekman_end.png"), joinpath(@__DIR__, "../..")),
    "Ekman End",
)
