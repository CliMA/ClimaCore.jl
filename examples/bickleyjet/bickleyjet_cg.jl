push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: slab, Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore: slab
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
)


domain = Domains.RectangleDomain(
    -2π..2π,
    -2π..2π,
    x1periodic = true,
    x2periodic = true,
)

n1, n2 = 16, 16
Nq = 4
Nqh = 7
mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

Iquad = Spaces.Quadratures.GLL{Nqh}()
Ispace = Spaces.SpectralElementSpace2D(grid_topology, Iquad)

function init_state(x, p)
    @unpack x1, x2 = x
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(x2)^(-2)

    # Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x1) * cos(p.k * x2)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(x2 + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (x2 + p.l / 10) / p.l^2 * cos(p.k * x1) * cos(p.k * x2)
    u₁′ += p.k * gaussian * cos(p.k * x1) * sin(p.k * x2)
    u₂′ = -p.k * gaussian * sin(p.k * x1) * cos(p.k * x2)


    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * x2)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, p)
    @unpack ρ, ρu, ρθ = state
    u = ρu ./ ρ
    return (
        ρ = ρu,
        ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * LinearAlgebra.I),
        ρθ = ρθ .* u,
    )
end

function energy(state, p)
    @unpack ρ, ρu = state
    u = ρu ./ ρ
    return ρ * (u.u1^2 + u.u2^2) / 2 + p.g * ρ^2 / 2
end

function total_energy(y, parameters)
    sum(state -> energy(state, parameters), y)
end


function rhs!(dydt, y, _, t)

    I = Operators.Interpolate(Ispace)
    div = Operators.WeakDivergence()
    R = Operators.Restrict(space)

    rparameters = Ref(parameters)
    @. dydt = -R(div(flux(I(y), rparameters)))

    Spaces.weighted_dss!(dydt)
    return dydt
end

# Next steps:
# 1. add the above to the design docs (divergence + over-integration + DSS)
# 2. add boundary conditions

dydt = similar(y0)
rhs!(dydt, y0, nothing, 0.0)


# Solve the ODE operator
prob = ODEProblem(rhs!, y0, (0.0, 80.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.02,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

using Plots

t = 30.0
y = sol.u[31]
plot(y.ρθ, clim = (-1, 1))

dydt = rhs!(similar(y), y, nothing, t)
plot(dydt.ρθ)


function invariant_form(y)
    @unpack ρ, ρu, ρθ = y
    return (ρ = ρ, u = ρu / ρ, ρθ = ρθ)
end

z = invariant_form.(y)

function flux_rhs!(dydt, y, _, t)

    I = Operators.Interpolate(Ispace)
    div = Operators.WeakDivergence()
    R = Operators.Restrict(space)

    rparameters = Ref(parameters)
    @. dydt = -R(div(flux(I(y), rparameters)))

    Spaces.weighted_dss!(dydt)
    return dydt
end


function invariant_rhs!(dydt, y, _, t)
    space = Fields.axes(y)

    I = Operators.Interpolate(Ispace)
    R = Operators.Restrict(space)


    div = Operators.WeakDivergence()
    grad = Operators.Gradient()
    curl = Operators.StrongCurl()

    @unpack g = parameters

    J = Fields.Field(Ispace.local_geometry.J, Ispace)


    @. dydt.ρ = -R(div(I(y.ρ) * I(y.u)))
    @. dydt.u =
        -R(
            grad(g * I(y.ρ) + norm(I(y.u))^2 / 2) +
            J * (I(y.u) × (curl(I(y.u)))),
        )
    @. dydt.ρθ = -R(div(I(y.ρθ) * I(y.u)))

    Spaces.weighted_dss!(dydt)

    return dydt
end

dzdt = invariant_rhs!(similar(z), z, nothing, t)
function flux_form_dt(dzdt, z)
    ρ = z.ρ
    u = z.u
    dρdt = dzdt.ρ
    dudt = dzdt.u
    dρθdt = dzdt.ρθ
    return (ρ = dρdt, ρu = ρ * dudt + dρdt * u, ρθ = dρθdt)
end


dydt_z = flux_form_dt.(dzdt, z)

function init_s(coord)
    ρ = 1.0
    θ = 1.0
    u = Cartesian12Vector(sin(coord.x1), cos(coord.x2))

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

ys = init_s.(Fields.coordinate_field(space))
dysdt = flux_rhs!(similar(ys), ys, nothing, 0)
zs = invariant_form.(ys)
dzsdt = invariant_rhs!(similar(zs), zs, nothing, 0)
dysdt_z = flux_form_dt.(dzsdt, zs)


coords = Fields.coordinate_field(space)

function velfield(coord)
    x = coord.x1
    y = coord.x2
    k1 = 0.1
    k2 = 0.3
    l1 = 0.5
    l2 = 0.7
    Cartesian12Vector(sin(k1 * x + l1 * y), cos(k2 * x + l2 * y))
end

v = velfield.(coords)
curlv = curl.(v)

function curlfield(coord)
    x = coord.x1
    y = coord.x2
    k1 = 0.1
    k2 = 0.3
    l1 = 0.5
    l2 = 0.7
    return -k2 * sin(k2 * x + l2 * y) - l1 * cos(k1 * x + l1 * y)
end
refcurlv = curlfield.(coords)

c = grad.(v.u2).u1 .- grad.(v.u1).u2

vccv = @. v × curlv
J = Fields.Field(space.local_geometry.J, space)
Jvccv = @. J * (v × curlv)
w = Cartesian12Vector.(Jvccv)

uc1 = v.u2 .* c
uc2 = .-v.u1 .* c


#=
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "cg"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/energy.png", "Total Energy")
=#
