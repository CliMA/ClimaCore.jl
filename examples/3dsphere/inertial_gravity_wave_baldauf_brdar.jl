using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

import ClimaCore: enable_threading
enable_threading() = true

using OrdinaryDiffEq
using ClimaCorePlots, Plots

using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Operators

##
## Parameters
##

const FT = Float64
const is_small_scale = true

const x_center = is_small_scale ? FT(100e3) : FT(3000e3)              # m
const d = is_small_scale ? FT(5e3) : FT(100e3)                        # m
const f = is_small_scale ? FT(0) : 2 * sin(π / 4) * 2π / FT(86164.09) # 1/s
const u₀ = is_small_scale ? FT(20) : FT(0)                            # m/s
const v₀ = FT(0)                                                      # m/s
const w₀ = FT(0)                                                      # m/s
const T₀ = FT(250)                                                    # K
const ΔT = FT(0.01)                                                   # K
const p₀_surface = FT(1e5)                                            # kg/m/s^2
const g = FT(9.80665)                                                 # m/s^2
const R = FT(287.05)                                                  # J/kg/K
const cₚ = FT(1005.0)                                                 # J/kg/K
const cᵥ = cₚ - R                                                     # J/kg/K

const xmax = is_small_scale ? FT(300e3) : FT(6000e3)
const zmax = FT(10e3)
Δx = is_small_scale ? FT(250) : FT(5000)
Δz = is_small_scale ? Δx / 2 : Δx / 40
npoly = 4
helem = Int(xmax / (Δx * (npoly + 1)))
velem = Int(zmax / Δz)

tmax = is_small_scale ? FT(60 * 60 * 0.5) : FT(60 * 60 * 8)
dt = is_small_scale ? FT(0.5) : FT(75)
save_every_n_steps = 1
ode_algorithm = OrdinaryDiffEq.SSPRK33

##
## Domain
##

hdomain = Domains.IntervalDomain(
    Geometry.XPoint{FT}(zero(FT)),
    Geometry.XPoint{FT}(xmax);
    periodic = true,
)
hmesh = Meshes.IntervalMesh(hdomain; nelems = helem)
htopology = Topologies.IntervalTopology(hmesh)
quad = Spaces.Quadratures.GLL{npoly + 1}()
hspace = Spaces.SpectralElementSpace1D(htopology, quad)

vdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(zero(FT)),
    Geometry.ZPoint{FT}(zmax);
    boundary_tags = (:bottom, :top),
)
vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)

center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

##
## Initial state
##

# This assumes a uniform value of Δz and T₀.
# @∂z_c2f(p) / I_c2f(ρ) + g = 0 ==>
# @∂z_c2f(p) = -g * I_c2f(ρ) ==>
# (p_{i + 1} - p_i)/Δz = -g/2 * (ρ_{i + 1} + ρ_i)
# p = ρRT₀ ==>
# (p_{i + 1} - p_i)/Δz = -g/2RT₀ * (p_{i + 1} + p_i) ==>
# p_{i + 1} = p_i * (1 - gΔz/2RT₀)/(1 + gΔz/2RT₀) ==>
# p_{i + 1} = p_1 * ((1 - gΔz/2RT₀)/(1 + gΔz/2RT₀))^i
# p_{i + 1} = p(Δz * (i + 1/2)) ==>
# p(z) = p_{z/Δz + 1/2} = p_1 * ((1 - gΔz/2RT₀)/(1 + gΔz/2RT₀))^(z/Δz - 1/2) =
#     = p(0) * ((1 - gΔz/2RT₀)/(1 + gΔz/2RT₀))^(z/Δz)

function initial_state_c(local_geometry)
    (; x, z) = local_geometry.coordinates

    # Continuous hydrostatic balance
    # p = p₀_surface * exp(-g * z / (R * T₀))

    # Discrete hydrostatic balance
    Δz = FT(125) # non-constant global variable...
    value = g * Δz / (2 * R * T₀)
    p = p₀_surface * ((1 - value)/(1 + value))^(z / Δz)

    T = T₀ # + exp(g * z / (2 * R * T₀)) * ΔT * exp(-(x - x_center)^2 / d^2) *
        # sin(π * z / zmax)
    return (; u = u₀, v = v₀, p, ρ = p / (R * T))
end
initial_state_f(local_geometry) = (; w = w₀)

Y = Fields.FieldVector(
    c = map(initial_state_c, Fields.local_geometry_field(center_space)),
    f = map(initial_state_f, Fields.local_geometry_field(face_space)),
)

##
## Tendency
##

const I_f2c = Operators.InterpolateF2C()
const I_c2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ∇ₕ = Operators.Gradient()
const ∇ᵥ_f2c = Operators.GradientF2C()
const ∇ᵥ_c2f = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
)
const B = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(FT(0)),
    top = Operators.SetValue(FT(0)),
)

macro ∂x(field)
    esc(:(getproperty.(Geometry.UVector.(∇ₕ.($field)), :u)))
end
macro ∂z_f2c(field)
    esc(:(getproperty.(Geometry.WVector.(∇ᵥ_f2c.($field)), :w)))
end
macro ∂z_c2f(field)
    esc(:(getproperty.(Geometry.WVector.(∇ᵥ_c2f.($field)), :w)))
end

cache = (; c = (; B_g = @. B(one(Y.f.w) * g)))

function tendency!(Yₜ, Y, cache, t)
    @. Yₜ.c.u = -Y.c.u * @∂x(Y.c.u) - @∂x(Y.c.p) / Y.c.ρ + f * Y.c.v
    @. Yₜ.c.u += -I_f2c(Y.f.w * @∂z_c2f(Y.c.u))
    
    @. Yₜ.c.v = -Y.c.u * @∂x(Y.c.v) - f * Y.c.u
    @. Yₜ.c.v += -I_f2c(Y.f.w * @∂z_c2f(Y.c.v))
    
    # temporarily store ∂w/∂x in Yₜ.f.w to avoid BroadcastStyle conflict
    ∂w∂x = Yₜ.f.w 
    @. ∂w∂x = @∂x(Y.f.w)
    @. Yₜ.f.w = -I_c2f(Y.c.u) * ∂w∂x - Y.f.w * I_c2f(@∂z_f2c(Y.f.w)) -
        @∂z_c2f(Y.c.p) / I_c2f(Y.c.ρ) - cache.c.B_g
    
    # temporarily store Dρ/Dt = ∂ρ/∂t + 𝐮⋅∇ρ in Yₜ.c.ρ to avoid recomputation
    DρDt = Yₜ.c.ρ
    @. DρDt = -Y.c.ρ * @∂x(Y.c.u)
    @. DρDt += -Y.c.ρ * @∂z_f2c(Y.f.w)
    
    @. Yₜ.c.p = -Y.c.u * @∂x(Y.c.p) + cₚ / cᵥ * Y.c.p / Y.c.ρ * DρDt
    @. Yₜ.c.p += -I_f2c(Y.f.w * @∂z_c2f(Y.c.p))
    
    @. Yₜ.c.ρ = -Y.c.u * @∂x(Y.c.ρ) + DρDt
    @. Yₜ.c.ρ += -I_f2c(Y.f.w * @∂z_c2f(Y.c.ρ))

    Spaces.weighted_dss!(Yₜ.c)
    Spaces.weighted_dss!(Yₜ.f)
    return Yₜ
end

##
## ODE solver
##

problem = ODEProblem(
    ODEFunction(tendency!; tgrad = (Yₜ, Y, p, t) -> (Yₜ .= FT(0))),
    Y,
    (FT(0), tmax),
    cache,
)
integrator = OrdinaryDiffEq.init(
    problem,
    ode_algorithm();
    dt = dt,
    saveat = dt * save_every_n_steps,
    adaptive = false,
    progress = true,
    progress_steps = 1,
)
sol = @timev OrdinaryDiffEq.solve!(integrator)

##
## Post-processing
##

# ENV["GKSwstype"] = "nul"
# path = joinpath(@__DIR__, "output", "inertial_gravity_wave")
# get_T(Y) = @. Y.c.p / (R * Y.c.ρ)

Yₜ = similar(Y)
tendency!(Yₜ, Y, cache, FT(0))

for symbol in propertynames(Yₜ.c)
    println(
        "The integral of Yₜ.c.$symbol across the domain at t = 0 is ",
        sum(getproperty(Yₜ.c, symbol)),
    )
end
println("The integral of Yₜ.f.w across the domain at t = 0 is ", sum(Yₜ.f.w))
println("The value of Yₜ.f.w at t = 0 is ", Yₜ.f.w)
plot(Yₜ.f.w)
