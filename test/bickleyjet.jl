using ClimateMachineCore.Geometry, LinearAlgebra, UnPack
import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore.Operators
import ClimateMachineCore.Geometry
using LinearAlgebra
using DifferentialEquations: ODEProblem, solve, SSPRK33
using ClimateMachineCore.Operators: ⊠

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
    x1min = -2π,
    x1max = 2π,
    x2min = -2π,
    x2max = 2π,
    x1periodic = true,
    x2periodic = true,
)

n1, n2 = 16, 16
Nq = 10
Nqh = 14
discretization = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
grid_topology = Topologies.GridTopology(discretization)
quad = Meshes.Quadratures.GLL{Nq}()
mesh = Meshes.Mesh2D(grid_topology, quad)

Iquad = Meshes.Quadratures.GLL{Nqh}()
Imesh = Meshes.Mesh2D(grid_topology, Iquad)

function init_state(x, p)
    @unpack x1, x2 = x
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(x2)^(-2)
    Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x1) * cos(p.k * x2)

    ## Vortical velocity fields
    u₁′ = Ψ′ * (p.k * tan(p.k * x2) + x2 / p.l^2)
    u₂′ = -Ψ′ * (p.k * tan(p.k * x1))

    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * x2)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(mesh), Ref(parameters))

function flux(state, p)
    @unpack ρ, ρu, ρθ = state
    u = ρu ./ ρ
    return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * I), ρθ = ρθ .* u)
end

function energy(state, p)
    @unpack ρ, ρu = state
    u = ρu ./ ρ
    return ρ * (u.u1^2 + u.u2^2) / 2 + p.g * ρ^2 / 2
end

function total_energy(y, parameters)
    E = energy.(y, Ref(parameters))
    sum(parent(Fields.field_values(E) .* Fields.mesh(E).local_geometry.WJ))
end

F = flux.(y0, Ref(parameters))
divF = Operators.slab_divergence(F)
#=
wdivF = Operators.slab_weak_divergence(F)
wdivF_data = Fields.field_values(wdivF)
WJ = copy(mesh.local_geometry.WJ) # quadrature weights * jacobian
Operators.horizontal_dss!(WJ, mesh)
wdivF_data .= mesh.local_geometry.WJ .⊠ wdivF_data
Operators.horizontal_dss!(wdivF_data, mesh)
wdivF_data .= inv.(WJ) .⊠ wdivF_data
=#

function reconstruct(rawdata, field)
    D = typeof(Fields.field_values(field))
    Fields.Field(D(rawdata), Fields.mesh(field))
end

function rhs!(rawdydt, rawdata, field, t)
    # reconstuct Field objects
    y = reconstruct(rawdata, field)
    y_data = Fields.field_values(y)

    mesh = Fields.mesh(field)

    dydt = reconstruct(rawdydt, field)
    dydt_data = Fields.field_values(dydt)

    Imat = Meshes.Quadratures.interpolation_matrix(Float64, Iquad, quad)
    Iy_data = similar(Imesh.local_geometry, eltype(dydt_data))
    Operators.tensor_product!(Iy_data, y_data, Imat)
    Iy = Fields.Field(Iy_data, Imesh)
    IF = flux.(Iy, Ref(parameters))
    IdivF = Operators.slab_weak_divergence(IF)
    IdivF_data = Fields.field_values(IdivF)
    IdivF_data .= (.-Imesh.local_geometry.WJ) .⊠ IdivF_data
    Operators.tensor_product!(dydt_data, IdivF_data, Imat')

    Operators.horizontal_dss!(dydt_data, mesh)

    WJ = copy(mesh.local_geometry.WJ) # quadrature weights * jacobian
    Operators.horizontal_dss!(WJ, mesh)
    dydt_data .= inv.(WJ) .⊠ dydt_data

    # mass matrix = dss(WJ)
    # inv mass
    return rawdydt
end
#-------
# div(ρuθ) = ρu * ρθ / ρ
# K : DSS scatter operator
# K' : DSS gather
# I: interpolation operator
# DH: differentiation matrix on high res grid
# DL: differentiation matrix on low res grid
#  => DH*I == I*DL

# (Kϕ)' WJ dydt = - (IDKϕ)' W (J (ρu * ρθ / ρ))

# ϕ' K'WJ dydt =  -ϕ' K' (I*DL)' WJ (I*ρu .* I*ρθ ./ I*ρ)
# ϕ' K'WJ dydt =  -ϕ' K' I' * [DH' WJ (I*ρu .* I*ρθ ./ I*ρ)]

# Next steps:
# 1. add the above to the design docs (divergence + over-integration + DSS)
# 2. add boundary conditions
# 3. clean up the above code
#   - Field operators
#   - remove the invWJ scaling from slab_weak_divergence
#     rename to rhs_slab_weak_divergence
# 4. add the inv(DSSed WJ) to the mesh


dydt = Fields.Field(similar(Fields.field_values(y0)), mesh)
rhs!(
    parent(Fields.field_values(dydt)),
    parent(Fields.field_values(y0)),
    y0,
    0.0,
);

# 1. make DifferentialEquations work on Fields: i think we need to extend RecursiveArrayTools
#    - this doesn't seem like it will work directly: ideally we want a way to unwrap and wrap as required
#    - scalar multiplication and vector addition
#
# 2. Define isapprox on Fields
# 3. weighted DSS



# Solve the ODE operator
prob = ODEProblem(rhs!, parent(Fields.field_values(y0)), (0.0, 200.0), y0)
sol = solve(prob, SSPRK33(), dt = 0.005, saveat = 1.0, progress = true)

using Plots
ENV["GKSwstype"] = "nul"

anim = @animate for u in sol.u
    heatmap(reconstruct(u, y0).ρθ, clim = (-2, 2))
end
mp4(anim, "bickleyjet.mp4", fps = 10)

Es = [total_energy(reconstruct(u, y0), parameters) for u in sol.u]
png(plot(Es), "energy.png")
