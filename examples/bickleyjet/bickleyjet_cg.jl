push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimateMachineCore.Geometry, LinearAlgebra, UnPack
import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore: slab
import ClimateMachineCore.Operators
import ClimateMachineCore.Geometry
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
    sum(state -> energy(state, parameters), y)
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


function rhs!(dydt, y, _, t)

    # ϕ' K' W J K dydt =  -ϕ' K' I' [DH' WH JH flux.(I K y)]
    #  =>   K dydt = - K inv(K' WJ K) K' I' [DH' WH JH flux.(I K y)]

    # where:
    #  ϕ = test function
    #  K = DSS scatter (i.e. duplicates points at element boundaries)
    #  K y = stored input vector (with duplicated values)
    #  I = interpolation to higher-order mesh
    #  D = derivative operator
    #  H = suffix for higher-order mesh operations
    #  W = Quadrature weights
    #  J = Jacobian determinant of the transformation `ξ` to `x`
    #
    Nh = Topologies.nlocalelems(y)

    # for all slab elements in mesh
    for h in 1:Nh
        y_slab = slab(y, h)
        dydt_slab = slab(dydt, h)
        Imesh_slab = slab(Imesh, h)

        # 1. Interpolate to higher-order mesh
        Iy_slab = Operators.interpolate(Imesh_slab, y_slab)

        # 2. compute fluxes
        #  flux.(I K y)
        IF_slab = flux.(Iy_slab, Ref(parameters))

        # 3. "weak" divergence
        #  DH' WH JH flux.(I K y)
        WdivF_slab = Operators.slab_weak_divergence(IF_slab)

        # 4. "back" interpolate to regular mesh
        #  I' [DH' WH JH flux.(I K y)]
        Operators.restrict!(dydt_slab, WdivF_slab)
    end

    # 5. Apply DSS gather operator
    #  K' I' [DH' WH JH flux.(I K y)]
    Meshes.horizontal_dss!(dydt)

    # 6. Solve for final result
    #  K inv(K' WJ K) K' I' [DH' WH JH flux.(I K y)]
    Meshes.variational_solve!(dydt)
end

# Next steps:
# 1. add the above to the design docs (divergence + over-integration + DSS)
# 2. add boundary conditions

dydt = Fields.Field(similar(Fields.field_values(y0)), mesh)
rhs!(dydt, y0, nothing, 0.0);


# Solve the ODE operator
prob = ODEProblem(rhs!, y0, (0.0, 200.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.02,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

using Plots
ENV["GKSwstype"] = "nul"

dirname = "cg"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = @animate for u in sol.u
    heatmap(u.ρθ, clim = (-1, 1), color = :balance)
end
mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
png(plot(Es), joinpath(path, "energy.png"))
