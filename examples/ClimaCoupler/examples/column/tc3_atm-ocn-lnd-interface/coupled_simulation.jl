using Oceananigans.TimeSteppers: time_step!, Clock, tick!

import SciMLBase: step!

using Printf

include("land_simulation.jl") #refactoring of land interface to come
include("ocean_simulation.jl")
include("atmos_simulation.jl")

struct CoupledSimulation{O, A, L, C}
    ocean :: O
    atmos :: A
    land :: L
    clock :: C
end

function step!(coupled_sim::CoupledSimulation, coupling_Δt)

    atmos_sim = coupled_sim.atmos
    ocean_sim = coupled_sim.ocean
    land_sim = coupled_sim.land

    clock = coupled_sim.clock
    next_time = clock.time + coupling_Δt
    @info("Coupling cycle", time = clock.time)

    # Extract states and parameters at coupled boundaries for flux calculations
    land_surface_T = land_sim.p[5]#parent(land_sim.u.x[3])[end]
    
    ocean_Nz = ocean_sim.model.grid.Nz
    @inbounds begin
        ocean_surface_T = ocean_sim.model.tracers.T[1, 1, ocean_Nz] # convert to K somewhere
    end

    # Step forward atmosphere
    atmos_sim.u.x[3] .= [0.0, 0.0, 0.0] # reset surface flux to be accumulated during each coupling_Δt 
    atmos_sim.p[2] .= land_surface_T # get land temperature and set on atmosphere (Tland is prognostic)

    # TODO: determine if this will be useful (init step not ran w/o this but same outcome)
    u_atmos = atmos_sim.u 
    u_atmos.x[3] .= u_atmos.x[3] .* -0.0
    set_u!(atmos_sim, u_atmos)

    step!(atmos_sim, next_time - atmos_sim.t, true)

    # Extract surface fluxes for ocean and land boundaries
    ∫surface_x_momentum_flux = atmos_sim.u.x[3][1] # kg / m s^2
    ∫surface_y_momentum_flux = atmos_sim.u.x[3][2] # kg / m s^2
    ∫surface_heat_flux = atmos_sim.u.x[3][3]       # W / m^2
    
    surface_flux_u_ocean = ocean_sim.model.velocities.u.boundary_conditions.top.condition
    surface_flux_v_ocean = ocean_sim.model.velocities.v.boundary_conditions.top.condition
    surface_flux_T_ocean = ocean_sim.model.tracers.T.boundary_conditions.top.condition

    # These parameters will live in the ocean simulation someday
    # For Cᴾ_ocean see 3.32 in http://www.teos-10.org/pubs/TEOS-10_Manual.pdf
    ρ_ocean = 1024.0 # [kg / m^3] average density at the ocean surface
    Cᴾ_ocean = 3991.9 # [J / kg K] reference heat capacity for conservative temperature

    @. surface_flux_u_ocean = 1 / ρ_ocean * ∫surface_x_momentum_flux / coupling_Δt 
    @. surface_flux_v_ocean = 1 / ρ_ocean * ∫surface_y_momentum_flux / coupling_Δt 
    @. surface_flux_T_ocean = 1 / (ρ_ocean * Cᴾ_ocean) * ∫surface_heat_flux / coupling_Δt 

    # We'll develop a new function step!(ocean_sim, coupling_Δt) :thumsup:
    time_step!(ocean_sim.model, next_time - ocean_sim.model.clock.time)
   
    # Advance land
    @show(∫surface_x_momentum_flux, ∫surface_y_momentum_flux, ∫surface_heat_flux)
    land_sim.p[4].top_heat_flux = ∫surface_heat_flux / coupling_Δt # [W/m^2] same BC across land Δt
    step!(land_sim, next_time - land_sim.t, true)

    tick!(clock, coupling_Δt)

    return nothing
end

function solve!(coupled_sim::CoupledSimulation, coupling_Δt, stop_time)
    @info("Coupler:", models = fieldnames(typeof(coupled_sim))[1:end-1])
    while coupled_sim.clock.time < stop_time
        step!(coupled_sim, coupling_Δt)
    end
end

f = 1e-4 # Coriolis parameter
g = 9.81 # Gravitational acceleration

ocean_Nz = 64  # Number of vertical grid points
ocean_Lz = 512 # Vertical extent of domain
ocean_T₀ = 20  # ᵒC, sea surface temperature
ocean_S₀ = 35  # psu, sea surface salinity

atmos_Nz = 30  # Number of vertical grid points
atmos_Lz = 200 # Vertical extent of domain

land_sim = land_simulation()
atmos_sim = atmos_simulation(land_sim, Nz=atmos_Nz, Lz=atmos_Lz)

# Build the ocean model
ocean_sim = ocean_simulation(Nz=ocean_Nz, Lz=ocean_Lz, f=f, g=g)

# Initialize the ocean state with a linear temperature and salinity stratification
α = ocean_sim.model.buoyancy.model.equation_of_state.α
β = ocean_sim.model.buoyancy.model.equation_of_state.β
Tᵢ(x, y, z) = 16 + α * g * 5e-5 * z
Sᵢ(x, y, z) = 35 - β * g * 5e-5 * z
set!(ocean_sim.model, T = Tᵢ, S = Sᵢ)

# Build a coupled simulation
clock = Clock(time=0.0)
coupled_sim = CoupledSimulation(ocean_sim, atmos_sim, land_sim, clock)

# Run it!
coupling_Δt = 0.01

stop_time = coupling_Δt*300#60*60
solve!(coupled_sim, coupling_Δt, stop_time)

using Plots
Plots.GRBackend()

dirname = "heat"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# atmos plots
sol_atm = coupled_sim.atmos.sol
t0_ρθ = parent(sol_atm.u[1].x[1])[:,4]
tend_ρθ = parent(sol_atm.u[end].x[1])[:,4]
t0_u = parent(sol_atm.u[1].x[1])[:,2]
tend_u = parent(sol_atm.u[end].x[1])[:,2]
t0_v = parent(sol_atm.u[1].x[1])[:,3]
tend_v = parent(sol_atm.u[end].x[1])[:,3]
z_centers =  collect(1:1:length(tend_u))#parent(Fields.coordinate_field(center_space_atm))[:,1]
Plots.png(Plots.plot([t0_ρθ tend_ρθ],z_centers, labels = ["t=0" "t=end"]), joinpath(path, "T_atm_height.png"))
Plots.png(Plots.plot([t0_u tend_u],z_centers, labels = ["t=0" "t=end"]), joinpath(path, "u_atm_height.png"))
Plots.png(Plots.plot([t0_v tend_v],z_centers, labels = ["t=0" "t=end"]), joinpath(path, "v_atm_height.png"))

# land plots
sol_lnd = coupled_sim.land.sol
t0_θ_l = parent(sol_lnd.u[1].x[1])
tend_θ_l = parent(sol_lnd.u[end].x[1])
t0_ρe = parent(sol_lnd.u[1].x[3])
tend_ρe = parent(sol_lnd.u[end].x[3])
z_centers =  collect(1:1:length(tend_ρe))#parent(Fields.coordinate_field(center_space_atm))[:,1]
Plots.png(Plots.plot([t0_θ_l tend_θ_l],z_centers, labels = ["t=0" "t=end"]), joinpath(path, "Th_l_lnd_height.png"))
Plots.png(Plots.plot([t0_ρe tend_ρe],z_centers, labels = ["t=0" "t=end"]), joinpath(path, "e_lnd_height.png"))

# ocean plots
sol_ocn = coupled_sim.ocean.model
#sol_ocn.velocities.u.data 
tend_T = sol_ocn.tracers.T.data[1,1,:]
z_centers =  collect(1:1:length(tend_T))
Plots.png(Plots.plot([tend_T tend_T],z_centers, labels = ["t=end" "t=end"]), joinpath(path, "T_ocn_height.png"))

# TODO
# - add domain info, similar to aceananigans: coupled_sim.ocean.model.grid. ... 
#       - like that oceananigans model prints out basic characteristics (nel, BCs etc)
# - oceananigans doesn't store all times...?
# - how would be do the accumulation in oceananigans?
# - ekman column - dw equation looks odd



