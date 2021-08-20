using Oceananigans.TimeSteppers: time_step!, Clock, tick!

import SciMLBase: step!

include("land_sim.jl") #this sets up the land model completely, creates land_integrator function
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

    # Extract interface states
    land_surface_temp = parent(land_sim.u.x[3])[end]

    #=
    ocean_Nz = ocean_sim.model.grid.Nz
    @inbounds begin
        ocean_surface_u = ocean_sim.model.velocities.u[1, 1, ocean_Nz]
        ocean_surface_v = ocean_sim.model.velocities.v[1, 1, ocean_Nz]
        ocean_surface_T = ocean_sim.model.tracers.T[1, 1, ocean_Nz]
    end
    =#

    # Step forward atmosphere
    atmos_sim.u.x[3] .= [0.0, 0.0, 0.0] # surface flux to be accumulated    
    atmos_sim.p[2] .= land_surface_temp # get land temperature and set on atmosphere (Tland is prognostic)
    step!(atmos_sim, next_time - atmos_sim.t, true)

    # Calculate ocean and land fluxes
    fluxes_per_s = - atmos_sim.u.x[3] / coupling_Δt 
    
    # Since land model struct isnt set up yet, pass flux to use as argument
    flux_computed_by_atmos = 0.0 # can see if we recover the original land sim even in "coupled" mode :)
    land_sim.p[4].top_heat_flux = flux_computed_by_atmos # same BC across Δt
    step!(land_sim, next_time - land_sim.t, true)
   
    ∫surface_flux_u_atmos = atmos_sim.u.x[3][1]
    ∫surface_flux_v_atmos = atmos_sim.u.x[3][2]
    ∫surface_flux_ρθ_atmos = atmos_sim.u.x[3][3]

    surface_flux_u_ocean = ocean_sim.model.velocities.u.boundary_conditions.top.condition
    surface_flux_v_ocean = ocean_sim.model.velocities.v.boundary_conditions.top.condition
    surface_flux_T_ocean = ocean_sim.model.tracers.T.boundary_conditions.top.condition

    @. surface_flux_u_ocean = ρ_atmos / ρ_ocean * ∫surface_flux_u_atmos / coupling_Δt 
    @. surface_flux_v_ocean = ρ_atmos / ρ_ocean * ∫surface_flux_v_atmos / coupling_Δt 
    @. surface_flux_T_ocean = 1 / ρ_ocean       * ∫surface_flux_ρθ_atmos / coupling_Δt 

    # We'll develop a new function step!(ocean_sim, Δt) :thumsup:
    time_step!(ocean_sim.model, next_time - ocean_sim.model.clock.time)
    
    tick!(clock, Δt)

    return nothing
end

function solve!(coupled_sim::CoupledSimulation, coupling_Δt, stop_time)
    # most basic version of this...
    while coupled_sim.clock.time < stop_time
        step!(coupled_sim, coupling_Δt)
    end
end

f = 1e-4 # Coriolis parameter
g = 9.81 # Gravitational acceleration
ρ_ocean = 1024.0  # Ocean density

ocean_Nz = 64  # Number of vertical grid points
ocean_Lz = 512 # Vertical extent of domain
ocean_T₀ = 20  # ᵒC, sea surface temperature
ocean_S₀ = 35  # psu, sea surface salinity

atmos_Nz = 64  # Number of vertical grid points
atmos_Lz = 512 # Vertical extent of domain

land_sim = land_simulation()
atmos_sim = atmos_simulation(land_sim, Nz=atmos_Nz, Lz=atmos_Lz)

# Build the ocean model
ocean_sim = ocean_simulation(Nz=ocean_Nz, Lz=ocean_Lz, f=f, g=g)

# Initialize the ocean state with a linear temperature and salinity stratification
α = ocean_sim.model.buoyancy.model.equation_of_state.α
β = ocean_sim.model.buoyancy.model.equation_of_state.β
Tᵢ(x, y, z) = 20 + α * g * 5e-5 * z
Sᵢ(x, y, z) = 35 - β * g * 5e-5 * z
set!(ocean_sim.model, T = Tᵢ, S = Sᵢ)

# Build a coupled simulation
clock = Clock(time=0.0)
coupled_sim = CoupledSimulation(ocean_sim, atmos_sim, land_sim, clock)

# Run it!
coupling_Δt = 0.01
step!(coupled_sim, coupling_Δt)

stop_time = 60*60
# solve!(coupled_sim, coupling_Δt, stop_time)
