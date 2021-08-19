using Oceananigans.TimeSteppers: time_step!

include("ocean_simulation.jl")
include("atmos_sim.jl")
include("land_sim.jl") #this sets up the land model completely, creates land_integrator function

f = 1e-4 # Coriolis parameter
g = 9.81 # Gravitational acceleration
ρ_ocean = 1024.0  # Ocean density

ocean_Nz = 64  # Number of vertical grid points
ocean_Lz = 512 # Vertical extent of domain
ocean_T₀ = 20  # ᵒC, sea surface temperature
ocean_S₀ = 35  # psu, sea surface salinity

atmos_Nz = 64  # Number of vertical grid points
atmos_Lz = 512 # Vertical extent of domain

#Land will not necessarily have the same depth as ocean, but it's fine - fluxes only at surface
#land_Nz = 64  # Number of vertical grid points
#land_Lz = 512 # Vertical extent of domain


land_sim = SoilModel(land_integrator)#land_simulation(; land_Nz, land_Lz) #right now land doesnt have an interface like this

atmos_sim = atmos_simulation(; atmos_Nz, atmos_Lz)
ocean_sim = ocean_simulation(; ocean_Nz, ocean_Lz, ocean_T₀, ocean_S₀)

struct CoupledSimulation{O, A, L}
    ocean :: O
    atmos :: A
    land :: L
end

function step!(coupled_sim::CoupledSimulation, Δt)

    atmos_sim = coupled_sim.atmos
    ocean_sim = coupled_sim.ocean
    land_sim = coupled_sim.land

    step!(atmos_sim, Δt, lnd_state)

    # Set land fluxes
    # Flux on energy (J/m^2/s)
    #since land model struct isnt set up yet, pass flux to use as argument
    step!(land_sim, Δt, flux_computed_by_atmos)

    # Set ocean fluxes
    ∫surface_flux_u_atmos = atmos_sim.pr
    u_atmos_surface_flux = atmos_sim.
    v_atmos_surface_flux = atmos_sim. 
    ρθ_atmos_surface_flux = atmos_sim. 
    # salinity fluxes?

    u_ocean_top_bc = ocean_sim.model.velocities.u.boundary_conditions.top
    v_ocean_top_bc = ocean_sim.model.velocities.v.boundary_conditions.top
    T_ocean_top_bc = ocean_sim.model.tracers.T.boundary_conditions.top
    # S_ocean_top_bc = ocean_sim.model.tracers.S.boundary_conditions.top

    @. u_ocean_top_bc.condition = ρ_atmos / ρ_ocean * u_atmos_surface_flux
    @. v_ocean_top_bc.condition = ρ_atmos / ρ_ocean * v_atmos_surface_flux
    @. T_ocean_top_bc.condition = 1 / ρ_ocean * ρθ_atmos_surface_flux

    time_step!(ocean_sim.model, Δt, euler=true)

    return nothing
end