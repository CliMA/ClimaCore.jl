using Oceananigans.TimeSteppers: time_step!, Clock, tick!

import SciMLBase: step!

using Printf

include("dummy_surface_fluxes.jl") # placeholder for SurfaceFluxes.jl

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
    land_surface_T = land_sim.p[5]
    
    ocean_Nz = ocean_sim.model.grid.Nz
    @inbounds begin
        ocean_surface_T = ocean_sim.model.tracers.T[1, 1, ocean_Nz] # convert to K somewhere
    end

    # Step forward atmosphere
    atmos_sim.u.x[3] .= [0.0, 0.0, 0.0] # reset surface flux to be accumulated during each coupling_Δt 
    atmos_sim.p[2] .= land_surface_T # get land temperature and set on atmosphere (Tland is prognostic)

    # TODO: determine if this will be useful (init step not ran w/o this but same outcome)
    # u_atmos = atmos_sim.u 
    # u_atmos.x[3] .= u_atmos.x[3] .* -0.0
    # set_u!(atmos_sim, u_atmos)

    step!(atmos_sim, next_time - atmos_sim.t, true)

    # Extract surface fluxes for ocean and land boundaries
    ∫surface_x_momentum_flux = atmos_sim.u.x[3][1] # kg / m s^2
    ∫surface_y_momentum_flux = atmos_sim.u.x[3][2] # kg / m s^2
    ∫surface_heat_flux = atmos_sim.u.x[3][3]       # W / m^2
    
    surface_flux_u_ocean = ocean_sim.model.velocities.u.boundary_conditions.top.condition
    surface_flux_v_ocean = ocean_sim.model.velocities.v.boundary_conditions.top.condition
    surface_flux_T_ocean = ocean_sim.model.tracers.T.boundary_conditions.top.condition

    @. surface_flux_u_ocean = 1 / ocean_ρ * ∫surface_x_momentum_flux / coupling_Δt 
    @. surface_flux_v_ocean = 1 / ocean_ρ * ∫surface_y_momentum_flux / coupling_Δt 
    @. surface_flux_T_ocean = 1 / (ocean_ρ * ocean_Cᴾ) * ∫surface_heat_flux / coupling_Δt 

    # We'll develop a new function step!(ocean_sim, coupling_Δt) :thumsup:
    time_step!(ocean_sim.model, 0.02)#next_time - ocean_sim.model.clock.time)
    data = (T = deepcopy(ocean_sim.model.tracers.T), time = ocean_sim.model.clock.time)
    push!(ocean_data, data)
    
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
# These parameters will live in the ocean simulation someday
# For ocean_Cᴾ see 3.32 in http://www.teos-10.org/pubs/TEOS-10_Manual.pdf
ocean_ρ = 1024.0 # [kg / m^3] average density at the ocean surface
ocean_Cᴾ = 3991.9 # [J / kg K] reference heat capacity for conservative temperature

atmos_Nz = 30  # Number of vertical grid points
atmos_Lz = 200 # Vertical extent of domain

# land domain params (need to abstract)
# n = 50
# zmax = FT(0)
# zmin = FT(-1)
land_Nz = n
land_Lz = zmax - zmin # this needs abstraction

start_time = 0.0 
stop_time = 1#coupling_Δt*100#60*60

# Build the respective models
land_sim = land_simulation()
atmos_sim = atmos_simulation(land_sim, Nz=atmos_Nz, Lz=atmos_Lz, start_time = start_time, stop_time = stop_time,)
ocean_sim, ocean_data = ocean_simulation(Nz=ocean_Nz, Lz=ocean_Lz, f=f, g=g)
# data = deepcopy(ocean_sim.model.tracers.T)
# push!(ocean_data, data)

# Build a coupled simulation
clock = Clock(time=0.0)
coupled_sim = CoupledSimulation(ocean_sim, atmos_sim, land_sim, clock)

# Run it!
coupling_Δt = 0.02
solve!(coupled_sim, coupling_Δt, stop_time)

# Visualize
using Plots
Plots.GRBackend()

dirname = "heat"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# Atmos plots
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

# Land plots
sol_lnd = coupled_sim.land.sol
t0_θ_l = parent(sol_lnd.u[1].x[1])
tend_θ_l = parent(sol_lnd.u[end].x[1])
t0_ρe = parent(sol_lnd.u[1].x[3])
tend_ρe = parent(sol_lnd.u[end].x[3])
t0_ρc_s = volumetric_heat_capacity.(t0_θ_l, parent(θ_i), Ref(msp.ρc_ds), Ref(param_set)) #convert energy to temp
t0_T = temperature_from_ρe_int.(t0_ρe, parent(θ_i),t0_ρc_s, Ref(param_set)) #convert energy to temp
tend_ρc_s = volumetric_heat_capacity.(tend_θ_l, parent(θ_i), Ref(msp.ρc_ds), Ref(param_set)) #convert energy to temp
tend_T = temperature_from_ρe_int.(tend_ρe, parent(θ_i),tend_ρc_s, Ref(param_set)) #convert energy to temp
z_centers =  collect(1:1:length(tend_ρe))#parent(Fields.coordinate_field(center_space_atm))[:,1]
Plots.png(Plots.plot([t0_θ_l tend_θ_l],parent(zc), labels = ["t=0" "t=end"]), joinpath(path, "Th_l_lnd_height.png"))
Plots.png(Plots.plot([t0_T tend_T],parent(zc), labels = ["t=0" "t=end"]), joinpath(path, "T(K)_lnd_height.png"))

# Ocean plots
sol_ocn = coupled_sim.ocean.model
t0_T = ocean_data[1].T.data[1,1,:]
tend_T = ocean_data[end].T.data[1,1,:] # = sol_ocn.tracers.T.data[1,1,:]
z_centers =  collect(1:1:length(tend_T))
Plots.png(Plots.plot([t0_T tend_T],z_centers, labels = ["t=0" "t=end"], ylims=(0,100)), joinpath(path, "T_ocn_height.png"))

# Time evolution of all enthalpies (do for total enery = p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ, which is actually conserved)
atm_sum_u_t = [sum(parent(u.x[1])[:,4]) for u in sol_atm.u] ./ atmos_Nz .* atmos_Lz * parameters.C_p # J / m2
lnd_sfc_u_t = [sum(parent(u.x[3])[:]) for u in sol_lnd.u] ./ land_Nz .* land_Lz # J / m2
ocn_sum_u_t = ([sum(interior(u.T)[1,1,:] .+ 273.15 ) for u in ocean_data]) ./ ocean_Nz .* ocean_Lz * ocean_ρ * ocean_Cᴾ # J / m2

v1 = lnd_sfc_u_t .- lnd_sfc_u_t[1]
v2 = atm_sum_u_t .- atm_sum_u_t[1]
v3 = ocn_sum_u_t .- ocn_sum_u_t[1]
Plots.png(Plots.plot(sol_lnd.t, [v1 v2 v1+v2], labels = ["lnd" "atm" "tot"]), joinpath(path, "enthalpy_both_surface_time_atm_lnd.png"))
Plots.png(Plots.plot(sol_lnd.t, [v3 v2 v3+v2], labels = ["ocn" "atm" "tot"]), joinpath(path, "enthalpy_both_surface_time_atm_ocn.png"))
Plots.png(Plots.plot(sol_lnd.t, [v1 v3 v1-v3], labels = ["lnd" "ocn" "diff"]), joinpath(path, "enthalpy_both_surface_time_ocn_lnd.png"))

# relative error of enthapy
using Statistics
total = atm_sum_u_t + lnd_sfc_u_t
rel_error = (total .- total[1]) / mean(total)
Plots.png(Plots.plot(sol_lnd.t, rel_error, labels = ["tot"]), joinpath(path, "rel_error_surface_time.png"))

# TODO
# - add domain info, similar to aceananigans: coupled_sim.ocean.model.grid. ... 
#       - like that oceananigans model prints out basic characteristics (nel, BCs etc)
# - oceananigans doesn't store all times...?
# - how would be do the accumulation in oceananigans?
# - ekman column - dw equation looks odd
