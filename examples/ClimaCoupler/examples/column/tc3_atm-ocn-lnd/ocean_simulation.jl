using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity

#####
##### Parameters
#####

"""
    ocean_simulation(; kwargs...)

Return an `Oceananigans.Simulation` of a column model initialized with
mutable array surface boundary conditions and a linear density stratification.

Arguments
=========

    * Nz: Number of vertical grid points
    * Lz: [m] Vertical extent of domain,
    * f:  [s-1] Coriolis parameter,
    * g:  [m s-2] Gravitational acceleration,
    * T₀: [ᵒC], sea surface temperature,
    * S₀: [psu], sea surface salinity,
    * α: Thermal expansion coefficient
    * β: Haline contraction coefficient
"""
function ocean_simulation(; Nz = 64,  # Number of vertical grid points
                            Lz = 512, # Vertical extent of domain
                            f = 1e-4, # Coriolis parameter
                            g = 9.81, # Gravitational acceleration
                            α = 2e-4, # Thermal expansion coefficient
                            β = 8e-5, # Haline contraction coefficient
                          )

    grid = RegularRectilinearGrid(size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(-Lz, 0), topology=(Periodic, Periodic, Bounded))
    
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition([0.0]))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition([0.0]))
    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition([0.0]))
    S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition([0.0]))
    
    eos = LinearEquationOfState(α=α, β=β)
    
    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = (:T, :S, :e),
                                        free_surface = ImplicitFreeSurface(gravitational_acceleration=g),
                                        buoyancy = SeawaterBuoyancy(gravitational_acceleration=g, equation_of_state=eos),
                                        coriolis = FPlane(f=f),
                                        boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs, v=v_bcs),
                                        closure = TKEBasedVerticalDiffusivity(),)
    
    simulation = Oceananigans.Simulation(model, Δt=0.02, stop_iteration=1)
    
    # Initialize the ocean state with a linear temperature and salinity stratification
    α = simulation.model.buoyancy.model.equation_of_state.α
    β = simulation.model.buoyancy.model.equation_of_state.β
    Tᵢ(x, y, z) = 16 + α * g * 5e-5 * z
    Sᵢ(x, y, z) = 35 - β * g * 5e-5 * z
    set!(simulation.model, T = Tᵢ, S = Sᵢ)

    # collect data (needs optimising)
    ocean_data = []
    data = (T = deepcopy(simulation.model.tracers.T), time = simulation.model.clock.time)
    push!(ocean_data, data)
    
    return simulation, ocean_data
end
