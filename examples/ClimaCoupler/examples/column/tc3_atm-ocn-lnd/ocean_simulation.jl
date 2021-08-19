using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity

#####
##### Parameters
#####

function ocean_simulation(; Nz = 64  # Number of vertical grid points
                            Lz = 512 # Vertical extent of domain
                            f = 1e-4 # Coriolis parameter
                            g = 9.81 # Gravitational acceleration
                            T₀ = 20  # ᵒC, sea surface temperature
                            S₀ = 35  # psu, sea surface salinity
                            α = 2e-4 # Thermal expansion coefficient
                            β = 8e-5 # Haline contraction coefficient
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
                                        closure = TKEBasedVerticalDiffusivity())
    
    # Half temperature, half salinity stratification
    N² = 1e-5
    dTdz = + α * g * N² / 2
    dSdz = - β * g * N² / 2
    Tᵢ(x, y, z) = T₀ + dTdz * z
    Sᵢ(x, y, z) = S₀ + dSdz * z
    set!(model, T = Tᵢ, S = Sᵢ)
    
    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

    return simulation
end

    # simulation = Simulation(model, Δt=1minute, stop_time=12hour)