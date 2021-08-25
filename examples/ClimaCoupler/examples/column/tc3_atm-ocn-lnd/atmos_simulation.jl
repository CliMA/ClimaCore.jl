using ClimaAtmos
using ClimaCore.Geometry

using ClimaCore: DataLayouts, Operators, Geometry
using ClimaAtmos.Interface: TimeStepper, Simulation
using ClimaAtmos.Interface: PeriodicRectangle, BarotropicFluidModel
using ClimaCore: Fields, Domains, Topologies, Meshes, Spaces

using IntervalSets
using UnPack
using OrdinaryDiffEq: SSPRK33
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using Logging: global_logger
using TerminalLoggers: TerminalLogger

using RecursiveArrayTools
using LinearAlgebra
using OrdinaryDiffEq
using Random

global_logger(TerminalLogger())

const CI = !isnothing(get(ENV, "CI", nothing))

# general parameters
const FT = Float64

# define model equations:
include("atmos_rhs.jl")

parameters = (
        # atmos physics (overwriting CliMAParameters defaults?)
        MSLP = 1e5, # mean sea level pressure [Pa]
        grav = 9.8, # gravitational acceleration [m /s^2]
        R_d = 287.058, # R dry (gas constant / mol mass dry air)  [J / K / kg]
        C_p = 287.058 * 1.4 / (1.4 - 1), # heat capacity at constant pressure [J / K / kg]
        C_v = 287.058 / (1.4 - 1), # heat capacity at constant volume [J / K / kg]
        R_m = 87.058, # moist R, assumed to be dry [J / K / kg]
        f = 7.29e-5, # Coriolis parameters [1/s]
        ν = 0.1, #0.01 # viscosity, diffusivity
        Ch = 0.0015, # bulk transfer coefficient for sensible heat
        Cd = 0.01 / (2e2 / 30.0), #drag coeff
        ug = 1.0,
        vg = 0.0,
        d = sqrt(2.0 * 0.01 / 5e-5), #?

        # radiation parameters for DryBulkFormulaWithRadiation() SurfaceFluxType
        τ    = 0.9,     # atmospheric transmissivity
        α    = 0.5,     # surface albedo
        σ    = 5.67e-8, # Steffan Boltzmann constant [kg / s^3 / K^4]
        g_a  = 0.06,    # aerodynamic conductance for heat transfer [kg / m^2 / s]
        ϵ    = 0.98,    # broadband emissivity / absorptivity of the surface
        F_a  = 0.0,     # downward LW flux from the atmosphere [W / m^2]
        F_sol = 1361,   # incoming solar TOA radiation [W / m^2]
        τ_d   = 10,     # idealized daily cycle period [s]

        # surface fluxes
        λ = FT(0.01),#FT(1e-5)    # coupling transfer coefficient for LinearRelaxation() SurfaceFluxType 
    )

function atmos_simulation(land_simulation;
                          Lz,
                          Nz,
                          minimum_reference_temperature = 230.0, # K
                          start_time = 0.0,
                          stop_time = 1.0,
                          Δt_min  = 0.02,
                          )

    # Get surface temperature from the land simulation
    land_surface_temperature = land_sim.p[5]

    ########
    # Set up atmos domain
    ########
    
    domain_atm  = Domains.IntervalDomain(0, Lz, x3boundary = (:bottom, :top)) # struct
    mesh_atm = Meshes.IntervalMesh(domain_atm, nelems = Nz) # struct, allocates face boundaries to 5,6
    center_space_atm = Spaces.CenterFiniteDifferenceSpace(mesh_atm) # collection of the above, discretises space into FD and provides coords
    face_space_atm = Spaces.FaceFiniteDifferenceSpace(center_space_atm)
    
    """ Initialize fields located at cell centers in the vertical. """
    function init_centers(zc, parameters)
        UnPack.@unpack grav, C_p, MSLP, R_d = parameters

        # temperature
        Γ = grav / C_p
        T = max(land_surface_temperature - Γ * zc, minimum_reference_temperature)
    
        # pressure
        p = MSLP * (T / land_surface_temperature)^(grav / (R_d * Γ))

        if T == minimum_reference_temperature
            z_top = (land_surface_temperature - minimum_reference_temperature) / Γ
            H_min = R_d * minimum_reference_temperature / grav
            p *= exp(-(zc - z_top) / H_min)
        end
    
        # potential temperature
        θ = land_surface_temperature
    
        # density
        ρ = p / (R_d * θ * (p / MSLP)^(R_d / C_p))
    
        # velocties
        u = 1.0
        v = 0.0
    
        return (ρ = ρ, u = u, v = v, ρθ = ρ * θ)
    end
    
    """ Initialize fields located at cell interfaces in the vertical. """
    function init_faces(zf, parameters)
        return (; w = 0.0 .* zf)
    end
    
    # Initialize the atmospheric states Yc and Yf
    z_centers = Fields.coordinate_field(center_space_atm)
    z_faces = Fields.coordinate_field(face_space_atm)
    Yc = init_centers.(z_centers, Ref(parameters))
    Yf = init_faces.(z_faces, Ref(parameters))
    T_atm_0 = (Yc = Yc, Yf = Yf)

    # Put all prognostic variable arrays into a vector and ensure that solve can partition them
    Y_atm = ArrayPartition((T_atm_0.Yc, T_atm_0.Yf, zeros(3)))
    prob_atm = ODEProblem(∑tendencies_atm!, Y_atm, (start_time, stop_time), (parameters, [land_surface_temperature]))
    simulation = init(prob_atm, SSPRK33(), dt = Δt_min, saveat = 1 * Δt_min)
    
    return simulation
end


# abstract type AtmosModel end

# struct AtmosModel <: AtmosModel
#     integrator::A
# end


# function step!(model, Δt, land_surface_temp)

#     model.integrator.u.x[3] .= [0.0, 0.0, 0.0] # surface flux to be accumulated    
#     model.integrator.p[2] .= land_surface_temp # get land temperature and set on atmosphere (Tland is prognostic)
#     OrdinaryDiffEq.step!(model.integrator, Δt, true) #Δt here does not need to be the land model step
#     # the integrator contains the land model timestep dt. with [Δt, true] args the integrator will step by dt until it has
#     # advanced exactly Δt. 

# end

