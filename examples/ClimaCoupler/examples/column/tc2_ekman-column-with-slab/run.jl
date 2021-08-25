using Base: show_supertypes
#push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

# add https://github.com/CliMA/ClimaCore.jl/#main
# add https://github.com/CliMA/ClimaAtmos.jl/#main

# import required modules
import ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

using ClimaAtmos

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger

using RecursiveArrayTools

using OrdinaryDiffEq, Test, Random

global_logger(TerminalLogger())

const CI = !isnothing(get(ENV, "CI", nothing))

# general parameters
const FT = Float64

include("dummy_surface_fluxes.jl") # placeholder for SurfaceFluxes.jl


########
# Set up parameters
########

parameters = (
        # timestepping parameters 
        Δt_min = 0.02, # minimum model timestep [s]
        timerange = (0.0, 100.0),  # start time and end time [s]
        odesolver = SSPRK33(), # timestepping method from DifferentialEquations.jl (used in both models here)
        nsteps_atm = 1, # no. time steps of atm before coupling 
        nsteps_lnd = 1, # no. time steps of lnd before coupling 
        saveat = 0.2, # interval at which to save diagnostics [s]

        # atmos domain
        zmin_atm = FT(0.0),     # height of atmos stack bottom
        zmax_atm = FT(200.0), # height of atmos domain top
        n = 30,                 # number of vertical levels

        # atmos physics (eventually will be importing/overwriting CliMAParameters defaults?)
        T_surf = 300.0, # K
        T_min_ref = 230.0, #K
        MSLP = 1e5, # mean sea level pressure [Pa]
        grav = 9.8, # gravitational constant [m /s^2]
        R_d = 287.058, # R dry (gas constant / mol mass dry air)  [J / K / kg]
        C_p = 287.058 * 1.4 / (1.4 - 1), # heat capacity at constant pressure [J / K / kg]
        C_v = 287.058 / (1.4 - 1), # heat capacity at constant volume [J / K / kg]
        R_m = 87.058, # moist R, assumed to be dry [J / K / kg]
        f = 7.29e-5,#5e-5,# Coriolis parameters [1/s]
        ν = .1, #0.01,# viscosity, diffusivity
        Cd = 0.01 / (2e2 / 30.0), #drag coeff
        Ch = 0.01 / (2e2 / 30.0), #thermal transfer coefficient
        ug = 1.0,
        vg = 0.0,
        d = sqrt(2.0 * 0.01 / 5e-5), #?

        # soil slab
        h_s  = 100,     # depth of the modelled soil model [m]
        c_s  = 800,     # specific heat for land (soil)  [J / K / kg]
        κ_s  = 0.0,     # soil conductivity [W / m / K] (set to 0 for energy conservation checks)
        T_h  = 280,     # temperature of soil at depth h [K]
        ρ_s  = 1500,    # density for land (soil) [kg / m^3]

        # radiation
        τ    = 0.9,     # atmospheric transmissivity
        α    = 0.5,     # surface albedo
        σ    = 5.67e-8, # Steffan Boltzmann constant [kg / s^3 / K^4]
        g_a  = 0.06,    # aerodynamic conductance for heat transfer [kg / m^2 / s]
        ϵ    = 0.98,    # broadband emissivity / absorptivity of the surface
        F_a  = 0.0,     # downward LW flux from the atmosphere [W / m^2]
        F_sol = 1361,   # incoming solar TOA radiation [W / m^2]
        τ_d   = 10,     # idealized daily cycle period [s]

        # surface fluxes
        λ = FT(0.01),#FT(1e-5)    # coupling transfer coefficient (to be replaced by the bulk formula)

        # Q: ML???
    )


########
# Set up atmos domain
########

domain_atm  = Domains.IntervalDomain(parameters.zmin_atm, parameters.zmax_atm, x3boundary = (:bottom, :top)) # struct
mesh_atm = Meshes.IntervalMesh(domain_atm, nelems = parameters.n) # struct, allocates face boundaries to 5,6
center_space_atm = Spaces.CenterFiniteDifferenceSpace(mesh_atm) # collection of the above, discretises space into FD and provides coords
face_space_atm = Spaces.FaceFiniteDifferenceSpace(center_space_atm)


########
# Set up inital conditions
########
function init_centers(zc, parameters)
    UnPack.@unpack T_surf, T_min_ref, grav, C_p, MSLP, R_d = parameters

    # temperature
    Γ = grav / C_p
    T= max(T_surf - Γ * zc, T_min_ref)

    # pressure
    p = MSLP * (T / T_surf)^(grav / (R_d * Γ))
    if T == T_min_ref
        z_top = (T_surf - T_min_ref) / Γ
        H_min = R_d * T_min_ref / grav
        p *= exp(-(zc - z_top) / H_min)
    end

    # potential temperature
    θ = T_surf

    # density
    ρ = p / (R_d * θ * (p / MSLP)^(R_d / C_p))

    # velocties
    u = 1.0
    v = 0.0

    return (ρ = ρ, u = u, v = v, ρθ = ρ * θ)
end

function init_faces(zf, parameters)
    return (; w = 0.0 .* zf)
end

# atmos IC state
z_centers = Fields.coordinate_field(center_space_atm)
z_faces = Fields.coordinate_field(face_space_atm)
Yc = init_centers.(z_centers, Ref(parameters))
Yf = init_faces.(z_faces, Ref(parameters))
T_atm_0 = (; Yc = Yc, Yf = Yf)

# land IC state
T_lnd_0 = [260.0] # initiates lnd progostic var

ics = (;
        atm = T_atm_0,
        lnd = T_lnd_0
        )


########
# Set up rhs! and BCs
########

# define model equations:
include("atmos_rhs_fully_coupled.jl") 

function ∑tendencies_lnd!(dT_sfc, T_sfc, (parameters, F_sfc), t)
    """
    Slab ocean:
    ∂_t T_sfc = F_sfc + G
    """
    p = parameters
    G = 0.0 # place holder for soil dynamics

    @. dT_sfc = (F_sfc * p.C_p + G) / (p.h_s * p.ρ_s * p.c_s)
end


########
# Set up time steppers
########

# specify timestepping info
stepping = (;
        Δt_min = parameters.Δt_min,
        timerange = parameters.timerange,
        Δt_cpl = max(parameters.nsteps_atm, parameters.nsteps_lnd) * parameters.Δt_min, # period of coupling cycle [s]
        odesolver = parameters.odesolver,
        nsteps_atm = parameters.nsteps_atm,
        nsteps_lnd = parameters.nsteps_lnd,
        )

# coupler comm functions which export / import / transform variables
coupler_get(x) = x
coupler_put(x) = x

# Solve the ODE operator
function coupler_solve!(stepping, ics, parameters)
    t = 0.0
    Δt_min  = stepping.Δt_min
    Δt_cpl  = stepping.Δt_cpl
    t_start = stepping.timerange[1]
    t_end   = stepping.timerange[2]
    nsteps_atm = stepping.nsteps_atm
    nsteps_lnd = stepping.nsteps_lnd

    # init coupler fields
    coupler_F_sfc = coupler_put([0.0])
    coupler_T_lnd = coupler_put(copy(ics.lnd))

    # atmos copies of coupler variables
    atm_T_lnd = coupler_get(copy(coupler_T_lnd))
    atm_F_sfc = coupler_get(copy(coupler_F_sfc))

    # SETUP ATMOS
    # put all prognostic variable arrays into a vector and ensure that solve can partition them
    T_atm = ics.atm
    Y_atm = ArrayPartition(( T_atm_0.Yc, T_atm_0.Yf , atm_F_sfc))
    prob_atm = ODEProblem(∑tendencies_atm!, Y_atm, (t_start, t_end), (parameters, atm_T_lnd))
    integ_atm = init(
                        prob_atm,
                        stepping.odesolver,
                        dt = Δt_cpl / nsteps_atm,
                        saveat = parameters.saveat,)

    # land copies of coupler variables
    T_lnd = ics.lnd
    lnd_F_sfc = copy(coupler_F_sfc)

    # SETUP LAND
    prob_lnd = ODEProblem(∑tendencies_lnd!, T_lnd, (t_start, t_end), (parameters, lnd_F_sfc))
    integ_lnd = init(
                        prob_lnd,
                        stepping.odesolver,
                        dt = Δt_cpl / nsteps_lnd,
                        saveat = parameters.saveat,)

    # coupler stepping
    for t in (t_start : Δt_cpl : t_end)

        ## Atmos
        # pre_atmos
        # coupler_get_atmosphere!(atmos_simulation, coupler)
        #   1) get the land state (e.g., temperature) from the coupler
        #   2) reset the flux accumulator to 0

        integ_atm.p[2] .= coupler_get(coupler_T_lnd) # get land temperature and set on atmosphere (Tland is prognostic)
        integ_atm.u.x[3] .= [0.0] # surface flux to be accumulated


        # run atmos
        # NOTE: use (t - integ_atm.t) here instead of Δt_cpl to avoid accumulating roundoff error in our timestepping.
        step!(integ_atm, t - integ_atm.t, true)

        # post_atmos
        # coupler_put_atmosphere!(atmos_simulation, coupler)
        #  1) compute the time-averaged flux to/from the land
        #  2) regrid
        #  3) store that in the coupler

        # negate sign
        coupler_F_sfc .= - coupler_put(integ_atm.u.x[3]) / Δt_cpl

        ## Land

        # pre_land
        # coupler_get_land!(land_simulation, coupler)
        #  1) get thet time-averaged flux from the coupler and save it in the `lnd_F_sfc` parameter of `integ_lnd`
        lnd_F_sfc .= coupler_get(coupler_F_sfc)

        # run land
        step!(integ_lnd, t - integ_lnd.t, true)

        # post land
        # coupler_put_land!(land_simulation, coupler)
        #  1) store required land state (e.g., temperature) in the coupler 
        coupler_T_lnd .= coupler_put(integ_lnd.u) # update T_sfc
    end

    return integ_atm, integ_lnd
end


########
# Run simulation
########

integ_atm, integ_lnd = coupler_solve!(stepping, ics, parameters)
sol_atm, sol_lnd = integ_atm.sol, integ_lnd.sol


########
# Visualisation
########

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "heat"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# animation
# anim = Plots.@animate for u in sol_atm.u
#     Plots.plot(u.x[1], xlim=(220,280))
# end
# Plots.mp4(anim, joinpath(path, "heat.mp4"), fps = 10)

# atmos vertical profile at t=0 and t=end
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

# time evolution
atm_sum_u_t = [sum(parent(u.x[1])[:,4]) for u in sol_atm.u] ./ parameters.n .* parameters.zmax_atm * parameters.C_p # J / m2
lnd_sfc_u_t = [u[1] for u in sol_lnd.u] * parameters.h_s * parameters.ρ_s * parameters.c_s # J / m2

v1 = lnd_sfc_u_t .- lnd_sfc_u_t[1]
v2 = atm_sum_u_t .- atm_sum_u_t[1]
Plots.png(Plots.plot(sol_lnd.t, [v1 v2 v1+v2], labels = ["lnd" "atm" "tot"]), joinpath(path, "energy_both_surface_time.png"))

# relative error
using Statistics
total = atm_sum_u_t + lnd_sfc_u_t
rel_error = (total .- total[1]) / mean(total)
Plots.png(Plots.plot(sol_lnd.t, rel_error, labels = ["tot"]), joinpath(path, "rel_error_surface_time.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/heat_end.png", "Heat End Simulation")

# TODO here
# - integrate CouplerMachine.jl
# - revamp for 2D 
# - integrate SurfaceFluxes.jl
# - revamp for 2D 

# Questions / Comments
# 

