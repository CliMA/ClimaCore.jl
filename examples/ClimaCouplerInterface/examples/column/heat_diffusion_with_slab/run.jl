    using Base: show_supertypes
#push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

# add https://github.com/CliMA/ClimaCore.jl
# add https://github.com/CliMA/ClimaAtmos.jl

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
@boilerplate

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger

using RecursiveArrayTools

using OrdinaryDiffEq, Test, Random

global_logger(TerminalLogger())

const CI = !isnothing(get(ENV, "CI", nothing))

# general parameters
const FT = Float64

########
# Set up parameters
########

parameters = (
        # atmos domain
        zmin_atm = FT(0.0),     # height of atmos stack bottom
        zmax_atm = FT(200.0), # height of atmos domain top
        n = 30,                 # number of vertical levels

        # atmos physics (overwriting CliMAParameters defaults?)
        T_surf = 300.0, # K
        T_min_ref = 230.0, #K
        MSLP = 1e5, # mean sea level pressure [Pa]
        grav = 9.8, # gravitational constant [m /s^2]
        R_d = 287.058, # R dry (gas constant / mol mass dry air)  [J / K / kg]
        C_p = 287.058 * 1.4 / (1.4 - 1), # heat capacity at constant pressure [J / K / kg]
        C_v = 287.058 / (1.4 - 1), # heat capacity at constant volume [J / K / kg]
        R_m = 87.058, # moist R, assumed to be dry [J / K / kg]
        f = 5e-5, # Coriolis parameters [1/s]
        ν = .01, # viscosity, diffusivity
        Cd = 0.01 / (2e2 / 30.0), #drag coeff
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
        λ = FT(1e-5)    # coupling transfer coefficient (to be replaced by the bulk formula)

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
        Δt_min = 0.01,
        timerange = (0.0, 6.0),
        Δt_cpl = 1.0,
        odesolver = SSPRK33(),
        nsteps_atm = 4,
        nsteps_lnd = 1,
        )

# coupling parameters
calculate_flux(T_sfc, T1) = parameters.λ .* (T_sfc .- T1)

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
                        dt = Δt_min,
                        saveat = 10 * Δt_min,)

    # land copies of coupler variables
    T_lnd = ics.lnd
    lnd_F_sfc = copy(coupler_F_sfc)

    # SETUP LAND
    prob_lnd = ODEProblem(∑tendencies_lnd!, T_lnd, (t_start, t_end), (parameters, lnd_F_sfc))
    integ_lnd = init(
                        prob_lnd,
                        stepping.odesolver,
                        dt = Δt_min,
                        saveat = 10 * Δt_min,)

    # coupler stepping
    for t in (t_start : Δt_cpl : t_end)
        # integ_atm.p = (atm_parameters, atm_T_lnd)
        # integ_atm.u.x = (T_atm_0.Yc, T_atm_0.Yf , atm_F_sfc)


         ## Atmos
         # pre_atmos
        # coupler_get_atmosphere!(atmos_simulation, coupler)
        #   1) get the land temperature from the coupler
        #   2) regrid
        #   3) reset the flux accumulator to 0

         integ_atm.p[2] .= coupler_get(coupler_T_lnd) # get land temperature and set on atmosphere (Tland is prognostic)
         integ_atm.u.x[3] .= [0.0] # surface flux to be accumulated


         # run atmos
         # NOTE: use (t - integ_atm.t) here instead of Δt_cpl to avoid accumulating roundoff error in our timestepping.
         step!(integ_atm, t - integ_atm.t, true)

         # post_atmos
        # coupler_put_atmosphere!(atmos_simulation, coupler)
        #  1) compute the time-averaged flux to/from the land
        #  2) store that in the coupler

        # negate sign
         coupler_F_sfc .= - coupler_put(integ_atm.u.x[3]) / Δt_cpl

        ## Land
        # integ_lnd.p = (lnd_parameters, atm_T_lnd)
        # integ_atm.u.x = (T_atm_0.Yc, T_atm_0.Yf , atm_F_sfc)

        # pre_land
        # coupler_get_land!(land_simulation, coupler)
        #  1) get thet time-averaged flux from the coupler

        lnd_F_sfc .= coupler_get(coupler_F_sfc)
        #@show lnd_F_sfc
        # run land
        step!(integ_lnd, t - integ_lnd.t, true)

        # post land
        # coupler_put_land!(land_simulation, coupler)
        #  1) 
        coupler_T_lnd .= coupler_put(integ_lnd.u) # update T_sfc
    end

    return integ_atm, integ_lnd
end


# run
integ_atm, integ_lnd = coupler_solve!(stepping, ics, parameters)
sol_atm, sol_lnd = integ_atm.sol, integ_lnd.sol

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "heat"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol_atm.u
    Plots.plot(u.x[1], xlim=(220,280))
end
Plots.mp4(anim, joinpath(path, "heat.mp4"), fps = 10)
Plots.png(Plots.plot(sol_atm.u[end].x[1] ), joinpath(path, "T_atm_end.png"))

atm_sfc_u_t = [parent(u.x[1])[1] for u in sol_atm.u]
Plots.png(Plots.plot(sol_atm.t, atm_sfc_u_t), joinpath(path, "T_atmos_surface_time.png"))

lnd_sfc_u_t = [u[1] for u in sol_lnd.u]
Plots.png(Plots.plot(sol_lnd.t, lnd_sfc_u_t), joinpath(path, "T_land_surface_time.png"))

atm_sum_u_t = [sum(parent(u.x[1])[:]) for u in sol_atm.u] ./ parameters.n

v1 = lnd_sfc_u_t .- lnd_sfc_u_t[1]
v2 = atm_sum_u_t .- atm_sum_u_t[1]
Plots.png(Plots.plot(sol_lnd.t, [v1 v2 v1+v2], labels = ["lnd" "atm" "tot"]), joinpath(path, "heat_both_surface_time.png"))
Plots.png(Plots.plot(sol_lnd.t, [v1+v2], labels = ["tot"]), joinpath(path, "heat_total_surface_time.png"))


function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/heat_end.png", "Heat End Simulation")

# TODO
# - add flux accumulation ()®ecursive array error

# Questions / Comments
# - ok to add bottom flux as prognostic variable again?
# - MPIStateArray overhead issue doesn't apply
# - coupler src code can still be used, ust the do_step function needs to be rewritten
# - quite hard to find original functions e.g. which solve etc
# - extracting values from individual levels is quite clunky
# - Fields don't seem to contain variable names... (maybe?)

# Refs:

# ODEProblem(f,u0,tspan; _..) https://diffeq.sciml.ai/release-2.1/types/ode_types.html
    # ~/.julia/packages/DiffEqBase/NarCz/src/solve.jl:66
    # for options for solve, see: https://diffeq.sciml.ai/stable/basics/common_solver_opts/