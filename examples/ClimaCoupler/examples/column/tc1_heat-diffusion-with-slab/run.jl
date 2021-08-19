using Base: show_supertypes
#push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

# add https://github.com/CliMA/ClimaCore.jl
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

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger

using RecursiveArrayTools

using OrdinaryDiffEq, Test, Random

using Statistics

global_logger(TerminalLogger())

const CI = !isnothing(get(ENV, "CI", nothing))

# general parameters
const FT = Float64

parameters = (
    # atmos parameters
    zmin_atm = FT(0.0), # height of atm stack bottom [m]
    zmax_atm = FT(1.0), # height of atm stack top [m]
    n = 15,  # number of elements in atm stack 
    μ = FT(0.0001), # diffusion coefficient
    T_top = FT(280.0), # fixed temperature at the top of the domain_atm
    T_atm_ini = FT(280.0), # initial condition of at temperature (isothermal) [K]
    # slab parameters
    h_lnd = FT(0.5), # depth of slab layer [m]
    T_lnd_ini = FT(260.0), # initial condition of at temperature (isothermal) [K]
    # coupling parameters 
    λ = FT(1e-5), # transfer coefficient 
)

# surface flux calculation (coarse bulk formula)
calculate_flux(T_sfc, T1, parameters) = - parameters.λ * (T_sfc - T1)

# initiate atm model domain and grid
domain_atm  = Domains.IntervalDomain(parameters.zmin_atm, parameters.zmax_atm, x3boundary = (:bottom, :top)) # struct
mesh_atm = Meshes.IntervalMesh(domain_atm, nelems = parameters.n) # struct, allocates face boundaries to 5,6: atmos
center_space_atm = Spaces.CenterFiniteDifferenceSpace(mesh_atm) # collection of the above, discretises space into FD and provides coords

# define model equations
function ∑tendencies_atm!(du, u, (parameters, T_sfc), t)
    """
    Heat diffusion equation
        dT/dt =  ∇ μ ∇ T
        where
            T  = 280 K              at z = zmax_atm
            dT/dt = - ∇ F_sfc       at z = zmin_atm
    
    We also use this model to calculate and accumulate the downward surface fluxes, F_sfc:
        F_sfc = - λ * (T_sfc - T1) 
        d(F_integrated)/dt  = F_sfc
        where
            F_integrated is reset to 0 at the beginning of each coupling cycle
            T1 = atm temperature near the surface (here assumed equal to the first model level)
    """
    
    T = u.x[1] # u.x = vector of prognostic variables from DifferentialEquations
    F_sfc = calculate_flux(T_sfc[1], parent(T)[1], parameters)

    # set BCs
    bcs_bottom = Operators.SetValue(F_sfc) 
    bcs_top = Operators.SetValue(FT(parameters.T_top))

    gradc2f = Operators.GradientC2F(top = bcs_top) # Dirichlet BC
    gradf2c = Operators.GradientF2C(bottom = bcs_bottom) # Neumann BC

    # tendency calculations
    @. du.x[1] = gradf2c( parameters.μ * gradc2f(T)) # dT/dt
    du.x[2] .= - F_sfc[1] # d(F_integrated)/dt

end

function ∑tendencies_lnd!(dT_sfc, T_sfc, (parameters, F_accumulated), t)
    """
    Slab layer equation
        lnd d(T_sfc)/dt = - F_accumulated + G
        where 
            F_accumulated = F_integrated / Δt_coupler
    """
    G = 0.0 # place holder for soil dynamics
    @. dT_sfc = ( - F_accumulated + G) / parameters.h_lnd 
end

# initialize all variables and display models
T_atm_0 = Fields.ones(FT, center_space_atm) .* parameters.T_atm_ini # initiates a spatially uniform atm progostic var
T_lnd_0 = [parameters.T_lnd_ini] # initiates lnd progostic var
ics = (;
        atm = T_atm_0,
        lnd = T_lnd_0
        )

# specify timestepping info
stepping = (;
        Δt_min = 0.02,
        timerange = (0.0, 6.0),
        Δt_coupler = 1.0,
        odesolver = SSPRK33(),
        nsteps_atm = 8, # number of timesteps of atm per coupling cycle
        nsteps_lnd = 1, # number of timesteps of lnd per coupling cycle
        )

# coupler comm functions which export / import / transform variables (for now just place holders)
coupler_get(x) = x
coupler_put(x) = x

# Solve the ODE operator
function coupler_solve!(stepping, ics, parameters)
    t = 0.0
    Δt_min  = stepping.Δt_min
    Δt_coupler  = stepping.Δt_coupler
    t_start = stepping.timerange[1]
    t_end   = stepping.timerange[2]

    # init coupler fields
    coupler_F_sfc = [0.0]
    coupler_T_lnd = copy(ics.lnd)

    # atmos copies of coupler variables
    atm_T_lnd = copy(coupler_T_lnd)
    atm_F_sfc = copy(coupler_F_sfc)

    # SETUP ATMOS
    # put all prognostic variable arrays into a vector and ensure that solve can partition them
    T_atm = ics.atm
    Y_atm = ArrayPartition((T_atm, atm_F_sfc))
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
    for t in (t_start : Δt_coupler : t_end)

        ## Atmos
        # pre_atmos
        integ_atm.p[2] .= coupler_get(coupler_T_lnd) # integ_atm.p is the parameter vector of an ODEProblem from DifferentialEquations
        integ_atm.u.x[2] .= [0.0] # surface flux to be accumulated

        # run atmos
        # NOTE: use (t - integ_atm.t) here instead of Δt_coupler to avoid accumulating roundoff error in our timestepping.
        step!(integ_atm, t - integ_atm.t, true)

        # post_atmos
        coupler_F_sfc .= coupler_put(integ_atm.u.x[2]) / Δt_coupler

        ## Land
        # pre_land
        lnd_F_sfc .= coupler_get(coupler_F_sfc)
        
        # run land
        step!(integ_lnd, t - integ_lnd.t, true)

        # post land
        coupler_T_lnd .= coupler_put(integ_lnd.u) # update T_sfc
    end

    return integ_atm, integ_lnd
end


# run
integ_atm, integ_lnd = coupler_solve!(stepping, ics, parameters)
sol_atm, sol_lnd = integ_atm.sol, integ_lnd.sol

# plots and conservation check
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
t0_ = parent(sol_atm.u[1].x[1])[:,1]
tend_ = parent(sol_atm.u[end].x[1])[:,1]
z_centers =  parent(Fields.coordinate_field(center_space_atm))[:,1]
Plots.png(Plots.plot([t0_ tend_],z_centers, labels = ["t=0" "t=end"]), joinpath(path, "T_atm_height.png"))

atm_sfc_u_t = [parent(u.x[1])[1] for u in sol_atm.u]
Plots.png(Plots.plot(sol_atm.t, atm_sfc_u_t), joinpath(path, "T_atmos_surface_time.png"))

lnd_sfc_u_t = [u[1] for u in sol_lnd.u]
Plots.png(Plots.plot(sol_lnd.t, lnd_sfc_u_t), joinpath(path, "T_land_surface_time.png"))

# convert to the same units (analogous to energy conservation, assuming that is both domains density=1 and thermal capacity=1)
lnd_sfc_u_t = [u[1] for u in sol_lnd.u] .* parameters.h_lnd
atm_sum_u_t = [sum(parent(u.x[1])[:]) for u in sol_atm.u] .* (parameters.zmax_atm - parameters.zmin_atm) ./ parameters.n

v1 = lnd_sfc_u_t .- lnd_sfc_u_t[1] 
v2 = atm_sum_u_t .- atm_sum_u_t[1] 
Plots.png(Plots.plot(sol_lnd.t, [v1 v2 v1+v2], labels = ["lnd" "atm" "tot"]), joinpath(path, "Δenergy_both_surface_time.png"))

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

# Next steps
# - extend atmos physics to Ekman Column
# - use ClimaAtmos interface & optimise the coupler_solve! functions accordingly
# - use coupler module functions 

# Refs:
# - ODEProblem(f,u0,tspan; _..) https://diffeq.sciml.ai/release-2.1/types/ode_types.html
# - for options for solve, see: https://diffeq.sciml.ai/stable/basics/common_solver_opts/