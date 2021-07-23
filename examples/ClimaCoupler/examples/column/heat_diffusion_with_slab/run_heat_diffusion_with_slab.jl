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
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear

global_logger(TerminalLogger())

const CI = !isnothing(get(ENV, "CI", nothing))

# general parameters
const FT = Float64

# coupling parameters
λ = FT(1e-5) # transfer coefficient
calculate_flux(T_sfc, T1) = λ .* (T_sfc .- T1)

# domain parameters
zmin_atm = FT(0.0)
zmax_atm = FT(1.0)
zmin_lnd = FT(-1.0)
zmax_lnd = FT(0.0)

n = 10

# initiate model domain and grid
domain_atm  = Domains.IntervalDomain(zmin_atm, zmax_atm, x3boundary = (:bottom, :top)) # struct
#domain_lnd  = Domains.IntervalDomain(zmin_lnd, zmax_lnd, x3boundary = (:bottom, :top)) # struct

mesh_atm = Meshes.IntervalMesh(domain_atm, nelems = n) # struct, allocates face boundaries to 5,6: atmos
#mesh_lnd = Meshes.IntervalMesh(domain_lnd, nelems = 1) # struct, allocates face boundaries to 5,6: land

cs_atm = Spaces.CenterFiniteDifferenceSpace(mesh_atm) # collection of the above, discretises space into FD and provides coords
#cs_lnd = Spaces.CenterFiniteDifferenceSpace(mesh_lnd)

# define model equations:
function ∑tendencies_atm!(du, u, (parameters, coupler_T_sfc), t)
    # Heat diffusion:
    # ∂_t T = α ∇²T
    # where
    # ∂_t T = n \cdot F   at z = zmin_atm
    # ∂_t T = 0           at z = zmax_atm
    # We also use this model to accumulate fluxes
    # ∂_t ϕ_bottom = n \cdot F

    α = FT(0.1) # diffusion coefficient

    T = u.x[1]

    F_sfc = calculate_flux(coupler_T_sfc[1], parent(T)[1] )

    # set BCs
    bcs_bottom = Operators.SetValue(F_sfc) # struct w bottom BCs
    bcs_top = Operators.SetValue(FT(230.0))

    gradc2f = Operators.GradientC2F(top = bcs_top) # gradient struct w BCs
    gradf2c = Operators.GradientF2C(bottom = bcs_bottom)

    # tendency calculations
    @. du.x[1] = α * gradf2c(gradc2f(T)) 
    du.x[2] .= F_sfc[1]

    @show "here :)"
    @show t
end

function ∑tendencies_lnd!(dT_sfc, T_sfc, (parameters, coupler_F_sfc), t)
    """
    Slab ocean:
    ∂_t T_sfc = F_sfc + G
    """
    G = 0.0 # placeholder for soil dynamics

    dT_sfc = coupler_F_sfc .+ G
    #return @. dT_sfc = coupler_F_sfc .+ G
end

# initialize all variables and display models
parameters = nothing
T_atm_0 = Fields.ones(FT, cs_atm) .* 280 # initiates atm progostic var
T_sfc_0 = [260.0] # initiates lnd progostic var
ics = (;
        atm = T_atm_0,
        lnd = T_sfc_0
        )

# specify timestepping info
stepping = (;
        Δt_min = 0.02,
        timerange = (0.0, 3.0),
        Δt_cpl = 1.0,
        odesolver = SSPRK33(),
        nsteps_atm = 1,
        nsteps_lnd = 3,
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

    # init coupler fields
    coupler_F_sfc = [0.0]
    coupler_T_sfc = ics.lnd
    coupler_T_atm = ics.atm

    T_atm = coupler_T_atm
    F_sfc = coupler_F_sfc
    T_sfc = coupler_T_sfc
    
    # SETUP ATMOS
    # put all prognostic variable arrays into a vector and ensure that solve can partition them 
    Y = ArrayPartition((T_atm, F_sfc))
    prob_atm = ODEProblem(∑tendencies_atm!, Y, (t, t + Δt_cpl), (parameters, T_sfc) )
    integ_atm = init(
                        prob_atm,  
                        stepping.odesolver,
                        tstops = collect(t : Δt_min : t + Δt_cpl),
                        saveat = 10 * Δt_min,)

    # SETUP LAND
    F_sfc = copy(coupler_F_sfc)
    prob_lnd = ODEProblem(∑tendencies_lnd!, T_sfc, (t, t + Δt_cpl), (parameters, F_sfc))
    integ_lnd = init(
                        prob_lnd,  
                        stepping.odesolver,
                        tstops = collect(t : Δt_min : t + Δt_cpl),
                        saveat = 10 * Δt_min,)

    # coupler stepping
    for t in collect(t_start : Δt_cpl : t_end)
        #@show t

        # pre_atmos
        T_sfc .= coupler_get(coupler_T_sfc)
        T_atm .= coupler_get(coupler_T_atm)
        F_sfc .= [0.0] # surfce flux to be accumulated 

        # run atmos
        add_tstop!(integ_atm, t)
        solve!(integ_atm)

        # post_atmos
        coupler_T_atm = coupler_put(integ_atm.sol.u[end].x[1])
        coupler_F_sfc = coupler_put(integ_atm.sol.u[end].x[2])  / Δt_cpl

        # pre_land
        F_sfc = coupler_get(coupler_F_sfc)
        T_sfc = coupler_get(coupler_T_sfc)

        # run land
        add_tstop!(integ_lnd, t)
        solve!(integ_lnd)

        # post land
        coupler_T_sfc = coupler_put(integ_lnd.sol.u[end]) # update T_sfc
        coupler_F_sfc = coupler_F_sfc .* 0.0 # reset accumulated surcafe flux at the end of coupling cycle

    end

    return integ_atm.sol, integ_lnd.sol
end

# run
sol_atm, sol_lnd = coupler_solve!(stepping, ics, parameters)

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "heat"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol_atm.u
    Plots.plot(u.x[1], xlim = (0, 1))
end
Plots.mp4(anim, joinpath(path, "heat.mp4"), fps = 10)
Plots.png(Plots.plot(sol_atm.u[end].x[1] ), joinpath(path, "heat_end.png"))


u_t = copy(parent(sol_atm.u[1].x[1]))[:,1]

Plots.png(Plots.plot(u_t ), joinpath(path, "heat_time.png"))

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