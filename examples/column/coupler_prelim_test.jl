push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

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
global_logger(TerminalLogger())

const CI = !isnothing(get(ENV, "CI", nothing))

const FT = Float64

a = FT(0.0)
b = FT(1.0)
n = 10
α = FT(0.1)

domain = Domains.IntervalDomain(a, b, x3boundary = (:bottom, :top)) # struct
mesh_atmos = Meshes.IntervalMesh(domain, nelems = n) # struct, allocates face boundaries to 5,6: atmos
mesh_land = Meshes.IntervalMesh(domain, nelems = 1) # struct, allocates face boundaries to 5,6: land

cs_atmos = Spaces.CenterFiniteDifferenceSpace(mesh_atmos) # collection of the above, discretises space into FD and provides coords
cs_land  = Spaces.CenterFiniteDifferenceSpace(mesh_land) 

T = Fields.zeros(FT, cs_atmos) # initiates progostic var
T_sfc = Fields.zeros(FT, cs_land) .* 260 # initiates progostic var
bottom_flux = Fields.zeros(FT, cs_land) # initiates progostic var

# Solve Heat Equation: ∂_t T = α ∇²T
u0 = [T; 0.0]
function ∑tendencies_atmos!(du, u, _, t, T_sfc)

    T = u[1]
    bottom_flux = u[2]
    bottom_flux = 10e-5 .* (T_sfc .- copy(parent(T))[1])
    bcs_bottom = Operators.SetValue(bottom_flux) # struct w bottom BCs
    bcs_top = Operators.SetValue(FT(230.0))

    gradc2f = Operators.GradientC2F(bottom = bcs_bottom, top = bcs_top) # gradient struct w BCs
    gradf2c = Operators.GradientF2C()

    #bottom_flux_accum = bottom_flux #TODO

    du[1] = T #α .* gradf2c(gradc2f(T))
    du[2] = copy(parent(bottom_flux))
    return @. du
end
@show ∑tendencies_atmos!(similar(u0), u0, nothing, 0.0, T_sfc); 

function ∑tendencies_land!(dT, T_sfc, _, t, bottom_flux)

    # bcs_bottom = Operators.SetValue(FT(0.0)) # struct w bottom BCs
    # bcs_top = Operators.SetGradient(FT(1.0))

    return @. dT = bottom_flux
end
@show ∑tendencies_land!(similar(T_sfc), T_sfc, nothing, 0.0, bottom_flux); 

# Solve the ODE operator
Δt = 0.02

prob = ODEProblem(∑tendencies!, u0, (0.0, 10.0)) #ODEProblem(f,u0,tspan; _..) https://diffeq.sciml.ai/release-2.1/types/ode_types.html
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10 * Δt,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);  
# ~/.julia/packages/DiffEqBase/NarCz/src/solve.jl:66
# for options for solve, see: https://diffeq.sciml.ai/stable/basics/common_solver_opts/

#T_sfc = copy(parent(sol.u[1]))[end] # update T_sfc = be top of the

prob = ODEProblem(∑tendencies!, T_sfc, (10.0, 20.0)) #ODEProblem(f,u0,tspan; _..) https://diffeq.sciml.ai/release-2.1/types/ode_types.html
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10 * Δt,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);  


ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "heat"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u, xlim = (0, 1))
end
Plots.mp4(anim, joinpath(path, "heat.mp4"), fps = 10)
Plots.png(Plots.plot(sol.u[end], xlim = (0, 1)), joinpath(path, "heat_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/heat_end.png", "Heat End Simulation")


# Questions / Comments
# - ok to add bottom flux as prognostic variable again? 
# - MPIStateArray overhead issue doesn't apply
# - coupler src code can still be used, ust the do_step function needs to be rewritten
# - quite hard to find original functions e.g. which solve etc
# - extracting values from individual levels is quite clunky
# - Fields don't seem to contain variable names... (maybe?)