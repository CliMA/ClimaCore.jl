# to do:
# rename soil_bounds field in roots? p_ups -> p_soil?
# for loops necessary?
# profile to see where time is taken
# compare to expected case
# is it better to compute an average flow based on mean(ψ_{soil, i} -ψ_root), or to
# compute an average ψ_{soil} and then compute a flow, mean(ψ_{soil, i}) -ψ_root?
# conversion factors correct?

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

# using SPAC node
using SoilPlantAirContinuum
using CLIMAParameters:AbstractEarthParameterSet
using CLIMAParameters.Planet: ρ_cloud_liq, grav, molmass_water
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()


using OrdinaryDiffEq: ODEProblem, solve, SSPRK33,Rosenbrock23, Tsit5,SSPRK432,TsitPap8
using Plots
using Dierckx
using BenchmarkTools

const FT = Float64

## Helper functions
function update_root_system!(node,soil_matric_potential, param_set)
    # compute conversion factor (MPa from m)
    ρ_w = FT(ρ_cloud_liq(param_set)) #kg/m^3
    g = FT(grav(param_set))
    cf = ρ_w * g * FT(1e-6)
    for _i_root in eachindex(node.plant_hs.roots)
        node.plant_hs.roots[_i_root].p_ups = soil_matric_potential[_i_root] * cf;
    end
end


function evaluate_steady_state_root_model!(node, ψ_at_boundaries, param_set)
    # compute conversion factor (1/s from moles/s)
    update_root_system!(node, ψ_at_boundaries, param_set)
    layer_fluxes!(node);
    molar_mass_of_water = FT(molmass_water(param_set)) # kg/mol
    v_soil = node.ga * root_layer_width # m^3
    ρ_w = FT(ρ_cloud_liq(param_set)) # kg/m^3
    flows = node.plant_hs.cache_q
    volumetric_water_content_roc = flows .* molar_mass_of_water ./ ρ_w ./v_soil 
    return volumetric_water_content_roc
end


#function advance_root_model!(node, ψ_at_boundaries, Δt, param_set)
#    # compute conversion factor (1/s from moles/s)
#    update_root_system!(node, ψ_at_boundaries, param_set)
#    layer_fluxes!(node, Δt);
#    molar_mass_of_water = FT(molmass_water(param_set)) # kg/mol
#    v_soil = node.ga * root_layer_width # m^3
#    ρ_w = FT(ρ_cloud_liq(param_set)) # kg/m^3
#    flows = node.plant_hs.cache_q
#    volumetric_water_content_roc = flows .* molar_mass_of_water ./ ρ_w ./v_soil 
#    return volumetric_water_content_roc
#end


function square(z,z1,z2)
    result = ((z> z1) & (z <= z2)) ? FT(1) : FT(0)
    return result
end

function root_layers(root_boundaries, root_sink_at_roots)
    base = (z) -> FT(0)
    for i in 1:1:length(root_sink_at_roots)
        new_function = (z) -> root_sink_at_roots[i]*square(z,root_boundaries[i+1], root_boundaries[i]) 
        base = let base = base; (z) -> (base(z) + new_function(z)); end
        
    end
    return base
end



### create soil system
n = 15
zmin = FT(-0.2)
zmax = FT(0)
ksat = FT(0 / (3600 * 100))
vgn = FT(3.96)
vgα = FT(2.7)
vgm = FT(1)- FT(1)/vgn
θr = FT(0.075)
ν = FT(0.287)
Δt = FT(0.05)
tf = FT(800*Δt)
t0 = FT(0)


domain = Domains.IntervalDomain(zmin, zmax, x3boundary = (:bottom, :top))
mesh = Meshes.IntervalMesh(domain, nelems = n)

cs = Spaces.CenterFiniteDifferenceSpace(mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)
zc = Fields.coordinate_field(cs)

function effective_saturation(θ; ν = ν, θr = θr)
    return (θ-θr)/(ν-θr)
end

function matric_potential(S; vgn = vgn, vgα = vgα, vgm = vgm)
    ψ_m = -((S^(-FT(1) / vgm) - FT(1)) * vgα^(-vgn))^(FT(1) / vgn)
    return ψ_m
end

function hydraulic_conductivity(S; vgm = vgm, ksat = ksat)
    if S < FT(1)
        K = sqrt(S) * (FT(1) - (FT(1) - S^(FT(1) / vgm))^vgm)^FT(2)
    else
        K = FT(1)
    end
    return K*ksat
end

# initial soil conditions
function hydrostatic_profile(z, zmin, porosity, n, α, θr)
    myf = eltype(z)
    m = FT(1 - 1 / n)
    S = FT((FT(1) + (α * (z - zmin))^n)^(-m))
    return FT(S * (porosity-θr)+θr)
end
θl = hydrostatic_profile.(zc, FT(-0.5), ν, vgn, vgα, θr)
S0 = effective_saturation.(θl;ν =ν, θr = θr)
ψ0 = matric_potential.(S0; vgn = vgn, vgα = vgα, vgm = vgm)

# create root system & initialize

root_layer_width = FT(0.05)
root_boundaries = root_boundaries=collect(FT,0:-root_layer_width:-0.15)
maximum_root_depth = FT(-0.15)
node=SPACMono{FT}(soil_bounds=root_boundaries,z_root=maximum_root_depth);
initialize_spac_canopy!(node);
layer_fluxes!(node);
#interpolate to root boundaries
myspline = Spline1D(parent(zc)[:], parent(ψ0)[:])
ψ0_at_boundaries = myspline(root_boundaries)
update_root_system!(node, ψ0_at_boundaries,param_set)
Nroots = length(node.plant_hs.roots)


# Solve Richard's Equation: ∂_t θl = ∂_z [K(θl)(∂_z ψ(θl) +1)] + S(θr, θl)
Nsteps= Int(round((tf-t0)/Δt))
save_every_N =10

dons = Dict([0 => Dict()])

function ∑tendencies!(dθl, θl, z, t)
    bc_t = Operators.SetGradient(FT(0))
    bc_b = Operators.SetGradient(FT(0))
    
    gradc2f = Operators.GradientC2F(bottom = bc_b, top = bc_t)
    gradf2c = Operators.GradientF2C()
    
    S = effective_saturation.(θl; ν = ν, θr = θr)
    K = hydraulic_conductivity.(S; vgm = vgm, ksat = ksat)
    ψ = matric_potential.(S; vgn = vgn, vgα = vgα, vgm = vgm)
    h = ψ .+ zc


    # No flux at top and bottom -> center = face
    K_top = hydraulic_conductivity(parent(S)[end]; vgm = vgm, ksat = ksat)
    bc_ktop = Operators.SetValue(K_top)
    K_bot = hydraulic_conductivity(parent(S)[1]; vgm = vgm, ksat = ksat)
    bc_kbot = Operators.SetValue(K_bot)
    If = Operators.InterpolateC2F(;
                                  bottom = bc_kbot,
                                  top = bc_ktop)

    if add_roots
        step = (t-t0)/Δt
        if step % save_every_N == 0
            state = Dict{String, Array}(
                "t" => [t],
                "ϑl" => parent(θl)[:],
                "roots" => node.plant_hs.cache_p,
            )
            dons[Int(step/save_every_N)] = state
        end
        
        
        # interpolate θ to the root z values
        myspline = Spline1D(parent(zc)[:], parent(ψ)[:])
        ψ_at_roots= myspline.(root_boundaries) 
        root_sink_at_roots = evaluate_steady_state_root_model!(node, parent(ψ_at_roots)[:], param_set) ## advances the roots, updates node in place
        root_layer_mask = root_layers(root_boundaries, root_sink_at_roots)
        root_sink_at_soil = root_layer_mask.(zc)
        return @. dθl = gradf2c( If(K).* gradc2f(h)) .- root_sink_at_soil
    else
        
        step = (t-t0)/Δt
        println(step)
        if step % save_every_N == 0
            println(Int(step/save_every_N))
            
            state = Dict{String, Array}(
                "t" => [t],
                "ϑl" => parent(θl)[:],
            )
            dons[Int(step/save_every_N)] = state
        end
        return @. dθl = gradf2c( If(K).* gradc2f(h))
    end
    
    
end

# Solve the ODE operator

prob = ODEProblem(∑tendencies!, θl, (t0, tf))

saveat  = save_every_N*Δt
algo = Tsit5()
dt = Δt
p = true
pm = (dt, u, p, t) -> t
function simulate(prob,algo, dt, saveat, p, pm)
    
    sol = solve(
        prob,
        algo,#
        dt= dt,
        saveat = saveat,
        progress = p,
        progress_message = pm,
    );
    return sol
end
add_roots = false
sol_no_roots = @time simulate(prob,algo, dt, saveat, p, pm);
#  0.000763 seconds (3.28 k allocations: 586.672 KiB)

#add_roots = true
#sol_roots = @time simulate(prob,algo, dt, saveat, p, pm);
#   0.093268 seconds (1.30 M allocations: 41.863 MiB)
# 30% speedup if using `advance` rather than `evaluate`





# Plot soil solution only
#=
dirname = "richards"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()
N = length(sol_roots.t)

anim = @animate for k in 1:2:N
    t = sol_roots.t[k]
    Plots.plot(
    parent(sol_roots.u[k]),parent(zc),
    xlim = (0, 0.287),
    ylim = (-0.2, 0),
        title = string(string(Int(round(t)))," sec"),
        lc = :green,
        lw = 2,
        ls = :dash,
        label = "Roots",
        m = :o,
        xlabel = "θ(z)",
        ylabel = "z",
    )
    Plots.plot!(
        parent(sol_no_roots.u[k]),parent(zc),
        lc = :black,
        lw = 2,
        ls = :dash,
        label = "No roots",
    legend = :outerright,
    m = :o,
    )

end
Plots.gif(anim, joinpath(path, "richards_sandy_grass.gif"), fps = 10)
=#
