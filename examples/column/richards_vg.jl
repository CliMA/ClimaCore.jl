#push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

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

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33,Rosenbrock23, Tsit5,SSPRK432
using Plots

const FT = Float64

n = 60
z₀ = FT(-1.5)
z₁ = FT(0)
ksat = FT(34 / (3600 * 100))
vgn = FT(3.96)
vgα = FT(2.7)
vgm = FT(1)- FT(1)/vgn
θr = FT(0.075)
ν = FT(0.287)
θl_0 = FT(0.1)
θl_surf = FT(0.267)
Δt = FT(0.5)
tf = FT(60 * 60 * 0.8)
t0 = FT(0)


domain = Domains.IntervalDomain(z₀, z₁, x3boundary = (:bottom, :top))
mesh = Meshes.IntervalMesh(domain, nelems = n)

cs = Spaces.CenterFiniteDifferenceSpace(mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)
zc = Fields.coordinate_field(cs)
#zp = (z₀ + z₁ / n / 2):(z₁ / n):(z₁ - z₁ / n / 2)

###doesnt work without kwargs?
###ERROR: cannot infer concrete eltype of effective_saturation on (Float64,)

function effective_saturation(θ; ν = ν, θr = θr)

    if θ < θr
        println("Effective saturation is negative")
        println(θ)
    end

    if θ > ν
        println("Effective saturation is positive")
        println(θ)
    end
    return (θ-θr)/(ν-θr)
end

function matric_potential(S; vgn = vgn, vgα = vgα, vgm = vgm)
    if S < FT(0)
        println("Effective saturation is negative")
        println(S)
    end

    if S > FT(1)
        println("Effective saturation is positive")
        println(S)
    end
    
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


θl = Fields.zeros(FT, cs) .+ θl_0

# Solve Richard's Equation: ∂_t θl = ∂_z [K(θl)(∂_z ψ(θl) +1)]


function ∑tendencies!(dθl, θl, z, t)
    #the BC needs to be on the thing we take the gradient of? no, not per Jake.
    # try putting a dirichlet BC on state variable at top and on grad(h) at bottom
    θ_top = θl_surf
    S_top = effective_saturation.(θ_top; ν = ν, θr = θr)
    h_top = matric_potential(S_top; vgn = vgn, vgα = vgα, vgm = vgm)#ztop = 0
    bc_t = Operators.SetValue(h_top)
    bc_b = Operators.SetGradient(FT(1))

    gradc2f = Operators.GradientC2F(bottom = bc_b, top = bc_t)
    gradf2c = Operators.GradientF2C()
    
    S = effective_saturation.(θl; ν = ν, θr = θr)
    K = hydraulic_conductivity.(S; vgm = vgm, ksat = ksat)
    ψ = matric_potential.(S; vgn = vgn, vgα = vgα, vgm = vgm)
    h = ψ .+ zc
    #In this case, we have a dirichlet on h at the top -> known top value of
    # θ. so we should have a known value of K at the top to agree with.
    K_top = hydraulic_conductivity(S_top; vgm = vgm, ksat = ksat)
    bc_ktop = Operators.SetValue(K_top)
    # At the bottom, we have a BC on grad(h), or dh/dθ grad(θ). 
    # does that mean bc_val = dh/θ_last center at bottom (θ_center - θ_bottom_face)/Δ/2?
    # so we have constrained θ_bottom_face? then K_bottom_face should agree with this.
    #probably shouldnt be an extrapolation
    If = Operators.InterpolateC2F(;
                                  bottom = Operators.Extrapolate(),
                                  top = bc_ktop)
    return @. dθl = gradf2c( If(K).* gradc2f(h))
    
end
#@show ∑tendencies!(similar(θl), θl, nothing, 0.0);

# Solve the ODE operator

prob = ODEProblem(∑tendencies!, θl, (t0, tf))
sol = solve(
    prob,
    Rosenbrock23(),#Tsit5(),
    dt = Δt,
    saveat = 60 * Δt,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);


dirname = "richards"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()
datapath = joinpath(@__DIR__,"..","..","code")
data = joinpath(datapath, "sand_bonan_sp801.csv")
    ds_bonan = readdlm(data, ',')
    bonan_moisture = reverse(ds_bonan[:, 1])
    bonan_z = reverse(ds_bonan[:, 2]) ./ 100.0
anim = @animate for (nt, u) in enumerate(sol.u)
    Plots.plot(
        parent(u),parent(zc),
        xlim = (0, 0.287),
        ylim = (-1.5, 0),
        title = string(string(Int(round(((nt-1)*Δt*60-t0)/60)))," min"),
        lc = :black,
        lw = 2,
        ls = :dash,
        label = "",
        legend = :outerright,
        m = :o,
        xlabel = "θ(z)",
        ylabel = "z",
    )

    Plots.plot!(bonan_moisture, bonan_z, label = "Bonan solution, 48min", lw = 2, lc = :red)
end
Plots.gif(anim, joinpath(path, "richards_sand.gif"), fps = 10)
