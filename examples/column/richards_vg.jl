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
using DiffEqCallbacks
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33,Rosenbrock23, Tsit5,SSPRK432, Feagin14, TsitPap8,CarpenterKennedy2N54
using Plots
using DelimitedFiles
using Printf
using UnPack

const FT = Float64

struct SoilParams{FT <: AbstractFloat}
    ν::FT
    vgn::FT
    vgα::FT
    vgm::FT
    ksat::FT
    θr::FT
    S_s::FT
end

abstract type bc end

struct θDirichlet{FT <: AbstractFloat} <: bc
    θvalue::FT
end
struct FreeDrainage <: bc end
struct RunoffBC{FT <: AbstractFloat} <: bc
    rain_rate::FT
end
struct NormalFlux{FT <: AbstractFloat} <: bc
    #bc actually on minus the flux
    negative_flux_val::FT
end

function effective_saturation(θ; ν = ν, θr = θr)
    
    if θ < θr
        println("Effective saturation is negative")
        println(θ)
    end
    return (θ-θr)/(ν-θr)
end
function matric_potential(S; vgn = vgn, vgα = vgα, vgm = vgm)
    mft = eltype(S)
    if S < mft(0)
        println("Effective saturation is negative")
        println(S)
    end
    if S < mft(1)
        ψ = -((S^(-mft(1) / vgm) - mft(1)) * vgα^(-vgn))^(mft(1) / vgn)
    else
        ψ = mft(0)
    end
    
    return ψ
end

function pressure_head(S; vgn = vgn, vgα = vgα, vgm = vgm, ν = ν, θr = θr, S_s = S_s)
    mft = eltype(S)
    if S < mft(0)
        println("Effective saturation is negative")
        println(S)
    end
    if S < mft(1)
            ψ = matric_potential(S;vgn = vgn, vgα = vgα, vgm = vgm)
    else
        θ = S* (ν-θr) + θr 
        ψ = (θ-ν)/S_s
    end
    
    return ψ
end


function hydraulic_conductivity(S; vgm = vgm, ksat = ksat)
    mft = eltype(S)
    if S < mft(1)
        K = sqrt(S) * (mft(1) - (mft(1) - S^(mft(1) / vgm))^vgm)^mft(2)
    else
        K = mft(1)
    end
        return K*ksat
end
function hydrostatic_profile(z, zmin, porosity, n, α, θr)
    mft = eltype(z)
    m = mft(1 - 1 / n)
    S = mft((mft(1) + (α * (z - zmin))^n)^(-m))
    return mft(S * (porosity-θr)+θr)
end


function compute_richards_rhs!(dθl, θl, p, top::θDirichlet,bottom::FreeDrainage)
    # θ_top = something, K∇h_bottom = K_bottom (∇h = 1). If ∇h = 1, θ_b = θ_f, so K = Kbottom at the face.

    sp = p[2]
    zc = p[1]
    @unpack ν,vgn,vgα,vgm,ksat,θr = sp
    S = effective_saturation.(θl; ν = ν, θr = θr)
    K = hydraulic_conductivity.(S; vgm = vgm, ksat = ksat)
    ψ = matric_potential.(S; vgn = vgn, vgα = vgα, vgm = vgm)
    h = ψ .+ zc
    
    
    θ_top = θl_surf
    S_top = effective_saturation.(θ_top; ν = ν, θr = θr)
    h_top = matric_potential(S_top; vgn = vgn, vgα = vgα, vgm = vgm)#ztop = 0
    K_top = hydraulic_conductivity(S_top; vgm = vgm, ksat = ksat)
    bc_ktop = Operators.SetValue(K_top)
    If = Operators.InterpolateC2F(;
                                  top = bc_ktop)
    
    bc_t = Operators.SetValue(h_top)
    gradc2f = Operators.GradientC2F(top = bc_t) # set value on h_top
    
    bc_b = Operators.SetValue(FT(1)*parent(K)[1])
    gradf2c = Operators.GradientF2C(bottom = bc_b) # set value on K∇h at bottom
    
    
    
    return @. dθl = gradf2c( If(K) * gradc2f(h))
end


function compute_richards_rhs!(dθl, θl, p, top::RunoffBC, bottom::NormalFlux)
    # Dirichet at top or flux BC at top, depending on rain and ic.
    # No flux at bottom
    sp = p[2]
    zc = p[1]
    @unpack ν,vgn,vgα,vgm,ksat,θr = sp

    S = effective_saturation.(θl; ν = ν, θr = θr)
    K = hydraulic_conductivity.(S; vgm = vgm, ksat = ksat)
    ψ = pressure_head.(S; vgn = vgn, vgα = vgα, vgm = vgm, ν = ν, θr = θr, S_s = S_s)
    h = ψ .+ zc
    # compute ic
    θ_top  = ν
    S_top = effective_saturation.(θ_top; ν = ν, θr = θr)
    ψ_top = pressure_head(S_top; vgn = vgn, vgα = vgα, vgm = vgm, ν = ν, θr = θr, S_s = S_s)
    h_top = ψ_top #(z_top = 0)
    bc_t = Operators.SetValue(h_top) # gradient is determined from this and h_interior
    bc_b = Operators.SetGradient(FT(0)) # This can be anything b.c we just care about top value.
    
    
    # ponding BC
    gradc2f = Operators.GradientC2F(bottom = bc_b, top = bc_t)
    grad_h_ponding = gradc2f(h)
    K_top = ksat
    ic = parent(grad_h_ponding)[end]*K_top
    
    
    if rain_rate < ic # ponding:
        gradc2f = Operators.GradientC2F(top = bc_t)
        If = Operators.InterpolateC2F(top = Operators.SetValue(K_top))
        flux_bc_b = Operators.SetValue(FT(0))
        gradf2c = Operators.GradientF2C(bottom = flux_bc_b) # no flux at bottom
        return @. dθl = gradf2c(If(K)*gradc2f(h))
    else
        #flux_top = rain_rate
        If_rain = Operators.InterpolateC2F()
        top_flux = -rain_rate
        bottom_flux = FT(0)
        gradc2f_rain = Operators.GradientC2F()
        gradf2c = Operators.GradientF2C(top = Operators.SetValue(top_flux), bottom = Operators.SetValue(bottom_flux))
        return @. dθl = gradf2c(If_rain(K) * gradc2f_rain(h))
        
    end

end



@testset "Richards sand 1" begin
    #small difference in Bonan solution and ours, at same resolution.
    n = 150
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
    Δt = FT(0.05)
    tf = FT(60 * 60 * 0.8)
    t0 = FT(0)

    msp = SoilParams{FT}(ν,vgn,vgα,vgm, ksat, θr, FT(1e-4))
    bottom_bc = FreeDrainage()
    top_bc = θDirichlet(θl_surf)
    domain = Domains.IntervalDomain(z₀, z₁, x3boundary = (:bottom, :top))
    mesh = Meshes.IntervalMesh(domain, nelems = n)
    
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = Fields.coordinate_field(cs)
    
    
    θl = Fields.zeros(FT, cs) .+ θl_0
    
    # Solve Richard's Equation: ∂_t θl = ∂_z [K(θl)(∂_z ψ(θl) +1)]
    
    p = [zc, msp,top_bc, bottom_bc]
    function ∑tendencies!(dθl, θl, p, t)
        top = p[3]
        bot = p[4]
        compute_richards_rhs!(dθl, θl,p , top,bot)
    end
#@show ∑tendencies!(similar(θl), θl, nothing, 0.0);
    
    # Solve the ODE operator
    
    prob = ODEProblem(∑tendencies!, θl, (t0, tf),p)
    sol = solve(
        prob,
        Tsit5(),
        dt = Δt,
        saveat = 600 * Δt,
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
    println("mse")
    println(sqrt.(sum((bonan_moisture .- parent(sol.u[end])).^2.0)))
end


@testset "runoff bc" begin
    # not sure yet if this test is passing.
    n = 100
    z₀ = FT(-0.35)
    z₁ = FT(0)
    ksat = FT(0.0443 / (3600 * 100))
    vgn = FT(2)
    vgα = FT(2.6)
    vgm = FT(1)- FT(1)/vgn
    θr = FT(0.0)
    S_s = FT(1e-4)
    ν = FT(0.495)
    Δt = FT(0.01)
    tf = FT(60*60)
    t0 = FT(0)
    msp = SoilParams{FT}(ν,vgn,vgα,vgm, ksat, θr, FT(S_s))
    
    domain = Domains.IntervalDomain(z₀, z₁, x3boundary = (:bottom, :top))
    mesh = Meshes.IntervalMesh(domain, nelems = n)
    
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = Fields.coordinate_field(cs)
    
    θl = hydrostatic_profile.(zc, FT(-0.35), ν, vgn, vgα, θr) 
    
    rain_rate = FT(-5e-4)

    top_bc = RunoffBC{FT}(rain_rate)
    bottom_bc = NormalFlux{FT}(FT(0.0))
    p = [zc, msp, top_bc, bottom_bc]
    function ∑tendencies!(dθl, θl, p, t)
        top = p[3]
        bot = p[4]
        compute_richards_rhs!(dθl, θl,p , top,bot)
    end
    prob = ODEProblem(∑tendencies!, θl, (t0, tf),p)
    sol = solve(
        prob,
        CarpenterKennedy2N54(),
        dt = Δt,
    );


    climatemachine_data = readdlm("../runoff_bc_output_climatemachine.csv", '\t')
    z_cm =climatemachine_data[1,:]
    t_cm =climatemachine_data[2,:]
        
    dirname = "richards"
    path = joinpath(@__DIR__, "output", dirname)
    mkpath(path)
    ENV["GKSwstype"] = "nul"
    import Plots
    Plots.GRBackend()
    thinned_sol = [sol.u[k] for k in 1:600:length(sol.u)]
    anim = @animate for (nt, u) in enumerate(thinned_sol)
      #  if nt % 500 == 0
        Plots.plot(
            parent(u),parent(zc),
            xlim = (0.3, 0.5),
            ylim = (-0.35, 0),
            title = string(string(Int(round(((nt-1)*Δt-t0)/60)))," min"),
            lc = :black,
            lw = 2,
            ls = :dash,
            label = "",
            legend = :outerright,
            m = :o,
            xlabel = "θ(z)",
            ylabel = "z",
        )
        Plots.plot!(t_cm, z_cm, label = "climate machine")
    #end
    end
    
    Plots.gif(anim, joinpath(path, "richards_bc.gif"), fps = 10)
end


end

