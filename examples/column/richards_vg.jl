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

const FT = Float64



@testset "Richards sand 1" begin
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
        # h_top = something, ∇h_bottom = something (grad and state values set on same thing)
        θ_top = θl_surf
        S_top = effective_saturation.(θ_top; ν = ν, θr = θr)
        h_top = matric_potential(S_top; vgn = vgn, vgα = vgα, vgm = vgm)#ztop = 0
        bc_t = Operators.SetValue(h_top) # gradient is determined from this and h_interior
        bc_b = Operators.SetGradient(FT(1)) # forces ∇h to be this by setting h_face appropriately
        
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
        # At the bottom, we have a BC on grad(h). I thought this meant that h_face at the bottom was forced to be something
        # so we should know theta_bottom_face -> K_bottom_face should be known.
        # probably shouldnt be an extrapolation
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
        Tsit5(),
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
end


@testset "runoff bc" begin
    n = 50
    z₀ = FT(-0.35)
    z₁ = FT(0)
    ksat = FT(0.0443 / (3600 * 100))
    vgn = FT(2)
    vgα = FT(2.6)
    vgm = FT(1)- FT(1)/vgn
    θr = FT(0.0)
    S_s = FT(1e-4)
    ν = FT(0.495)
    Δt = FT(0.1)
    tf = FT(60*60)
    t0 = FT(0)
    
    
    domain = Domains.IntervalDomain(z₀, z₁, x3boundary = (:bottom, :top))
    mesh = Meshes.IntervalMesh(domain, nelems = n)
    
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = Fields.coordinate_field(cs)
    
    function effective_saturation(θ; ν = ν, θr = θr)
        
        if θ < θr
            println("Effective saturation is negative")
            println(θ)
        end
        return (θ-θr)/(ν-θr)
    end
    
    function pressure_head(S; vgn = vgn, vgα = vgα, vgm = vgm, ν = ν, θr = θr, S_s = S_s)
        if S < FT(0)
            println("Effective saturation is negative")
            println(S)
        end
        if S < FT(1)
            ψ = -((S^(-FT(1) / vgm) - FT(1)) * vgα^(-vgn))^(FT(1) / vgn)
        else
            θ = S* (ν-θr) + θr 
            ψ = (θ-ν)/S_s
        end
        
        return ψ
    end


    function hydraulic_conductivity(S; vgm = vgm, ksat = ksat)
        if S < FT(1)
            K = sqrt(S) * (FT(1) - (FT(1) - S^(FT(1) / vgm))^vgm)^FT(2)
        else
            K = FT(1)
        end
        return K*ksat
    end
    function hydrostatic_profile(z, zmin, porosity, n, α, θr)
        myf = eltype(z)
        m = FT(1 - 1 / n)
        S = FT((FT(1) + (α * (z - zmin))^n)^(-m))
        return FT(S * (porosity-θr)+θr)
    end
    θl = hydrostatic_profile.(zc, FT(-0.35), ν, vgn, vgα, θr) 
    
    # Solve Richard's Equatio2n: ∂_t θl = ∂_z [K(θl)(∂_z ψ(θl) +1)]
    rain_rate = FT(-5e-4)
    function ∑tendencies!(dθl, θl, z, t)
        # compute ic
        θ_top  = ν
        S_top = effective_saturation.(θ_top; ν = ν, θr = θr)
        ψ_top = pressure_head(S_top; vgn = vgn, vgα = vgα, vgm = vgm, ν = ν, θr = θr, S_s = S_s)
        h_top = ψ_top #(z_top = 0)
        bc_t = Operators.SetValue(h_top) # gradient is determined from this and h_interior
        bc_b = Operators.SetGradient(FT(0)) # This is the same as flux_bottom = 0
        

        # ponding BC
        gradc2f = Operators.GradientC2F(bottom = bc_b, top = bc_t)
        
        S = effective_saturation.(θl; ν = ν, θr = θr)
        K = hydraulic_conductivity.(S; vgm = vgm, ksat = ksat)
        ψ = pressure_head.(S; vgn = vgn, vgα = vgα, vgm = vgm, ν = ν, θr = θr, S_s = S_s)
        h = ψ .+ zc


        grad_h_ponding = gradc2f(h)
        K_top = ksat
        bc_ktop = Operators.SetValue(K_top)
        If = Operators.InterpolateC2F(;
                                      bottom = Operators.Extrapolate(),
                                      top = bc_ktop)
        top_flux_ponding = -FT(1) .* If(K) .* grad_h_ponding

        

        i_c = parent(top_flux_ponding)[end]
        if rain_rate < i_c # ponding:
            gradf2c = Operators.GradientF2C()
            return @. dθl = -gradf2c(top_flux_ponding)
        else
            #flux_top = rain_rate
            If_rain = Operators.InterpolateC2F(;
                                               bottom = Operators.Extrapolate(),
                                               top = Operators.Extrapolate())
            Kf = If_rain(K)
            ∇h_bottom = FT(0)
            ∇h_top = rain_rate ./ (-Kf)
            gradc2f_rain = Operators.GradientC2F(top = Operators.SetValue(∇h_top), bottom = Operators.SetValue(∇h_bottom))
            ∇hf = gradc2f_rain(h)
            return @. dθl = -gradf2c(-Kf * ∇hf)

        end
    end
#@show ∑tendencies!(similar(θl), θl, nothing, 0.0);
    
    # Solve the ODE operator
    # saved_values = SavedValues(Float64, typeof(θl))
#    cb = SavingCallback((u,t,integrator)->(θl), saved_values, saveat=t0:500*Δt:tf)
    prob = ODEProblem(∑tendencies!, θl, (t0, tf))
    sol = solve(
        prob,
        CarpenterKennedy2N54(),
        dt = Δt,
        #saveat = 500 * Δt,
#        callback = cb also doesnt work
        #progress = true,
        #progress_message = (dt, u, p, t) -> t,
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

