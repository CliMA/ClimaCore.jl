using LinearAlgebra

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies, DataLayouts
import ClimaCore.Geometry: ⊗

using ClimaCore.DataLayouts: IJFH
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using SpecialFunctions
using StaticArrays

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())


const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
    A = 1.90695, # Spectral integration constant (4.5c Braun et al. (2018))
    k₁ = 2/3,
    k₂ = -5/3 
)

domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(-2π),
        Geometry.XPoint(2π),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(-2π),
        Geometry.YPoint(2π),
        periodic = true,
    ),
)

n1, n2 = 8, 8
Nq = 4
Nqh = 7
Δx = 4π / n1 / Nq
mesh = Meshes.RectilinearMesh(domain, n1, n2)
grid_topology = Topologies.Topology2D(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

Iquad = Spaces.Quadratures.GLL{Nqh}()
Ispace = Spaces.SpectralElementSpace2D(grid_topology, Iquad)

function init_state(coord, p)
    x, y = coord.x, coord.y
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(y)^(-2)

    # Ψ′ = exp(-(y + p.l / 10)^2 / 2p.l^2) * cos(p.k * x) * cos(p.k * y)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(y + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)

    u = Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * y)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, param)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    return (
        ρ = ρu,
        ρu = ((ρu ⊗ u) + (param.g * ρ^2 / 2) * LinearAlgebra.I),
        ρθ = ρθ * u,
    )
end

function energy(state, param)
    ρ, ρu = state.ρ, state.ρu
    u = ρu / ρ
    return ρ * (u.u^2 + u.v^2) / 2 + param.g * ρ^2 / 2
end

function total_energy(y, parameters)
    sum(energy.(y, Ref(parameters)))
end

"""
  structure_function(A::Field)
Computes (δuᵢ)ⱼ = (uᵢ(x₀ ± 𝑒ⱼΔ) - uᵢ(x₀))²
which is (δuᵢ)ⱼ = (Aⱼ - Bᵢ)². This is the second
order structure function. This can be generalised to
higher order structure functions. Function halo is the 
nearest bounding "square/cube" in 2D/3D. 
For "target" point O, the halo is thus given by 
x--x--x
x--O--x
x--x--x. 
Future versions will allow the specification of arbitrary 
windowed moving averages. (Typically the filter width guides
the window size in the SV-SGS model). 

References:
(1) doi:10.1017/S0022112009006867 
(2) doi:10.1017/jfm.2018.766
"""
function structure_function(χ::Fields.Field)
    space = axes(χ)
    FT = Spaces.undertype(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    χ_data = Fields.field_values(zeros(space))
    horz_x_elems = space.topology.elemorder.indices.:1[end]
    horz_y_elems = space.topology.elemorder.indices.:2[end]
    CartInd = space.topology.elemorder
    out = similar(χ)

    # Loop over horizontal elements
    for hx in 1:horz_x_elems
      for hy in 1:horz_y_elems
        # Get global index
        nh = hx*hy
        χ_slab =  parent(Spaces.slab(χ,nh))
        # Get nodal Cartesian indices
        R = CartesianIndices(χ_slab)
        Ifirst, Ilast = first(R), last(R)
        I1 = oneunit(Ifirst)
        # Moving/windowed filter
        for I in R
          n, Σ = 0, zero(eltype(out))
          for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            if I != J
              Σ += (χ_slab[I] - χ_slab[J])^2
              n += 1
            end
          end
          parent(out)[I[1],I[2],1,nh] = Σ/n
        end
      end
    end
    return out
end

"""
  structure_function(A::AbstractArray,B::AbstractArray)
Computes (δuᵢ)ⱼ = (uᵢ(x₀ ± 𝑒ⱼΔ) - uᵢ(x₀))²
which is (δuᵢ)ⱼ = (Aⱼ - Bᵢ)². This is the second
order structure function. This can be generalised to
higher order structure functions. Function halo is the 
nearest bounding "square/cube" in 2D/3D. 
For "target" point O, the halo is thus given by 
x--x--x
x--O--x
x--x--x
"""
function structure_function(A::AbstractArray, B::AbstractArray)
    @assert typeof(A) == typeof(B)
    F₂ = similar(A)
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = oneunit(Ifirst)
    for I in R
      @show I
        n, Σ = 0, zero(eltype(F₂))
        for J in max(Ifirst, I-I1):min(Ilast, I+I1)
          if I != J
            Σ += (A[I] - B[J])^2
            n += 1
          end
        end
        F₂[I] = Σ/n
    end
    F₂
end

"""
  strainrate(∇𝒰)
For a velocity gradient field, ∇𝒰, compute the 
(ClimaCore specific) strain rate tensor, and
return the corresponding field object. 
"""
function strainrate(∇𝒰::Fields.Field)
  space = axes(∇𝒰)
  FT = Spaces.undertype(space)
  Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
  horz_x_elems = space.topology.elemorder.indices.:1[end]
  horz_y_elems = space.topology.elemorder.indices.:2[end]
  
  𝒮 = similar(∇𝒰)

  ∇𝒰_11 = @. ∇𝒰.components.data.:1
  ∇𝒰_12 = @. ∇𝒰.components.data.:2
  ∇𝒰_21 = @. ∇𝒰.components.data.:3
  ∇𝒰_22 = @. ∇𝒰.components.data.:4

  # Symmetric Rate of Strain Tensor Components
  S11 = @. ∇𝒰_11 + ∇𝒰_11
  S12 = @. ∇𝒰_12 + ∇𝒰_21
  S21 = @. ∇𝒰_21 + ∇𝒰_12
  S22 = @. ∇𝒰_22 + ∇𝒰_22

  for hx in 1:horz_x_elems
    for hy in 1:horz_y_elems
      nh = hx*hy
      for i in 1:Nq
        for j in 1:Nq
          parent(𝒮)[i,j,1,nh]=parent(S11)[i,j,1,nh] 
          parent(𝒮)[i,j,2,nh]=parent(S12)[i,j,1,nh] 
          parent(𝒮)[i,j,3,nh]=parent(S21)[i,j,1,nh] 
          parent(𝒮)[i,j,4,nh]=parent(S22)[i,j,1,nh] 
        end
      end
    end
  end
  return 𝒮
end

"""
  compute_ℯᵥ(X::Field)
Compute the most extensional eigenvector for each grid point,
with the assumption that the turbulence is captured by stretched 
vortex ensembles within each subgrid-scale grouping
"""
function compute_ℯᵥ(X::Fields.Field)
  space = axes(X)
  FT = Spaces.undertype(space)
  Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
  horz_x_elems = space.topology.elemorder.indices.:1[end]
  horz_y_elems = space.topology.elemorder.indices.:2[end]
  E = Fields.Field(DataLayouts.IJFH{eltype(X), Nq}(ones(Nq, Nq, 2, horz_x_elems*horz_y_elems)), space)
  PX = parent(X)
  for hx in 1:horz_x_elems
    for hy in 1:horz_y_elems
      nh = hx*hy
      for i in 1:Nq
        for j in 1:Nq
          𝒮 = @MMatrix [PX[i,j,1,nh] PX[i,j,2,nh]; PX[i,j,3,nh] PX[i,j,4,nh]] 
          𝒮[isnan.(𝒮)] .= FT(0)
          𝒮[isinf.(𝒮)] .= FT(0)
          𝒱 = eigen(𝒮).vectors
          ℯᵥ = 𝒱[:,2]
          ℯᵥ¹ = ℯᵥ[1]
          ℯᵥ² = ℯᵥ[2]
          parent(E)[i,j,1,nh] = ℯᵥ¹
          parent(E)[i,j,2,nh] = ℯᵥ² 
        end
      end
    end
  end
  return E
end

"""
  compute_τ
Given the turbulent, subgrid energy, and the orientation of the 
most extensional eigenvector for an ensemble of stretched vortices, 
compute the modeled turbulent stress tensor.
"""
function compute_τ(K::Fields.Field, ℯᵥ::Fields.Field, ∇𝒰)
  space = axes(ℯᵥ)
  FT = Spaces.undertype(space)
  Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
  horz_x_elems = space.topology.elemorder.indices.:1[end]
  horz_y_elems = space.topology.elemorder.indices.:2[end]
  #τ = Fields.Field(DataLayouts.IJFH{eltype(K), Nq}(ones(Nq, Nq, 4, horz_x_elems*horz_y_elems)), space)
  τ = similar(∇𝒰)
  PE = parent(ℯᵥ)
  PK = parent(K)
  for hx in 1:horz_x_elems
    for hy in 1:horz_y_elems
      nh = hx*hy
      for i in 1:Nq
        for j in 1:Nq
          T1 = PK[i,j,1,nh] * (FT(1) - PE[i,j,1,nh]^2)
          T2 = PK[i,j,1,nh] * (FT(1) - PE[i,j,2,nh]^2)
          parent(τ)[i,j,1,nh] = T1
          parent(τ)[i,j,2,nh] = FT(0)
          parent(τ)[i,j,3,nh] = FT(0)
          parent(τ)[i,j,4,nh] = T2
        end
      end
    end
  end
  return τ
end


"""
  rhs!()
Tendency assembly function
"""
function rhs!(dydt, y, _, t)
    # Define Operators
    I = Operators.Interpolate(Ispace)
    div = Operators.WeakDivergence()
    grad = Operators.Gradient()
    R = Operators.Restrict(space)
    # Unpack Parameters
    rparameters = Ref(parameters)
    # Euler Equation Tendency [No Diffusion]
    @. dydt = -R(div(flux(I(y), rparameters)))
    # SV SGS Calculations
    𝒰 = @. y.ρu / y.ρ
    ∇𝒰 = @. grad(𝒰)
    # Assemble 𝒮 = 1/2(uᵢ,ⱼ + uⱼ,ᵢ)
    𝒮 = strainrate(∇𝒰)
    # Compute Most Extensional Eigenvector
    E = compute_ℯᵥ(𝒮)
    ℯᵥ¹ = @. E.components.data.:1
    ℯᵥ² = @. E.components.data.:2
    𝒮₁₁ = @. 𝒮.components.data.:1
    𝒮₁₂ = @. 𝒮.components.data.:2
    𝒮₂₁ = @. 𝒮.components.data.:3
    𝒮₂₂ = @. 𝒮.components.data.:4
    ã₁ = @. ℯᵥ¹*ℯᵥ¹*𝒮₁₁ 
    ã₂ = @. ℯᵥ¹*ℯᵥ²*𝒮₁₂
    ã₃ = @. ℯᵥ²*ℯᵥ¹*𝒮₂₁
    ã₄ = @. ℯᵥ²*ℯᵥ²*𝒮₂₂
    ã = @. abs(ã₁ + ã₂ + ã₃ + ã₄)
    # Compute Subgrid Tendency Based on Vortex Model
    ℱ = structure_function(y.ρ) # 4.5b
    k₁ = parameters.k₁
    K₀ε = @. ℱ / parameters.A / Δx^(k₁) # (4.5a)
    ν = 1.0
    kc = π / Δx
    Γ = @. gamma(-k₁/2, 2*ν*kc^2/3/abs(ã))
    Kₑ = @. 1/2 * y.ρ * K₀ε * (2*ν/3/ã)^(k₁/2) * Γ # (4.4)
    # Get SGS Flux
    τ = compute_τ(Kₑ, E, ∇𝒰)
    flux_sgs = @. y.ρ * τ
    @show summary(flux_sgs)
    @show flux_sgs
    @. dydt.ρu += R(div(I(flux_sgs)))
    # Tendency DSS Application
    Spaces.weighted_dss!(dydt)
    return dydt
end

# Next steps:
# 1. add the above to the design docs (divergence + over-integration + DSS)
# 2. add boundary conditions

dydt = similar(y0)
rhs!(dydt, y0, nothing, 0.0)


# Solve the ODE operator
prob = ODEProblem(rhs!, y0, (0.0, 10.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.01,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "cg"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
