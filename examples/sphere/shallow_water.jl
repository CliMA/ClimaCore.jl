using LinearAlgebra

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies, DataLayouts

using StaticArrays
using SpecialFunctions
using ClimaCore.DataLayouts: IJFH

import QuadGK
import OrdinaryDiffEq
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

# This example solves the shallow-water equations on a cubed-sphere manifold.
# This file contains five test cases:
# - One, called "steady_state", reproduces Test Case 2 in Williamson et al,
#   "A standard test set for numerical approximations to the shallow water
#   equations in spherical geometry", 1992. This test case gives the steady-state
#   solution to the non-linear shallow water equations. It consists of solid
#   body rotation or zonal flow with the corresponding geostrophic height field.
#   This can be run with an angle α that represents the angle between the north
#   pole and the center of the top cube panel.
# - A second one, called "steady_state_compact", reproduces Test Case 3 in the same
#   reference paper. This test case gives the steady-state solution to the
#   non-linear shallow water equations with nonlinear zonal geostrophic flow
#   with compact support.
# - A third one, called "mountain", reproduces Test Case 5 in the same
#   reference paper. It represents a zonal flow over an isolated mountain,
#   where the governing equations describe a global steady-state nonlinear
#   zonal geostrophic flow, with a corresponding geostrophic height field over
#   a non-uniform reference surface h_s.
# - A fourth one, called "rossby_haurwitz", reproduces Test Case 6 in the same
#   reference paper. It represents the solution of the nonlinear barotropic
#   vorticity equation on the sphere
# - A fifth one, called "barotropic_instability", reproduces the test case in
#   Galewsky et al, "An initial-value problem for testing numerical models of
#   the global shallow-water equations", 2004 (also in Sec. 7.6 of Ullirch et al,
#   "High-order ﬁnite-volume methods for the shallow-water equations on
#   the sphere", 2010). This test case consists of a zonal jet with compact
#   support at a latitude of 45°. A small height disturbance is then added,
#   which causes the jet to become unstable and collapse into a highly vortical
#   structure.

# Physical parameters needed
const R = 6.37122e6
const Ω = 7.292e-5
const g = 9.80616
const D₄ = 0.0e16 # hyperdiffusion coefficient
const Aₖ = 1.90695 # Spectral integration constant (4.5c Braun et al. (2018))
const k₁ = 1/3
const k₂ = -5/3
const ν = 1e-4 # Viscosity
const Δx = 123554.0

# Test case specifications
const test_name = get(ARGS, 1, "barotropic_instability") # default test case to run
const test_angle_name = get(ARGS, 2, "alpha0") # default test case to run
const steady_state_test_name = "steady_state"
const steady_state_compact_test_name = "steady_state_compact"
const mountain_test_name = "mountain"
const rossby_haurwitz_test_name = "rossby_haurwitz"
const barotropic_instability_test_name = "barotropic_instability"
const alpha0_test_name = "alpha0"
const alpha30_test_name = "alpha30"
const alpha45_test_name = "alpha45"
const alpha60_test_name = "alpha60"

# Test-specific physical parameters
if test_angle_name == alpha30_test_name
    const α = 30.0
elseif test_angle_name == alpha45_test_name
    const α = 45.0
elseif test_angle_name == alpha60_test_name
    const α = 60.0
else # default test case, α = 0.0
    const α = 0.0
end

if test_name == mountain_test_name
    const u0 = 20.0
    const h0 = 5960
    const a = 20.0 # radius of conical mountain
    const λc = 90.0 # center of mountain long coord, shifted by 180 compared to the paper, because our λ ∈ [-180, 180] (in the paper it was 270, with λ ∈ [0, 360])
    const ϕc = 30.0 # center of mountain lat coord
    const h_s0 = 2e3
elseif test_name == rossby_haurwitz_test_name
    const a = 4.0
    const h0 = 8.0e3
    const ω = 7.848e-6
    const K = 7.848e-6
elseif test_name == steady_state_compact_test_name
    const u0 = 2 * pi * R / (12 * 86400)
    const h0 = 2.94e4 / g
    const ϕᵦ = -30.0
    const ϕₑ = 90.0
    const xₑ = 0.3
elseif test_name == barotropic_instability_test_name
    const u_max = 80.0
    const xₑ = 0.3
    const αₚ = 19.09859
    const βₚ = 3.81971
    const h0 = 10158.18617 # value for initial height from Tempest https://github.com/paullric/tempestmodel/blob/master/test/shallowwater_sphere/BarotropicInstabilityTest.cpp#L86
    const h_hat = 120.0
    const ϕ₀ = 25.71428
    const ϕ₁ = 64.28571
    const ϕ₂ = 45.0
    const eₙ = exp(-4.0 / (deg2rad(ϕ₁) - deg2rad(ϕ₀))^2)
else # default case, steady-state test case
    const u0 = 2 * pi * R / (12 * 86400)
    const h0 = 2.94e4 / g
end

# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dir = "cg_sphere_shallowwater_$(test_name)"
dir = "$(dir)_$(test_angle_name)"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

function linkfig(figpath, alt = "")
    # Buildkite-agent upload figpath
    # Link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

# Set up discretization
ne = 9 # the rossby_haurwitz test case's initial state has a singularity at the pole. We avoid it by using odd number of elements
Nq = 4

domain = Domains.SphereDomain(R)
mesh = Meshes.EquiangularCubedSphere(domain, ne)
grid_topology = Topologies.Topology2D(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

coords = Fields.coordinate_field(space)

# Definition of Coriolis parameter
if test_name == rossby_haurwitz_test_name
    f = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long

        f = 2 * Ω * sind(ϕ)

        # Technically this should be a WVector, but since we are only in a 2D space,
        # WVector, Contravariant3Vector, Covariant3Vector are all equivalent.
        # This _won't_ be true in 3D however!
        Geometry.Contravariant3Vector(f)
    end
else # all other test cases share the same Coriolis parameter
    f = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates
        ϕ = coord.lat
        λ = coord.long

        f = 2 * Ω * (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))

        # Technically this should be a WVector, but since we are only in a 2D space,
        # WVector, Contravariant3Vector, Covariant3Vector are all equivalent.
        # This _won't_ be true in 3D however!
        Geometry.Contravariant3Vector(f)
    end
end

# Definition of bottom surface topography field
if test_name == mountain_test_name # define the non-uniform reference surface h_s
    h_s = map(Fields.coordinate_field(space)) do coord
        ϕ = coord.lat
        λ = coord.long
        r = sqrt(min(a^2, (λ - λc)^2 + (ϕ - ϕc)^2)) # positive branch
        h_s = h_s0 * (1 - r / a)
    end
else
    h_s = zeros(space)
end

# Set initial condition
if test_name == rossby_haurwitz_test_name
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates
        ϕ = coord.lat
        λ = coord.long

        A =
            ω / 2 * (2 * Ω + ω) * cosd(ϕ)^2 +
            1 / 4 *
            K^2 *
            cosd(ϕ)^(2 * a) *
            ((a + 1) * cosd(ϕ)^2 + (2 * a^2 - a - 2) - 2 * a^2 * cosd(ϕ)^-2)
        B =
            2 * (Ω + ω) * K / (a + 1) / (a + 2) *
            cosd(ϕ)^a *
            ((a^2 + 2 * a + 2) - (a + 1)^2 * cosd(ϕ)^2)
        C = 1 / 4 * K^2 * cosd(ϕ)^(2 * a) * ((a + 1) * cosd(ϕ)^2 - (a + 2))

        h =
            h0 +
            (R^2 * A + R^2 * B * cosd(a * λ) + R^2 * C * cosd(2 * a * λ)) / g

        uλ =
            R * ω * cosd(ϕ) +
            R * K * cosd(ϕ)^(a - 1) * (a * sind(ϕ)^2 - cosd(ϕ)^2) * cosd(a * λ)
        uϕ = -R * K * a * cosd(ϕ)^(a - 1) * sind(ϕ) * sind(a * λ)


        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )
        return (h = h, u = u)
    end
elseif test_name == steady_state_compact_test_name
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long

        if α == 0.0
            ϕprime = ϕ
            λprime = λ
        else
            ϕprime = asind(sind(ϕ) * cosd(α) - cosd(ϕ) * cosd(λ) * sind(α))
            λprime = asind(sind(λ) * cosd(ϕ) / cosd(ϕprime)) # for alpha45, this experiences numerical precision issues. The test case is designed for either alpha0 or alpha60

            # Temporary angle to ensure λprime is in the right quadrant
            λcond = cosd(α) * cosd(λ) * cosd(ϕ) + sind(α) * sind(ϕ)

            # If λprime is not in the right quadrant, adjust
            if λcond < 0.0
                λprime = -λprime - 180.0 # shifted by 180 compared to the paper, because our λ ∈ [-180, 180]
            end
            if λprime < -180.0
                λprime += 360.0
            end
        end

        # Set auxiliary function needed for initial state of velocity field
        b(x) = x ≤ 0.0 ? 0.0 : exp(-x^(-1))

        x(ϕprime) = xₑ * (ϕprime - ϕᵦ) / (ϕₑ - ϕᵦ)
        uλprime(ϕprime) =
            u0 * b(x(ϕprime)) * b(xₑ - x(ϕprime)) * exp(4.0 / xₑ)
        uϕprime = 0.0

        # Set integral needed for height initial state
        h_int(γ) =
            abs(γ) < 90.0 ?
            (2 * Ω * sind(γ) + uλprime(γ) * tand(γ) / R) * uλprime(γ) : 0.0

        # Set initial state for height field
        h =
            h0 - (R / g) * (pi / 180.0) * QuadGK.quadgk(h_int, -90.0, ϕprime)[1]

        # Set initial state for velocity field
        uϕ = -(uλprime(ϕprime) * sind(α) * sind(λprime)) / cosd(ϕ)
        if abs(cosd(λ)) < 1e-13
            if abs(α) > 1e-13
                if cosd(λ) > 0.0
                    uλ = -uϕ * cosd(ϕ) / tand(α)
                else
                    uλ = uϕ * cosd(ϕ) / tand(α)
                end
            else
                uλ = uλprime(ϕprime)
            end
        else
            uλ =
                (uϕ * sind(ϕ) * sind(λ) + uλprime(ϕprime) * cosd(λprime)) /
                cosd(λ)
        end

        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )

        return (h = h, u = u)
    end
elseif test_name == barotropic_instability_test_name
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long

        if α == 0.0
            ϕprime = ϕ
        else
            ϕprime = asind(sind(ϕ) * cosd(α) - cosd(ϕ) * cosd(λ) * sind(α))
        end

        # Set initial state of velocity field
        uλprime(ϕprime) =
            (u_max / eₙ) *
            exp(1.0 / (deg2rad(ϕprime - ϕ₀) * deg2rad(ϕprime - ϕ₁))) *
            (ϕ₀ < ϕprime < ϕ₁)
        uϕprime = 0.0

        # Set integral needed for height initial state
        h_int(γ) =
            abs(γ) < 90.0 ?
            (2 * Ω * sind(γ) + uλprime(γ) * tand(γ) / R) * uλprime(γ) : 0.0

        # Set initial state for height field
        h =
            h0 - (R / g) * (pi / 180.0) * QuadGK.quadgk(h_int, -90.0, ϕprime)[1]

        if λ > 0.0
            λ -= 360.0
        end
        if λ < -360.0 || λ > 0.0
            @info "Invalid longitude value"
        end

        # Add height perturbation
        h += h_hat * cosd(ϕ) * exp(-(λ^2 / αₚ^2) - ((ϕ₂ - ϕ)^2 / βₚ^2))

        uλ = uλprime(ϕprime)
        uϕ = uϕprime

        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )

        return (h = h, u = u)
    end
else # steady-state and mountain test cases share the same form of fields
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long
        h =
            h0 -
            (R * Ω * u0 + u0^2 / 2) / g *
            (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))^2
        uλ = u0 * (cosd(α) * cosd(ϕ) + sind(α) * cosd(λ) * sind(ϕ))
        uϕ = -u0 * sind(α) * sind(λ)

        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )

        return (h = h, u = u)
    end
end


function kolmogorov_prefactor(F₂)
    return @. F₂ / Aₖ / Δx^(k₁) # (4.5a)
end
function structure_function(χ::Fields.Field; p=2)
    space = axes(χ)
    FT = Spaces.undertype(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    horz_x_elems = space.topology.elemorder.indices.:1[end]
    horz_y_elems = space.topology.elemorder.indices.:2[end]
    ne = horz_x_elems * horz_y_elems
    CartInd = space.topology.elemorder
    out = similar(χ)

    # Loop over horizontal elements
    for hx in 1:horz_x_elems
      for hy in 1:horz_y_elems
        # Get global index
        nh = horz_x_elems*(hx-1) + hy
        #χ_slab =  parent(Spaces.slab(χ,nh))
        χ_slab =  parent(χ)[:,:,1,nh]
        # Get nodal Cartesian indices
        R = CartesianIndices(χ_slab)
        Ifirst, Ilast = first(R), last(R)
        I1 = oneunit(Ifirst)
        # Moving/windowed filter
        for I in R
          n, Σ = 0, zero(eltype(out))
          for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            if I != J
              Σ += (χ_slab[I] - χ_slab[J])^p
              n += 1
            end
          end
          parent(out)[I[1],I[2],1,nh] = Σ/n
        end
      end
    end
    return out
end
function strainrate(∇𝒰::Fields.Field)
  space = axes(∇𝒰)
  FT = Spaces.undertype(space)
  Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
  horz_x_elems = space.topology.elemorder.indices.:1[end]
  horz_y_elems = space.topology.elemorder.indices.:2[end]
  
  𝒮 = zero(∇𝒰)

  ∇𝒰_11 = @. ∇𝒰.components.data.:1
  ∇𝒰_12 = @. ∇𝒰.components.data.:2
  ∇𝒰_21 = @. ∇𝒰.components.data.:3
  ∇𝒰_22 = @. ∇𝒰.components.data.:4

  # Symmetric Rate of Strain Tensor Components
  S11 = @. ∇𝒰_11
  S12 = @. 1/2*(∇𝒰_12 + ∇𝒰_21) 
  S21 = @. 1/2*(∇𝒰_21 + ∇𝒰_12)
  S22 = @. ∇𝒰_22

  nh = horz_x_elems * horz_y_elems
  for he in 1:nh
    for i in 1:Nq
      for j in 1:Nq
        parent(𝒮)[i,j,1,he]=parent(S11)[i,j,1,he] 
        parent(𝒮)[i,j,2,he]=parent(S12)[i,j,1,he] 
        parent(𝒮)[i,j,3,he]=parent(S21)[i,j,1,he] 
        parent(𝒮)[i,j,4,he]=parent(S22)[i,j,1,he] 
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
  nfaces = 6
  E = Fields.Field(DataLayouts.IJFH{eltype(X), Nq}(ones(Nq, Nq, 2, horz_x_elems*horz_y_elems*nfaces)), space)
  PX = parent(X)
  nh = horz_x_elems * horz_y_elems * nfaces
  for he in 1:nh
    for i in 1:Nq
      for j in 1:Nq
        𝒮 = @SMatrix [PX[i,j,1,he] PX[i,j,2,he]; PX[i,j,3,he] PX[i,j,4,he]] 
        𝒱 = eigen(𝒮).vectors # Want the most extensional eigenvector, Julia sorts λ by default.
        ℯᵥ = 𝒱[:,2]
        ℯᵥ¹ = ℯᵥ[1]
        ℯᵥ² = ℯᵥ[2]
        parent(E)[i,j,1,he] = ℯᵥ¹
        parent(E)[i,j,2,he] = ℯᵥ² 
      end
    end
  end
  return E
end

"""
  compute_subgrid_stress
Given the turbulent, subgrid energy, and the orientation of the 
most extensional eigenvector for an ensemble of stretched vortices, 
compute the modeled turbulent stress tensor.
"""
function compute_subgrid_stress(K::Fields.Field, ℯᵥ::Fields.Field, ∇𝒰)
  space = axes(ℯᵥ)
  FT = Spaces.undertype(space)
  Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
  horz_x_elems = space.topology.elemorder.indices.:1[end]
  horz_y_elems = space.topology.elemorder.indices.:2[end]
  τ = similar(∇𝒰)
  PE = parent(ℯᵥ)
  PK = parent(K)
  nh = horz_x_elems * horz_y_elems * 6
  for he in 1:nh
    for i in 1:Nq
      for j in 1:Nq
        # Diagonal Terms
        T1 = PK[i,j,1,he] * (FT(1) - PE[i,j,1,he]^2)
        T2 = PK[i,j,1,he] * (FT(1) - PE[i,j,2,he]^2)
        # Off diagonal terms (Symmetric Stress Assumption)
        T3 = PK[i,j,1,he] * (FT(0) - PE[i,j,1,he]*PE[i,j,2,he])
        parent(τ)[i,j,1,he] = T1
        parent(τ)[i,j,2,he] = T3
        parent(τ)[i,j,3,he] = T3
        parent(τ)[i,j,4,he] = T2
      end
    end
  end
  return τ
end


function rhs!(dYdt, y, parameters, t)
    f = parameters.f
    h_s = parameters.h_s

    div = Operators.Divergence()
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    wgrad = Operators.WeakGradient()
    curl = Operators.Curl()
    wcurl = Operators.WeakCurl()

    # Compute hyperviscosity first
    @. dYdt.h = wdiv(grad(y.h))
    @. dYdt.u =
        wgrad(div(y.u)) -
        Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(y.u))))

    Spaces.weighted_dss!(dYdt)

    @. dYdt.h = -D₄ * wdiv(grad(dYdt.h))
    @. dYdt.u =
        -D₄ * (
            wgrad(div(dYdt.u)) - Geometry.Covariant12Vector(
                wcurl(Geometry.Covariant3Vector(curl(dYdt.u))),
            )
        )
    
    sgs_isactive = true
    if sgs_isactive
      local_space = axes(y.u)
      local_geometry = Fields.local_geometry_field(local_space)
      𝒰 = @. Geometry.LocalVector(Geometry.Covariant12Vector(y.u))
      ∇𝒰 = @. grad(𝒰)
      # Assemble 𝒮 = 1/2(uᵢ,ⱼ + uⱼ,ᵢ)
      𝒮 = strainrate(∇𝒰)
      norm𝒮 = @. 𝒮.components.data.:1^2 + 2 * 𝒮.components.data.:2^2 + 𝒮.components.data.:4^2
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
      kc = π / Δx
      F₂x = structure_function(𝒰.components.data.:1; p=2) # 4.5b
      F₂y = structure_function(𝒰.components.data.:2; p=2) # 4.5b
      F₂ = @. F₂x + F₂y
      K₀ε = @. kolmogorov_prefactor(F₂)
      Q = @. 2*ν*kc^2/3/(ã + 1e-14)
      Γ = @. gamma(-k₁, Q)
      Kₑ = @. 1/2 * K₀ε * (2*ν/3/(ã + 1e-14))^(k₁) * Γ # (4.4)
      # Get SGS Flux
      τ = compute_subgrid_stress(Kₑ, E, ∇𝒰)
      
      # STRETCHED VORTEX 
      flux_sgs = @. - τ
      flux_sgs1 = @. Geometry.Covariant12Vector(Geometry.UVVector(flux_sgs.components.data.:1, flux_sgs.components.data.:2))
      flux_sgs2 = @. Geometry.Covariant12Vector(Geometry.UVVector(flux_sgs.components.data.:2, flux_sgs.components.data.:4))
     @. dYdt.u.components.data.:1 += wdiv(flux_sgs1)
     @. dYdt.u.components.data.:2 += wdiv(flux_sgs2)
    end

    # Add in pieces
    @. begin
        dYdt.h += -wdiv(y.h * y.u)
        dYdt.u +=
            -grad(g * (y.h + h_s) + norm(y.u)^2 / 2) + y.u × (f + curl(y.u))
    end
    Spaces.weighted_dss!(dYdt)
    return dYdt
end

# Set up RHS function
dYdt = similar(Y)
parameters = (; f = f, h_s = h_s)
rhs!(dYdt, Y, parameters, 0.0)

# Solve the ODE
dt = 9 * 60
T = 86400 * 50

prob = ODEProblem(rhs!, Y, (0.0, T), parameters)
integrator = OrdinaryDiffEq.init(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev OrdinaryDiffEq.solve!(integrator)

@info "Test case: $(test_name)"
@info "  with α: $(test_angle_name)"
@info "Solution L₂ norm at time t = 0: ", norm(Y.h)
@info "Solution L₂ norm at time t = $(T): ", norm(sol.u[end].h)
@info "Fluid volume at time t = 0: ", sum(Y.h)
@info "Fluid volume at time t = $(T): ", sum(sol.u[end].h)

if test_name == steady_state_test_name ||
   test_name == steady_state_compact_test_name
    # In these cases, we use the IC as the reference exact solution
    @info "L₁ error at T = $(T): ", norm(sol.u[end].h .- Y.h, 1)
    @info "L₂ error at T = $(T): ", norm(sol.u[end].h .- Y.h)
    @info "L∞ error at T = $(T): ", norm(sol.u[end].h .- Y.h, Inf)
    # Pointwise final L₂ error
    Plots.png(
        Plots.plot(sol.u[end].h .- Y.h),
        joinpath(path, "final_height_L2_error.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "final_height_L2_error.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Absolute error in height",
    )
    # Height errors over time
    relL1err = Array{Float64}(undef, div(T, dt))
    for t in 1:div(T, dt)
        relL1err[t] = norm(sol.u[t].h .- Y.h, 1) / norm(Y.h, 1)
    end
    Plots.png(
        Plots.plot(
            [1:dt:T],
            relL1err,
            xlabel = "time [s]",
            ylabel = "Relative L₁ err",
            label = "",
        ),
        joinpath(path, "HeightRelL1errorVstime.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "HeightRelL1errorVstime.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height relative L1 error over time",
    )

    relL2err = Array{Float64}(undef, div(T, dt))
    for t in 1:div(T, dt)
        relL2err[t] = norm(sol.u[t].h .- Y.h) / norm(Y.h)
    end
    Plots.png(
        Plots.plot(
            [1:dt:T],
            relL2err,
            xlabel = "time [s]",
            ylabel = "Relative L₂ err",
            label = "",
        ),
        joinpath(path, "HeightRelL2errorVstime.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "HeightRelL2errorVstime.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height relative L2 error over time",
    )

    RelLInferr = Array{Float64}(undef, div(T, dt))
    for t in 1:div(T, dt)
        RelLInferr[t] = norm(sol.u[t].h .- Y.h, Inf) / norm(Y.h, Inf)
    end
    Plots.png(
        Plots.plot(
            [1:dt:T],
            RelLInferr,
            xlabel = "time [s]",
            ylabel = "Relative L∞ err",
            label = "",
        ),
        joinpath(path, "HeightRelL1InferrorVstime.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "HeightRelLInferrorVstime.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height relative L_Inf error over time",
    )
else # In the non steady-state cases, we only plot the latest output of the dynamic problem
    Plots.png(Plots.plot(sol.u[end].h), joinpath(path, "final_height.png"))
    linkfig(
        relpath(
            joinpath(path, "final_height.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height field at the final time step",
    )
end
