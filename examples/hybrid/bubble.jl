push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
import ClimaCore.Domains.Geometry: Cartesian2DPoint
using ClimaCore.Geometry

# set up function space
function hvspace_2D()
    FT = Float64
    vertdomain =
        Domains.IntervalDomain(FT(-1), FT(1); x3boundary = (:bottom, :top))
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 64)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vert_center_space)

    horzdomain = Domains.RectangleDomain(
        -1..1,
        -0..0,
        x1periodic = true,
        x2boundary = (:a, :b),
    )
    horzmesh = Meshes.EquispacedRectangleMesh(horzdomain, 10, 1)
    horztopology = Topologies.GridTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{8}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    return (hv_center_space, hv_face_space)
end

# set up rhs!
hv_center_space, hv_face_space = hvspace_2D()


function flux(state, p)
  @unpack ρ, ρu, ρθ = state
  u = ρu / ρ
  return (
      ρ = ρu,
      ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * LinearAlgebra.I),
      ρθ = ρθ * u,
  )
end



function rhs!(dY, Y, _, t)
    ρ = Y.ρ
    ρθ = Y.ρθ
    ρuₕ = Y.ρuₕ
    ρw = Y.ρw
    dρ = dY.ρ
    dρθ = dY.ρθ
    dρuₕ = dY.ρuₕ
    dρw = dY.ρw

    # density
    Ic2f = Operators.InterpolateC2F(
        top=Operators.Extrapolate()
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian13Vector(0.0,0.0)),
    )
    hdiv = Operators.Divergence()
    @. dρ = -divf2c(ρw)
    @. dρ -= hdiv(ρuₕ)
    Spaces.weighted_dss!(dρ)

    # velocities (auxiliary)
    @. uₕ = ρuₕ / ρ
    @. w = ρw / ρ

    # momenta

    -div(ρu * u' + p*I + ρ * ν * grad(u))

    T = ρu * u' + p*I + ρ * ν * grad(u)
    ρu

    hdiv(ρuₕ * uₕ')

    dρuₕ = -hdiv(ρuₕ * uₕ') - vdiv(ρw * uₕ') + div(ρν * grad(uₕ))

    dρw = -hdiv(ρuₕ * w') - vdiv(ρw * w') + ρ * div(ν * grad(w))

    # potential temperature density
    Ic2f = Operators.InterpolateC2F(
        top=Operators.Extrapolate()
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian13Vector(0.0,0.0)),
    )
    hdiv = Operators.Divergence()
    @. dρθ = -divf2c(ρθ * w)
    @. dρθ -= hdiv(ρθ * uₕ)
    Spaces.weighted_dss!(dρθ)

    # Eq2. d\rho = -\div( \rhou\_h + \rhow )
    # Eq3a. d\rhou_h = -\div( \rhou_h \circtimes \rhou/\rho) - \grad_h(p) - \grad_h(\Phi)
    #    b. d\rhow = -\div( \rhow \circtimes \rhou/\rho) - \grad_v(p) - \grad_v(\Phi)
    # Eq4. d\rho\theta = -\div( \rho\theta * \rhou\_h / \rho)
    # p is pressure, \Phi is geopotential

    return dY
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
ρ = map(coords) do coord
end
ρuₕ = map(coords) do coord
    Geometry.Cartesian1Vector.(ones(Float64, hv_center_space),)
end
ρw = map(coords) do coord
    Geometry.Cartesian13Vector.(
        zeros(Float64, hv_face_space),
        ones(Float64, hv_face_space),
    )
end
ρθ = map(coords) do coord
    exp(-((coord.x + 0.5)^2 + (coord.z + 0.5)^2) / (2 * 0.2^2))
end
Yc = map(coords) do coord
  (
    ρ = exp(-((coord.x + 0.5)^2 + (coord.z + 0.5)^2) / (2 * 0.2^2)),
    ρuₕ = Geometry.Cartesian1Vector.(1.0),
  )
end

function hflux(state, ∇u)
  @unpack ρ, ρuₕ = state
  uₕ = ρuₕ / ρ
  ν = 0.1
  return (
      ρ = ρuₕ,
      ρuₕ = ((ρuₕ ⊗ uₕ) + 0.0 * LinearAlgebra.I + ρ*ν*∇u),
  )
end
hgrad = Operators.Gradient()
@. hflux(Yc, hgrad(Yc.ρuₕ / Yc.ρ))


Y = Fields.FieldVector(ρ = ρ, ρuₕ = ρuₕ, ρw = ρw, ρθ = ρθ)

# run!
using OrdinaryDiffEq
Δt = 0.01
prob = ODEProblem(rhs!, Y, (0.0, 1.0))
sol = solve(prob, SSPRK33(), dt = Δt, saveat=0.05);

# post-processing
using Plots
Plots.png(Plots.plot(sol.u[1].h), "initial.png")
Plots.png(Plots.plot(sol.u[end].h), "final.png")

anim = Plots.@animate for u in sol.u
     Plots.plot(u.h, clim = (0, 1))
end
Plots.mp4(anim, "movie.mp4", fps = 10)
