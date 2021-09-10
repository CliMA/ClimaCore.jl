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


# initial conditions
coords = Fields.coordinate_field(hv_center_space)

Yc = map(coords) do coord
  ρ = exp(-((coord.x)^2 + (coord.z + 0.5)^2) / (2 * 0.2^2))
  (
    ρ = ρ,
    ρθ = ρ,
    ρuₕ = ρ * Geometry.Cartesian1Vector(1.0),
  )
end
ρw = map(Fields.coordinate_field(hv_face_space)) do coord
  ρ = exp(-((coord.x)^2 + (coord.z + 0.5)^2) / (2 * 0.2^2))
  ρ * Geometry.Cartesian3Vector(1.0)
end

Y = Fields.FieldVector(Yc=Yc, ρw=ρw)



#=
function flux(state, p)
  @unpack ρ, ρu, ρθ = state
  u = ρu / ρ
  return (
      ρ = ρu,
      ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * LinearAlgebra.I),
      ρθ = ρθ * u,
  )
end
=#


function rhs!(dY, Y, _, t)
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    # scalars
    interpc2f = Operators.InterpolateC2F(
        bottom=Operators.Extrapolate(),
        top=Operators.Extrapolate()
    )
    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )
    hdiv = Operators.Divergence()

    @. dYc.ρ = -vdivf2c(ρw)
    @. dYc.ρ -= hdiv(Yc.ρuₕ)

    @. dYc.ρθ = -vdivf2c(ρw * interpc2f( Yc.ρθ / Yc.ρ ))
    @. dYc.ρθ -= hdiv(Yc.ρuₕ * Yc.ρθ / Yc.ρ)

    # 1) dρu = -div(ρu ⊗ ρu / ρ)
    #   a) horizontal advection of horizontal momentum
    @. dYc.ρuₕ = -hdiv(Yc.ρuₕ ⊗ Yc.ρuₕ / Yc.ρ)
    #   b) vertical advection of horizontal momentum
    uvdivf2c = Operators.DivergenceF2C(
      bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0)),
      top = Operators.SetValue(Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0)), # ?
    )
    @. dYc.ρuₕ -= uvdivf2c(ρw ⊗ interpc2f(Yc.ρuₕ / Yc.ρ))

    #   c) horizontal advection of vertical momentum
    uₕc = @. interpc2f(Yc.ρuₕ / Yc.ρ)    # requires boundary conditions
    @. dρw  = -hdiv(uₕc ⊗ ρw)
    #   d) vertical advection of vertical momentum
    interpf2c = Operators.InterpolateF2C()
    vvdivc2f = Operators.DivergenceC2F(
      bottom = Operators.SetGradient(Geometry.Cartesian3Vector(0.0)),
      top = Operators.SetGradient(Geometry.Cartesian3Vector(0.0))
    )
    @. dρw  -= vvdivc2f(interpf2c(ρw ⊗ ρw) / Yc.ρ) # vdivc2f requires boundary conditions

    # 2) pressure gradient terms
    # not sure how pressure is computed, but for now we set it to 1.0
    p = @. 0.0 * Yc.ρ
    #   a) horizontal
    Ih =  Ref(Geometry.Axis2Tensor((Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()), @SMatrix [1.0]))
    @. dYc.ρuₕ -= hdiv(p * Ih)
    #   b) vertical
    Iv =  Ref(Geometry.Axis2Tensor((Geometry.Cartesian3Axis(), Geometry.Cartesian3Axis()), @SMatrix [1.0]))
    @. ρw -= vvdivc2f(p * Iv)

    # 3) diffusion

    hgrad = Operators.Gradient()
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C()

    #  a) horizontal div of horizontal grad of horiz momentun
    # TODO: a * b * c doesn't work, need to provide both methods, use parens for now
    κ = 1.0
    @. dYc.ρuₕ -= hdiv(κ * (Yc.ρ * hgrad(Yc.ρuₕ / Yc.ρ)))

    #  b) vertical div of vertical grad of horiz momentun
    Yfρ = @. interpc2f(Yc.ρ)
    @. dYc.ρuₕ -= uvdivf2c(κ * (Yfρ * gradc2f(Yc.ρuₕ / Yc.ρ)))

    #  c) horizontal div of horizontal grad of vert momentum
    # TODO: output is Cartesian3, computed result is Cartesian1
    # uₕc -> Cartesian1Vector
    # hgrad(uₕc) -> AT(CovariantAxis() , CartesianAxis())
    # hdiv(res) -> Cartesian1Vector()
    @. dρw -= hdiv(κ * (Yfρ * hgrad(uₕc)))

    #  d) vertical div of vertical grad of vert momentun
    @. dρw -= vvdivc2f(κ * (Yc.ρ * gradf2c(ρw / Yfρ)))

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    #=
    @. dρ   = -vdivf2c(ρw) # density
    @. dρ  -= hdiv(ρuₕ)

    @. dρθ  = -vdivf2c(interpc2f(ρθ) * w) # potential temperature density
    @. dρθ -= hdiv(ρθ * uₕ)
    Spaces.weighted_dss!(dρ)
    Spaces.weighted_dss!(dρθ)
    =#
    #=
    # vectors
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
    =#



    # Eq2. d\rho = -\div( \rhou\_h + \rhow )
    # Eq3a. d\rhou_h = -\div( \rhou_h \circtimes \rhou/\rho) - \grad_h(p) - \grad_h(\Phi)
    #    b. d\rhow = -\div( \rhow \circtimes \rhou/\rho) - \grad_v(p) - \grad_v(\Phi)
    # Eq4. d\rho\theta = -\div( \rho\theta * \rhou\_h / \rho)
    # p is pressure, \Phi is geopotential
    return dY
end

dYdt = similar(Y)
rhs!(dYdt, Y, nothing, 0.0)

# ρᶜ, ρθᶜ, uᶜ, wᶠ

#=
function hflux(state, ∇ᵢu, local_geom)
  @unpack ρ, ρuₕ = state
  uₕ = ρuₕ / ρ
  ν = 0.1
  ∂ξ¹∂x₁ = local_geom.∂ξ∂x[1,1]
  ∇₁u = ∇ᵢu[1]
  ∇u = Geometry.Axis2Tensor((Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()), SMatrix{1,1}(∂ξ¹∂x₁ * ∇₁u))
  return (
      ρ = ρuₕ,
      ρuₕ = ((ρuₕ ⊗ uₕ) + 0.0 * LinearAlgebra.I + ρ*ν*∇u),
  )
end
hgrad = Operators.Gradient()
∇u = @. hgrad(Yc.ρuₕ / Yc.ρ)

s1 = ClimaCore.slab(Fields.field_values(Yc),1,1)[1]
∇u1 = ClimaCore.slab(Fields.field_values(∇u),1,1)[1]
local_geom = ClimaCore.slab(Spaces.local_geometry_data(axes(Yc)),1,1)[1]

hflux(s1, ∇u1, local_geom)

u
v
w

v\partial_xu = v * e_x \cdot grad(u)

@. hflux(Yc, hgrad(Yc.ρuₕ / Yc.ρ), Fields.local_geometry_field(axes(Yc)))
=#

# run!
using OrdinaryDiffEq
Δt = 0.01
prob = ODEProblem(rhs!, Y, (0.0, 1.0))
sol = solve(prob, SSPRK33(), dt = Δt, saveat=0.05);

#=
# post-processing
using Plots
Plots.png(Plots.plot(sol.u[1].h), "initial.png")
Plots.png(Plots.plot(sol.u[end].h), "final.png")

anim = Plots.@animate for u in sol.u
     Plots.plot(u.h, clim = (0, 1))
end
Plots.mp4(anim, "movie.mp4", fps = 10)
=#