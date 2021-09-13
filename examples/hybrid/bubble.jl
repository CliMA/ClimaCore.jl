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
function hvspace_2D(
  xlim = (-π, π),
  zlim = (0, 4π),
  helem = 10,
  velem = 50,
  npoly = 4,
)
  FT = Float64
  vertdomain = Domains.IntervalDomain(
      FT(zlim[1]),
      FT(zlim[2]);
      x3boundary = (:bottom, :top),
  )
  vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
  vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
  horzdomain = Domains.RectangleDomain(
      xlim[1]..xlim[2],
      -0..0,
      x1periodic = true,
      x2boundary = (:a, :b),
  )
  horzmesh = Meshes.EquispacedRectangleMesh(horzdomain, helem, 1)
  horztopology = Topologies.GridTopology(horzmesh)

  quad = Spaces.Quadratures.GLL{npoly + 1}()
  horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

  hv_center_space =
      Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
  hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
  return (hv_center_space, hv_face_space)
end

# set up rhs!
hv_center_space, hv_face_space = hvspace_2D((-500,500),(0,1000))
#hv_center_space, hv_face_space = hvspace_2D((-500,500),(0,30000), 5, 30)

const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const R_m = R_d # moist R, assumed to be dry

function pressure(ρθ)
  if ρθ >= 0
    return MSLP * (R_d * ρθ / MSLP)^γ
  else
    @show ρθ
    return NaN
  end
end

Φ(z) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_dry_rising_bubble_2d(x, z)
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - g * z / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    ρθ = ρ * θ # potential temperature density

    return (ρ = ρ, ρθ = ρθ, ρuₕ = ρ * Geometry.Cartesian1Vector(0.0))
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space);
face_coords = Fields.coordinate_field(hv_face_space);

Yc = map(coords) do coord
  bubble = init_dry_rising_bubble_2d(coord.x, coord.z)
  bubble
end;

ρw = map(face_coords) do coord
  Geometry.Cartesian3Vector(0.0)
end;

Y = Fields.FieldVector(Yc = Yc, ρw = ρw);

function rhs!(dY, Y, _, t)
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    interpc2f = Operators.InterpolateC2F(
      bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
      top = Operators.SetValue(Geometry.Cartesian1Vector(0.0))
    )
    interpf2c = Operators.InterpolateF2C()
    vdivf2c = Operators.DivergenceF2C(
      bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
      top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )
    vvdivc2f = Operators.DivergenceC2F()
    hdiv = Operators.Divergence()
    uvdivf2c = Operators.DivergenceF2C(
      bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0)),
      top = Operators.SetValue(Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0)),
    )
    If = Operators.InterpolateC2F(
      bottom = Operators.Extrapolate(),
      top = Operators.Extrapolate(),
    )
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )
    ∂f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )

    uₕ = @. Yc.ρuₕ / Yc.ρ
    w = @. ρw / If(Yc.ρ)
    p = @. pressure(Yc.ρθ)

    # density
    @. dYc.ρ = -(∂(w * If(Yc.ρ)))
    @. dYc.ρ -= hdiv(Yc.ρuₕ)

    # potential temperature
    @. dYc.ρθ = -(∂(w * If(Yc.ρθ)))
    @. dYc.ρθ -= hdiv(uₕ * Yc.ρθ)

    # horizontal momentum
    Ih = Ref(Geometry.Axis2Tensor((Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()), @SMatrix [1.0]))
    @. dYc.ρuₕ = -hdiv(Yc.ρuₕ ⊗ uₕ + p * Ih)

    # vertical advection of horizontal momentum
    @. dYc.ρuₕ -= uvdivf2c(ρw ⊗ interpc2f(uₕ))

    # vertical momentum
    @. dρw = B(
        Geometry.transform(Geometry.Cartesian3Axis(),
        -(∂f(p)) - If(Yc.ρ) * ∂f(Φ(coords.z))) - vvdivc2f(interpf2c(ρw ⊗ w)),
    )
    uₕc = @. interpc2f(Yc.ρuₕ / Yc.ρ)    # requires boundary conditions
    @. dρw -= hdiv(uₕc ⊗ ρw)

    #=
    # scalars
    interpc2f = Operators.InterpolateC2F(
      bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
      top = Operators.SetValue(Geometry.Cartesian1Vector(0.0))
    )
    vdivf2c = Operators.DivergenceF2C(
      bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
      top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )
    hdiv = Operators.Divergence()

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
    @. dρw  -= vvdivc2f(interpf2c(ρw ⊗ ρw) / Yc.ρ) # vvdivc2f requires boundary conditions

    # 2) pressure gradient terms
    # not sure how pressure is computed, but for now we set it to 1.0
    p = @. pressure(Yc.ρθ)

    #   a) horizontal
    Ih =  Ref(Geometry.Axis2Tensor((Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()), @SMatrix [1.0]))
    @. dYc.ρuₕ -= hdiv(p * Ih)

    #   b) vertical
    Iv =  Ref(Geometry.Axis2Tensor((Geometry.Cartesian3Axis(), Geometry.Cartesian3Axis()), @SMatrix [1.0]))
    @. dρw -= vvdivc2f(p * Iv)

    # 3) diffusion
    hgrad = Operators.Gradient()
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C()
    einterpc2f = Operators.InterpolateC2F(
      bottom = Operators.Extrapolate(),
      top = Operators.Extrapolate(),
    )
    #  a) horizontal div of horizontal grad of horiz momentun
    # TODO: a * b * c doesn't work, need to provide recursive pairwise method? use parens for now
    κ = 10.0 # m^2/s
    @. dYc.ρuₕ += hdiv(κ * (Yc.ρ * hgrad(Yc.ρuₕ / Yc.ρ)))

    #  b) vertical div of vertical grad of horiz momentun
    Yfρ = @. einterpc2f(Yc.ρ)
    @. dYc.ρuₕ += uvdivf2c(κ * (Yfρ * gradc2f(Yc.ρuₕ / Yc.ρ)))

    #  c) horizontal div of horizontal grad of vert momentum
    @. dρw += hdiv(κ * (Yfρ * hgrad(ρw / Yfρ)))

    #  d) vertical div of vertical grad of vert momentun
    @. dρw += vvdivc2f(κ * (Yc.ρ * gradf2c(ρw / Yfρ)))

    #W = parent(ClimaCore.column(dρw, 1, 1))
    #@show W[1] W[2] W[end-1] W[end]
    =#

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)

    return dY
end

dYdt = similar(Y);
rhs!(dYdt, Y, nothing, 0.0);

# run!
using OrdinaryDiffEq
Δt = 0.02
prob = ODEProblem(rhs!, Y, (0.0, 700.0))
sol = solve(prob, SSPRK33(), dt = Δt, progress=true);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "bubble"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# post-processing
using Plots
anim = Plots.@animate for u in sol.u
     Plots.plot(u.Yc.ρθ ./ u.Yc.ρ)
end
Plots.mp4(anim, "result.mp4", fps = 20)