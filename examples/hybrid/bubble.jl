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
  helem = 5,
  velem = 100,
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

Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_d / C_v)
Φ(z) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_dry_rising_bubble_2d(x, z)
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.0 #0.5
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

    return (ρ = ρ, ρθ = ρθ,)
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coords) do coord
  bubble = init_dry_rising_bubble_2d(coord.x, coord.z)
  bubble
end

ρw = map(face_coords) do coord
  Geometry.Cartesian3Vector(0.0)
end

Y = Fields.FieldVector(Yc = Yc, ρw = ρw)

# function rhs!(dY, Y, _, t)
#     ρw = Y.ρw
#     Yc = Y.Yc
#     dYc = dY.Yc
#     dρw = dY.ρw

#     # pressure gradient force
#     ∂f = Operators.GradientC2F(
#         bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
#         top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
#     )
#     p = @. pressure(Yc.ρθ)
#     @. dρw = -∂f(p)

#     # gravity
#     einterpc2f = Operators.InterpolateC2F(
#       bottom = Operators.Extrapolate(),
#       top = Operators.Extrapolate(),
#     )
#     #gradc2f = Operators.GradientC2F(
#     #  bottom = Operators.SetGradient(0.0),
#     #  top = Operators.SetGradient(0.0)
#     #)
#     #@. dρw -= einterpc2f(Yc.ρ) * gradc2f(Φ(coords.z))
#     @. dρw -= einterpc2f(Yc.ρ) * Geometry.Covariant3Vector(grav)

#     return dY
# end
function rhs!(dY, Y, _, t)
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

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

    w = @. ρw / If(Yc.ρ)
    @. dYc.ρ = -(∂(w * If(Yc.ρ)))
    @. dYc.ρθ = -(∂(w * If(Yc.ρθ)))
    # @. dw = B(
    #     Geometry.transform(Geometry.Cartesian3Axis(),
    #     -(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - ∂f(Φ(coords.z))),
    # )
    @. dρw = B(
        Geometry.transform(Geometry.Cartesian3Axis(),
        -(∂f(pressure(Yc.ρθ))) - If(Yc.ρ) * ∂f(Φ(coords.z))),
    )
    return dY
end



dYdt = similar(Y)
rhs!(dYdt, Y, nothing, 0.0);

#=
# run!
using OrdinaryDiffEq
Δt = 0.01
prob = ODEProblem(rhs!, Y, (0.0, 100.0))
sol = solve(prob, SSPRK33(), dt = Δt, progress=true);
=#
# post-processing
#=
using Plots
anim = Plots.@animate for u in sol.u
     Plots.plot(u.Yc.ρθ)
end
Plots.mp4(anim, "result.mp4", fps = 10)
=#
