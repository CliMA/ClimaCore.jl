using Test
using LinearAlgebra

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
import ClimaCore.Utilities: half

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

# This experiment tests
#     1) hydrostatic and geostrophic balance;
#     2) linear instability.
# - "baroclinic_wave": the defaul simulation, following https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2241.
# - "balanced_flow": the same balanced background flow as the baroclinic wave but with zero perturbation.

# test specifications
const test_name = get(ARGS, 1, "baroclinic_wave") # default test case to run baroclinic wave

# parameters
const R = 6.371229e6 # radius
const grav = 9.80616 # gravitational constant
const Ω = 7.29212e-5 # Earth rotation (radians / sec)
const R_d = 287.0 # R dry (gas constant / mol mass dry air)
const κ = 2 / 7 # kappa
const γ = 1.4 # heat capacity ratio
const cp_d = R_d / κ # heat capacity at constant pressure
const cv_d = cp_d - R_d # heat capacity at constant volume
const p_0 = 1.0e5 # reference pressure
const k = 3
const T_e = 310 # temperature at the equator
const T_p = 240 # temperature at the pole
const T_0 = 0.5 * (T_e + T_p)
const T_tri = 273.16 # triple point temperature
const Γ = 0.005
const A = 1 / Γ
const B = (T_0 - T_p) / T_0 / T_p
const C = 0.5 * (k + 2) * (T_e - T_p) / T_e / T_p
const b = 2
const H = R_d * T_0 / grav
const z_t = 15.0e3
const λ_c = 20.0
const ϕ_c = 40.0
const d_0 = R / 6
const V_p = 1.0

τ_z_1(z) = exp(Γ * z / T_0)
τ_z_2(z) = 1 - 2 * (z / b / H)^2
τ_z_3(z) = exp(-(z / b / H)^2)
τ_1(z) = 1 / T_0 * τ_z_1(z) + B * τ_z_2(z) * τ_z_3(z)
τ_2(z) = C * τ_z_2(z) * τ_z_3(z)
τ_int_1(z) = A * (τ_z_1(z) - 1) + B * z * τ_z_3(z)
τ_int_2(z) = C * z * τ_z_3(z)
F_z(z) = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
I_T(ϕ) = cosd(ϕ)^k - k / (k + 2) * (cosd(ϕ))^(k + 2)
T(ϕ, z) = (τ_1(z) - τ_2(z) * I_T(ϕ))^(-1)
p(ϕ, z) = p_0 * exp(-grav / R_d * (τ_int_1(z) - τ_int_2(z) * I_T(ϕ)))
r(λ, ϕ) = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
U(ϕ, z) =
    grav * k / R * τ_int_2(z) * T(ϕ, z) * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
u(ϕ, z) = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U(ϕ, z))
v(ϕ, z) = 0.0
c3(λ, ϕ) = cos(π * r(λ, ϕ) / 2 / d_0)^3
s1(λ, ϕ) = sin(π * r(λ, ϕ) / 2 / d_0)
cond(λ, ϕ) = (0 < r(λ, ϕ) < d_0) * (r(λ, ϕ) != R * pi)
if test_name == "baroclinic_wave"
    δu(λ, ϕ, z) =
        -16 * V_p / 3 / sqrt(3) *
        F_z(z) *
        c3(λ, ϕ) *
        s1(λ, ϕ) *
        (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
        sin(r(λ, ϕ) / R) * cond(λ, ϕ)
    δv(λ, ϕ, z) =
        16 * V_p / 3 / sqrt(3) *
        F_z(z) *
        c3(λ, ϕ) *
        s1(λ, ϕ) *
        cosd(ϕ_c) *
        sind(λ - λ_c) / sin(r(λ, ϕ) / R) * cond(λ, ϕ)
    const κ₄ = 1.0e16 # m^4/s
elseif test_name == "balanced_flow"
    δu(λ, ϕ, z) = 0.0
    δv(λ, ϕ, z) = 0.0
    const κ₄ = 0.0
end
uu(λ, ϕ, z) = u(ϕ, z) + δu(λ, ϕ, z)
uv(λ, ϕ, z) = v(ϕ, z) + δv(λ, ϕ, z)

# set up function space
function sphere_3D(
    R = 6.37122e6,
    zlim = (0, 12.0e3),
    helem = 4,
    zelem = 12,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

Φ(z) = grav * z

function pressure(ρ, e, normuvw, z)
    I = e - Φ(z) - normuvw^2 / 2
    T = I / cv_d + T_tri
    return ρ * R_d * T
end

# set up 3D domain - spherical shell
hv_center_space, hv_face_space = sphere_3D(R, (0, 30.0e3), 4, 10, 4)

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
local_geometries = Fields.local_geometry_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

function initial_condition(ϕ, λ, z)
    ρ = p(ϕ, z) / R_d / T(ϕ, z)
    e = cv_d * (T(ϕ, z) - T_tri) + Φ(z) + (uu(λ, ϕ, z)^2 + uv(λ, ϕ, z)^2) / 2
    ρe = ρ * e

    return (ρ = ρ, ρe = ρe)
end

function initial_condition_velocity(local_geometry)
    coord = local_geometry.coordinates
    ϕ = coord.lat
    λ = coord.long
    z = coord.z
    return Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.UVVector(uu(λ, ϕ, z), uv(λ, ϕ, z)),
        local_geometry,
    )
end
# Coriolis
const f =
    @. Geometry.Contravariant3Vector(Geometry.WVector(2 * Ω * sind(coords.lat)))

Yc = map(coord -> initial_condition(coord.lat, coord.long, coord.z), coords)
uₕ = map(
    local_geometry -> initial_condition_velocity(local_geometry),
    local_geometries,
)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

function rhs!(dY, Y, _, t)
    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe = Y.Yc.ρe # scalar on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    z = coords.z

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.Divergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.Gradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.Curl()

    dρ .= 0 .* cρ

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients

    χe = @. dρe = hwdiv(hgrad(cρe / cρ))
    χuₕ = @. duₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)

    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. duₕ =
        -κ₄ * (
            hwgrad(hdiv(χuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(χuₕ))),
            )
        )

    # 1) Mass conservation
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    )
    cω³ = hcurl.(cuₕ) # Contravariant3Vector
    fω¹² = hcurl.(fw) # Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant12Vector.(
            Geometry.Covariant123Vector.(Ic2f.(cuₕ)),
        ) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    ce = @. cρe / cρ
    cp = @. pressure(cρ, ce, norm(cuvw), z)

    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuvw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy

    @. dρe -= hdiv(cuvw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end

dYdt = similar(Y)
rhs!(dYdt, Y, nothing, 0.0)

# run!
using OrdinaryDiffEq
# Solve the ODE
if test_name == "baroclinic_wave"
    time_end = 600
elseif test_name == "balanced_flow"
    time_end = 3600
end
dt = 5
prob = ODEProblem(rhs!, Y, (0.0, time_end))

haskey(ENV, "CI_PERF_SKIP_RUN") && exit() # for performance analysis

sol = @timev solve(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

# visualization artifacts
if test_name == "baroclinic_wave"
    @info "Solution L₂ norm at time t = 0: ", norm(Y.Yc.ρe)
    @info "Solution L₂ norm at time t = $(time_end): ", norm(sol.u[end].Yc.ρe)

    ENV["GKSwstype"] = "nul"
    using ClimaCorePlots, Plots
    Plots.GRBackend()
    dirname = "baroclinic_wave"
    path = joinpath(@__DIR__, "output", dirname)
    mkpath(path)

    function linkfig(figpath, alt = "")
        # buildkite-agent upload figpath
        # link figure in logs if we are running on CI
        if get(ENV, "BUILDKITE", "") == "true"
            artifact_url = "artifact://$figpath"
            print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
        end
    end

    u_phy = Geometry.transform.(Ref(Geometry.UVAxis()), sol.u[end].uₕ)
    Plots.png(
        Plots.plot(u_phy.components.data.:2, level = 3, clim = (-1, 1)),
        joinpath(path, "v.png"),
    )
    w_phy = Geometry.transform.(Ref(Geometry.WAxis()), sol.u[end].w)
    Plots.png(
        Plots.plot(w_phy.components.data.:1, level = 3 + half, clim = (-1, 1)),
        joinpath(path, "w.png"),
    )
elseif test_name == "balanced_flow"
    ENV["GKSwstype"] = "nul"
    using ClimaCorePlots, Plots
    Plots.GRBackend()
    dirname = "balanced_flow"
    path = joinpath(@__DIR__, "output", dirname)
    mkpath(path)

    function linkfig(figpath, alt = "")
        # buildkite-agent upload figpath
        # link figure in logs if we are running on CI
        if get(ENV, "BUILDKITE", "") == "true"
            artifact_url = "artifact://$figpath"
            print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
        end
    end

    u_phy = Geometry.transform.(Ref(Geometry.UVAxis()), sol.u[end].uₕ)
    Plots.png(
        Plots.plot(u_phy.components.data.:1, level = 3),
        joinpath(path, "u_end.png"),
    )

    u_err =
        Geometry.transform.(
            Ref(Geometry.UVAxis()),
            sol.u[end].uₕ .- sol.u[1].uₕ,
        )
    Plots.png(
        Plots.plot(u_err.components.data.:1, level = 3, clim = (-1, 1)),
        joinpath(path, "u_err.png"),
    )

    w_err = Geometry.transform.(Ref(Geometry.WAxis()), sol.u[end].w)
    Plots.png(
        Plots.plot(w_err.components.data.:1, level = 3 + half, clim = (-4, 4)),
        joinpath(path, "w_err.png"),
    )

    @test sol.u[end].Yc.ρ ≈ sol.u[1].Yc.ρ rtol = 5e-2
    @test sol.u[end].Yc.ρe ≈ sol.u[1].Yc.ρe rtol = 5e-2
    @test sol.u[end].uₕ ≈ sol.u[1].uₕ rtol = 5e-2
end
