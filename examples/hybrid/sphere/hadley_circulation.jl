# Hadley-like meridional circulation on the sphere (DCMIP 2012, Test 1-2). A
# prescribed overturning flow advects a tracer in the latitude-height plane and
# reverses, so the tracer should return to its initial state. Exercises vertical
# transport and its coupling to the horizontal, which pure horizontal advection
# tests leave untouched. Reference:
# http://www-personal.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf,
# Section 1.2
import ClimaComms
ClimaComms.@import_required_backends
using Test
using LinearAlgebra

import ClimaCore:
    ClimaCore,
    slab,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Operators
import ClimaCore.Utilities: half

import ClimaTimeSteppers as CTS

using NCDatasets, ClimaCoreTempestRemap
using Statistics: mean

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()


# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dir = "hadley_circulation"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

include(joinpath(@__DIR__, "../..", "example_utils.jl")) # linkfig


# set up function space
function sphere_3D(
    R = 6.37122e6,
    zlim = (0, 12.0e3),
    helem = 6,
    zelem = 30,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(
        vertdomain,
        Meshes.ExponentialStretching(FT(7e3));
        nelems = zelem,
    )
    device = ClimaComms.device(context)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

function tendency!(yₜ, y, parameters, t)
    (; dt, ρ, u, uₕ, uᵥ, q1, Δₕq1) = parameters
    Ic2f = Operators.InterpolateC2F()
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    upwind1 = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    upwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    FCTZalesak = Operators.FCTZalesak(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    function local_velocity(coord)
        (; z, lat) = coord
        U = u₀ * cosd(lat)
        V =
            -(R * w₀ * pi * ρ₀ / K / z_top / ρ_ref(z)) *
            sind(K * lat) *
            cosd(lat) *
            cos(pi * z / z_top) *
            cos(pi * t / τ)
        W =
            (w₀ * ρ₀ / K / ρ_ref(z)) *
            (-2 * sind(K * lat) * sind(lat) + K * cosd(lat) * cosd(K * lat)) *
            sin(pi * z / z_top) *
            cos(pi * t / τ)
        return Geometry.UVWVector(U, V, W)
    end
    center_coord = Fields.coordinate_field(uₕ)
    face_coord = Fields.coordinate_field(uᵥ)
    @. u = local_velocity(center_coord)
    @. uₕ = Geometry.project(Geometry.Covariant12Axis(), u)
    @. uᵥ =
        Geometry.project(Geometry.Covariant3Axis(), local_velocity(face_coord))

    @. q1 = y.ρq1 / ρ
    @. Δₕq1 = hwdiv(hgrad(q1))
    Spaces.weighted_dss!(Δₕq1)

    # Horizontal transport and hyperdiffusion
    @. yₜ.ρq1 = -hdiv(y.ρq1 * u) - κ₄ * hwdiv(ρ * hgrad(Δₕq1))

    # Vertical transport due to horizontal velocity
    @. yₜ.ρq1 -= vdivf2c(Ic2f(y.ρq1 * uₕ))

    # Vertical transport due to vertical velocity, corrected by Zalesak FCT
    @. yₜ.ρq1 -= vdivf2c(
        Ic2f(ρ) * (
            upwind1(uᵥ, q1) + FCTZalesak(
                upwind3(uᵥ, q1) - upwind1(uᵥ, q1),
                q1 / dt,
                q1 / dt - vdivf2c(Ic2f(ρ) * upwind1(uᵥ, q1)) / ρ,
            )
        ),
    )
end

function dss!(y, parameters, t)
    Spaces.weighted_dss!(y.ρq1)
end

const R = 6.37122e6        # radius
const grav = 9.8           # gravitational constant
const R_d = 287.058        # R dry (gas constant / mol mass dry air)
const z_top = 1.2e4        # height position of the model top
const p_top = 25494.4      # pressure at the model top
const T₀ = 300             # isothermal atmospheric temperature
const H = R_d * T₀ / grav  # scale height
const p₀ = 1.0e5           # reference pressure at the surface
const ρ₀ = p₀ / R_d / T₀   # density at the surface
const τ = 86400.0          # period of motion (1 day)
const K = 5                # number of overturning cells
const u₀ = 40.0            # reference zonal velocity
const w₀ = 0.15            # reference vertical velocity
const z₁ = 2000.0          # lower boundary of tracer layer
const z₂ = 5000.0          # upper boundary of tracer layer
const κ₄ = 1.0e16          # hyperviscosity

# time constants
t_end = τ                  # one full period, so the circulation reverses exactly
dt = 15.0 * 60.0

# set up 3D domain
hv_center_space, hv_face_space = sphere_3D()

p(z) = p₀ * exp(-z / H)
ρ_ref(z) = p(z) / (R_d * T₀)

# initial conditions
center_coord = Fields.coordinate_field(hv_center_space)
ρ = @. ρ_ref(center_coord.z)
q1_init = map(center_coord) do coord
    if z₁ < coord.z < z₂
        z₀ = (z₂ + z₁) / 2
        return (1.0 + cos(2pi * (coord.z - z₀) / (z₂ - z₁))) / 2
    else
        return 0.0
    end
end
y = Fields.FieldVector(ρq1 = ρ .* q1_init)

# run!
parameters = (;
    dt,
    ρ,
    u = Fields.Field(Geometry.UVWVector{Float64}, hv_center_space),
    uₕ = Fields.Field(Geometry.Covariant12Vector{Float64}, hv_center_space),
    uᵥ = Fields.Field(Geometry.Covariant3Vector{Float64}, hv_face_space),
    q1 = Fields.Field(Float64, hv_center_space),
    Δₕq1 = Fields.Field(Float64, hv_center_space),
)
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = tendency!, dss!),
    y,
    (0.0, t_end),
    parameters,
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = dt,
    saveat = collect(0.0:dt:t_end),
)

q1_series = map(u -> u.ρq1 ./ ρ, sol.u)
q1_deviations =
    map(q -> norm(q .- q1_series[1]) / norm(q1_series[1]), q1_series)
q1_error = q1_deviations[end]
rel_mass_err = abs(sum(sol.u[end].ρq1) - sum(sol.u[1].ρq1)) / sum(sol.u[1].ρq1)

# The overturning cells lift the tracer layer through the first half period and
# put it back through the second, so the mass-weighted mean height of the tracer
# traces out an excursion and returns (measured: 3.47 km → 5.22 km → 3.52 km).
z_center = Fields.coordinate_field(hv_center_space).z
mean_tracer_height = map(u -> sum(u.ρq1 .* z_center) / sum(u.ρq1), sol.u)

# The flow must displace the tracer: without this the test is
# passed by a tracer that never moves (measured peak deviation: 1.32, peak
# height excursion: 1.75 km).
@test maximum(q1_deviations) > 1
@test maximum(mean_tracer_height) - mean_tracer_height[1] > 1.5e3
# Having been displaced, it must come back (measured: 0.30 in the L₂ norm, and
# to within 56 m of its initial mean height).
@test q1_error < 0.33
@test abs(mean_tracer_height[end] - mean_tracer_height[1]) < 100
# The flux-form transport conserves tracer mass exactly (measured drift: 0).
@test rel_mass_err ≤ 4eps()
# The vertical transport is FCT-limited, but the horizontal transport is
# spectral and unlimited, so the sharp edges of the tracer layer ring: the
# excursion outside the initial [0, 1] range reaches 36% of the tracer range
# (with κ₄ = 0 it reaches 210%, so the hyperdiffusion is what contains it).
@test minimum(minimum, q1_series) > -0.5
@test maximum(maximum, q1_series) < 1.2

Plots.png(
    Plots.plot(q1_series[cld(end, 2)], level = 11, clim = (-0.1, 1.1)),
    joinpath(path, "q1_half_day_level_11.png"),
)

# Remapping infrastructure to have lat-altitude plots
remap_tmpdir = path * "/remaptmp/"
mkpath(remap_tmpdir)
datafile_cc = remap_tmpdir * "cc.nc"
NCDataset(datafile_cc, "c") do nc
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, hv_center_space, type = "cgll")
    def_space_coord(nc, hv_face_space, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by
    # default, unlimited size)
    nc_time = def_time_coord(nc)
    # defines the variable
    nc_q1 = defVar(nc, "q1", Float64, hv_center_space, ("time",))

    # write data to netcdf file
    for i in 1:length(sol.u)
        nc_time[i] = sol.t[i]
        nc_q1[:, i] = sol.u[i].ρq1 ./ ρ
    end
end

# write out our cubed sphere mesh
meshfile_cc = remap_tmpdir * "mesh_cubedsphere.g"
write_exodus(meshfile_cc, Spaces.topology(hv_center_space))

# write out RLL mesh
nlat = 90
nlon = 180
meshfile_rll = remap_tmpdir * "mesh_rll.g"
rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

# construct overlap mesh
meshfile_overlap = remap_tmpdir * "mesh_overlap.g"
overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

# construct remap weight file
weightfile = remap_tmpdir * "remap_weights.nc"
remap_weights(
    weightfile,
    meshfile_cc,
    meshfile_rll,
    meshfile_overlap;
    in_type = "cgll",
    in_np = Quadratures.degrees_of_freedom(
        Spaces.quadrature_style(hv_center_space),
    ),
)

# apply remap
datafile_rll = remap_tmpdir * "data_rll.nc"
apply_remap(datafile_rll, datafile_cc, weightfile, ["q1"])

# load remapped data and create statistics for plots
nt = NCDataset(datafile_rll, "r") do nc
    lat = Array(nc["lat"])
    z = Array(nc["z"])
    time = Array(nc["time"])
    q1 = Array(nc["q1"])
    (; lat, z, time, q1)
end
(; lat, z, time, q1) = nt

# calculate zonal average for plotting
calc_zonalave(x) = dropdims(mean(x, dims = 1), dims = 1)

q1_zonalave = calc_zonalave(q1)

times = 0.0:dt:t_end
levels = range(0, 1.1; step = 0.1) # contour levels
anim = Plots.@animate for i in 1:length(times)
    Plots.contourf(
        lat,
        z,
        q1_zonalave[:, :, i]',
        color = :balance,
        linewidth = 0,
        title = "q1 zonal average",
        xlabel = "lat (deg N)",
        ylabel = "z (m)",
        clim = (-0.1, 1.1),
        levels = levels,
    )
end

Plots.mp4(anim, joinpath(path, "q1_zonal_ave_anim.mp4"), fps = 10)

# remove temp remapping files
rm(remap_tmpdir, recursive = true)
