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

using OrdinaryDiffEq: ODEProblem, solve
using DiffEqBase
using ClimaTimeSteppers

using NCDatasets, ClimaCoreTempestRemap
using Statistics: mean

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

# 3D Hadley-like Meridional Circulation (DCMIP 2012 Test 1-2)
# Reference: http://www-personal.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf, Section 1.2


# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dir = "hadley_circulation"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end


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

# RHS tendency specification
function rhs!(dydt, y, parameters, t, alpha, beta)

    (; coords, face_coords, τ, u₀, w₀, K, ρ₀, ystar) = parameters

    ϕ = coords.lat
    zc = coords.z
    zf = face_coords.z
    ϕf = face_coords.lat

    uu = @. u₀ * cosd(ϕ)
    uv = @. -(R * w₀ * pi * ρ₀ / K / z_top / ρ_ref(zc)) *
       sind(K * ϕ) *
       cosd(ϕ) *
       cos(pi * zc / z_top) *
       cos(pi * t / τ)
    uw = @. (w₀ * ρ₀ / K / ρ_ref(zf)) *
       (-2 * sind(K * ϕf) * sind(ϕf) + K * cosd(ϕf) * cosd(K * ϕf)) *
       sin(pi * zf / z_top) *
       cos(pi * t / τ)

    uₕ = Geometry.Covariant12Vector.(Geometry.UVVector.(uu, uv))
    w = Geometry.Covariant3Vector.(Geometry.WVector.(uw))

    # aliases
    ρ = y.ρ
    ρq1 = y.ρq1
    dρq1 = dydt.ρq1

    # No change in density: divergence-free flow
    @. dydt.ρ = beta * dydt.ρ + 0 .* y.ρ

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    third_order_upwind_c2f = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    ### HYPERVISCOSITY
    @. ystar.ρq1 = hwdiv(hgrad(ρq1 / ρ))
    Spaces.weighted_dss!(ystar.ρq1)
    @. ystar.ρq1 = -κ₄ * hwdiv(ρ * hgrad(ystar.ρq1))

    cw = If2c.(w)
    cuvw = Geometry.Covariant123Vector.(uₕ) .+ Geometry.Covariant123Vector.(cw)

    @. dρq1 = beta * dρq1 - alpha * hdiv(cuvw * ρq1) + alpha * ystar.ρq1
    @. dρq1 -= alpha * vdivf2c(Ic2f(ρ) * third_order_upwind_c2f.(w, ρq1 ./ ρ))
    @. dρq1 -= alpha * vdivf2c(Ic2f(uₕ * ρq1))

    Spaces.weighted_dss!(dydt.ρ)
    Spaces.weighted_dss!(dydt.ρq1)

    return dydt
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
T = 86400.0 * 1.0
dt = 30.0 * 60.0

# set up 3D domain
hv_center_space, hv_face_space = sphere_3D()

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

p(z) = p₀ * exp(-z / H)
ρ_ref(z) = p(z) / R_d / T₀

y0 = map(coords) do coord
    z = coord.z

    z₀ = 0.5 * (z₂ + z₁)
    if z > z₁ && z < z₂
        q1 = 0.5 * (1.0 + cos(2pi * (z - z₀) / (z₂ - z₁)))
    else
        q1 = 0.0
    end

    ρq1 = ρ_ref(z) * q1

    return (ρ = ρ_ref(z), ρq1 = ρq1)
end

# IC FieldVector
y0 = Fields.FieldVector(ρ = y0.ρ, ρq1 = y0.ρq1)

ystar = copy(y0)
parameters = (;
    coords = coords,
    face_coords = face_coords,
    τ = τ,
    u₀ = u₀,
    w₀ = w₀,
    K = K,
    ρ₀ = ρ₀,
    ystar = ystar,
)

# Set up the initial flow
rhs!(ystar, y0, parameters, 0.0, dt, 1)

# run!
prob = ODEProblem(IncrementingODEFunction(rhs!), copy(y0), (0.0, T), parameters)
sol = solve(
    prob,
    SSPRK33ShuOsher(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

q1_error =
    norm(sol.u[end].ρq1 ./ ρ_ref.(coords.z) .- y0.ρq1 ./ ρ_ref.(coords.z)) /
    norm(y0.ρq1 ./ ρ_ref.(coords.z))
initial_mass = sum(y0.ρq1 ./ ρ_ref.(coords.z))
mass = sum(sol.u[end].ρq1 ./ ρ_ref.(coords.z))
rel_mass_err = norm((mass - initial_mass) / initial_mass)
@test q1_error ≈ 0.0 atol = 0.35
@test rel_mass_err ≈ 0.0 atol = 2.5e-4

Plots.png(
    Plots.plot(
        sol.u[trunc(Int, end / 2)].ρq1 ./ ρ_ref.(coords.z),
        level = 11,
        clim = (-0.1, 3.5),
    ),
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
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)

    # define variables
    nc_rho = defVar(nc, "rho", Float64, hv_center_space, ("time",))
    nc_q1 = defVar(nc, "q1", Float64, hv_center_space, ("time",))

    # write data to netcdf file
    for i in 1:length(sol.u)
        nc_time[i] = sol.t[i]

        # write fields to file
        nc_rho[:, i] = sol.u[i].ρ
        nc_q1[:, i] = sol.u[i].ρq1 ./ sol.u[i].ρ
    end
end

# write out our cubed sphere mesh
meshfile_cc = remap_tmpdir * "mesh_cubedsphere.g"
write_exodus(meshfile_cc, hv_center_space.horizontal_space.topology)

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
    in_np = Spaces.Quadratures.degrees_of_freedom(
        Spaces.quadrature_style(hv_center_space),
    ),
)

# apply remap
datafile_rll = remap_tmpdir * "data_rll.nc"
apply_remap(datafile_rll, datafile_cc, weightfile, ["rho", "q1"])

# load remapped data and create statistics for plots
nt = NCDataset(datafile_rll, "r") do nc
    lat = nc["lat"][:]
    z = nc["z"][:]
    time = nc["time"][:]
    q1 = nc["q1"][:]
    (; lat, z, time, q1)
end
(; lat, z, time, q1) = nt

# calculate zonal average for plotting
calc_zonalave(x) = dropdims(mean(x, dims = 1), dims = 1)

q1_zonalave = calc_zonalave(q1)

times = 0.0:dt:T
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
