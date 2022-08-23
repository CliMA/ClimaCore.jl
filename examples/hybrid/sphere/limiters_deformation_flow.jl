using Test
using LinearAlgebra
using UnPack

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
    Operators,
    Limiters
import ClimaCore.Utilities: half

using OrdinaryDiffEq: ODEProblem, solve
using DiffEqBase
using ClimaTimeSteppers

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

import ClimaCore.Limiters: apply_limit_slab!
using ClimaCore: DataLayouts
function apply_limit_slab!(
    slab_ρq::DataLayouts.IJF{<:Any, Nij},
    slab_ρ::DataLayouts.IJF{<:Any, Nij},
    slab_WJ::DataLayouts.IJF{<:Any, Nij},
    slab_q_bounds::DataLayouts.IF{<:Any, 2},
) where {Nij}

    Nf = DataLayouts.ncomponents(slab_ρq)

    array_ρq = parent(slab_ρq)
    array_ρ = parent(slab_ρ)
    array_w = parent(slab_WJ)
    array_q_bounds = parent(slab_q_bounds)

    rtol = eps(eltype(array_ρq)) # sqrt(eps(eltype(array_ρq)))
    maxiter = Nij * Nij

    # 1) compute ρ_tot
    ρ_tot = zero(eltype(array_ρ))
    for j in 1:Nij, i in 1:Nij
        ρ_tot += array_ρ[i, j, 1] * array_w[i, j, 1]
    end

    @assert ρ_tot > 0

    for f in 1:Nf
        q_min = array_q_bounds[1, f]
        q_max = array_q_bounds[2, f]

        # 2) compute ∫ρq
        ρq_tot = zero(eltype(array_ρq))
        for j in 1:Nij, i in 1:Nij
            ρq_tot += array_ρq[i, j, f] * array_w[i, j, 1]
        end

        # 3) set bounds
        q_avg = ρq_tot / ρ_tot
        q_min = min(q_min, q_avg)
        q_max = max(q_max, q_avg)

        # 3) compute total mass change
        for iter in 1:maxiter
            Δρq = zero(eltype(array_ρq))
            for j in 1:Nij, i in 1:Nij
                ρ = array_ρ[i, j, 1]
                ρq = array_ρq[i, j, f]
                ρq_max = ρ * q_max
                ρq_min = ρ * q_min
                w = array_w[i, j]
                if ρq > ρq_max
                    Δρq += (ρq - ρq_max) * w
                    array_ρq[i, j, f] = ρq_max
                elseif ρq < ρq_min
                    Δρq += (ρq - ρq_min) * w
                    array_ρq[i, j, f] = ρq_min
                end
            end

            if abs(Δρq) < rtol * ρq_tot
                break
            end

            if Δρq > 0 # add mass
                # compute total density
                Δρ = zero(eltype(array_ρ))
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq < ρ * q_max
                        Δρ += ρ * w
                    end
                end
                Δq = Δρq / Δρ # compute average ratio change
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq < ρ * q_max
                        array_ρq[i, j, f] += ρ * Δq
                    end
                end
            else # remove mass
                Δρ = zero(eltype(array_ρ))
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq > ρ * q_min
                        Δρ += ρ * w
                    end
                end
                Δq = Δρq / Δρ
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq > ρ * q_min
                        array_ρq[i, j, f] += ρ * Δq
                    end
                end
            end
        end
    end
end

# 3D deformation flow (DCMIP 2012 Test 1-1)
# Reference: http://www-personal.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf, Section 1.1

const FT = Float64

const R = FT(6.37122e6)        # radius
const grav = FT(9.8)           # gravitational constant
const R_d = FT(287.058)        # R dry (gas constant / mol mass dry air)
const z_top = FT(1.2e4)        # height position of the model top
const p_top = FT(25494.4)      # pressure at the model top
const T_0 = FT(300)            # isothermal atmospheric temperature
const H = R_d * T_0 / grav     # scale height
const p_0 = FT(1.0e5)          # reference pressure
const τ = FT(1036800.0)        # period of motion
const ω_0 = FT(23000) * FT(pi) / τ # maxium of the vertical pressure velocity
const b = FT(0.2)              # normalized pressure depth of divergent layer
const λ_c1 = FT(150.0)         # initial longitude of first tracer
const λ_c2 = FT(210.0)         # initial longitude of second tracer
const ϕ_c = FT(0.0)            # initial latitude of tracers
const centers = (
    Geometry.LatLongZPoint(ϕ_c, λ_c1, FT(0.0)),
    Geometry.LatLongZPoint(ϕ_c, λ_c2, FT(0.0)),
)
const z_c = FT(5.0e3)          # initial altitude of tracers
const R_t = R / 2              # horizontal half-width of tracers
const Z_t = FT(1000.0)         # vertical half-width of tracers
const D₄ = FT(1.0e16)          # hyperviscosity coefficient

lim_flag = true                # limiters flag
T = 86400 * FT(12.0)           # simulation times in seconds (12 days)
# dt = FT(60.0) * FT(60.0)     # time step in seconds (20 minutes)
zelems = 36
helems = 4

# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dirname = "limiters_deformation_flow"

if lim_flag == false
    dirname = "$(dirname)_no_lim"
end
if D₄ == 0
    dirname = "$(dirname)_D0"
end

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

# set up function space
function sphere_3D(
    R = FT(6.37122e6),
    zlim = (0, FT(12.0e3));
    helem = 4,
    zelem = 36,
    npoly = 4,
)
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
    return (horzspace, hv_center_space, hv_face_space)
end

# set up 3D domain
horzspace, hv_center_space, hv_face_space =
    sphere_3D(helem = helems, zelem = zelems)
global_geom = horzspace.global_geometry
topology = horzspace.topology

# Extract coordinates
center_coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

# Initialize pressure and density
p(z) = p_0 * exp(-z / H)
ρ_ref(z) = p(z) / R_d / T_0

tracers = map(center_coords) do coord
    z = coord.z
    zd = z - z_c
    λ = coord.long
    ϕ = coord.lat
    rd = Vector{FT}(undef, 2)
    # great circle distances
    for i in 1:2
        rd[i] = Geometry.great_circle_distance(coord, centers[i], global_geom)
    end
    # scaled distance functions
    d = Vector{FT}(undef, 2)
    for i in 1:2
        d[i] = min(1, (rd[i] / R_t)^2 + (zd / Z_t)^2)
    end
    q1 = FT(0.5) * (1 + cos(FT(pi) * d[1])) + FT(0.5) * (1 + cos(FT(pi) * d[2]))
    q2 = FT(0.9) - FT(0.8) * q1^2
    q3 = FT(0.0)
    if d[1] < FT(0.5) || d[2] < FT(0.5)
        q3 = FT(1.0)
    else
        q3 = FT(0.1)
    end
    if (z > z_c) && (ϕ > ϕ_c - rad2deg(FT(1) / 8)) && (ϕ < ϕ_c + rad2deg(FT(1) / 8))
        q3 = FT(0.1)
    end
    q4 = 1 - FT(3) / 10 * (q1 + q2 + q3)
    q5 = 1

    ρq1 = ρ_ref(z) * q1
    ρq2 = ρ_ref(z) * q2
    ρq3 = ρ_ref(z) * q3
    ρq4 = ρ_ref(z) * q4
    ρq5 = ρ_ref(z) * q5

    return (ρq1 = ρq1, ρq2 = ρq2, ρq3 = ρq3, ρq4 = ρq4, ρq5 = ρq5)
end

# Set initial state
y0 = Fields.FieldVector(ρ = ρ_ref.(center_coords.z), tracers = tracers)

function rhs!(dydt, y, parameters, t, alpha, beta)

    # Set up operators
    # Spectral horizontal operators
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    # Vertical staggered FD operators
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
    )
    third_order_upwind_c2f = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )

    # Define flow
    τ = parameters.τ
    center_coords = parameters.center_coords
    face_coords = parameters.face_coords

    ϕ = center_coords.lat
    λ = center_coords.long
    zc = center_coords.z
    zf = face_coords.z
    λp = λ .- 360 * t / τ
    k = 10 * R / τ

    ϕf = face_coords.lat
    λf = face_coords.long
    λpf = λf .- 360 * t / τ

    sp =
        @. 1 + exp((p_top - p_0) / b / p_top) - exp((p(zf) - p_0) / b / p_top) -
           exp((p_top - p(zf)) / b / p_top)
    ua = @. k * sind(λp)^2 * sind(2 * ϕ) * cos(FT(pi) * t / τ) +
       2 * FT(pi) * R / τ * cosd(ϕ)
    ud = @. ω_0 * R / b / p_top *
       cosd(λp) *
       cosd(ϕ)^2 *
       cos(2 * FT(pi) * t / τ) *
       (-exp((p(zc) - p_0) / b / p_top) + exp((p_top - p(zc)) / b / p_top))
    uu = @. ua + ud
    uv = @. k * sind(2 * λp) * cosd(ϕ) * cos(FT(pi) * t / τ)
    ω = @. ω_0 * sind(λpf) * cosd(ϕf) * cos(2 * FT(pi) * t / τ) * sp
    uw = @. -ω / ρ_ref(zf) / grav

    uₕ = Geometry.Covariant12Vector.(Geometry.UVVector.(uu, uv))
    w = Geometry.Covariant3Vector.(Geometry.WVector.(uw))

    # Compute velocity by interpolating faces to centers
    cw = If2c.(w)
    cuvw = Geometry.Covariant123Vector.(uₕ) .+ Geometry.Covariant123Vector.(cw)

    # Create convenient aliases:
    ρ = y.ρ
    # packed version of tracers tuple
    ρq = y.tracers
    # unpacked version of tracers
    @unpack ρq1, ρq2, ρq3, ρq4, ρq5 = y.tracers

    dρ = dydt.ρ
    # packed version of tracers tuple
    dρq = dydt.tracers
    # unpacked version of tracers
    dρq1 = dydt.tracers.ρq1
    dρq2 = dydt.tracers.ρq2
    dρq3 = dydt.tracers.ρq3
    dρq4 = dydt.tracers.ρq4
    dρq5 = dydt.tracers.ρq5

    # Define vertical fluxes:
    # - Density vertical flux
    vert_flux_wρ = vdivf2c.(w .* Ic2f.(ρ))
    # - Tracers vertical flux
    vert_flux_wρq1 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq1 ./ ρ),)
    vert_flux_wρq2 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq2 ./ ρ),)
    vert_flux_wρq3 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq3 ./ ρ),)
    vert_flux_wρq4 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq4 ./ ρ),)
    vert_flux_wρq5 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq5 ./ ρ),)
    # NOTE: When Upwind3rdOrderBiasedProductC2F is extended to work on tuples of tracers, we can use one call for all the vertical tracer fluxes, i.e.:
    # vert_flux_wρq = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq ./ ρ),)

    ## NOTE: With limiters, the order of the operations 1)-5) below matters!

    # 1) Horizontal Transport:
    # 1.1) Horizontal part of continuity equation:
    @. dρ = beta * dρ - alpha * hdiv(ρ * cuvw)

    # 1.2) Horizontal advection of tracers equations:
    @. dρq1 = beta * dρq1 - alpha * hdiv(ρq1 * cuvw)
    @. dρq2 = beta * dρq2 - alpha * hdiv(ρq2 * cuvw)
    @. dρq3 = beta * dρq3 - alpha * hdiv(ρq3 * cuvw)
    @. dρq4 = beta * dρq4 - alpha * hdiv(ρq4 * cuvw)
    @. dρq5 = beta * dρq5 - alpha * hdiv(ρq5 * cuvw)

    # 2) Apply the limiters:
    if parameters.lim_flag
        # First compute the min_q/max_q at the current time step
        Limiters.compute_bounds!(parameters.limiter1, ρq1, ρ)
        Limiters.compute_bounds!(parameters.limiter2, ρq2, ρ)
        Limiters.compute_bounds!(parameters.limiter3, ρq3, ρ)
        Limiters.compute_bounds!(parameters.limiter4, ρq4, ρ)
        Limiters.compute_bounds!(parameters.limiter5, ρq5, ρ)
        # Then apply the limiters
        Limiters.apply_limiter!(dρq1, dρ, parameters.limiter1)
        Limiters.apply_limiter!(dρq2, dρ, parameters.limiter2)
        Limiters.apply_limiter!(dρq3, dρ, parameters.limiter3)
        Limiters.apply_limiter!(dρq4, dρ, parameters.limiter4)
        Limiters.apply_limiter!(dρq5, dρ, parameters.limiter5)
    end

    # 3) Add Hyperdiffusion to the horizontal tracers equation (using weak div)
    # Set up working variable needed for hyperdiffusion
    ystar = similar(y)

    @. ystar.tracers.ρq1 = hwdiv(hgrad(ρq1 / ρ))
    Spaces.weighted_dss!(ystar.tracers.ρq1)
    @. ystar.tracers.ρq1 = -D₄ * hwdiv(ρ * hgrad(ystar.tracers.ρq1))
    @. dρq1 += alpha * ystar.tracers.ρq1

    @. ystar.tracers.ρq2 = hwdiv(hgrad(ρq2 / ρ))
    Spaces.weighted_dss!(ystar.tracers.ρq2)
    @. ystar.tracers.ρq2 = -D₄ * hwdiv(ρ * hgrad(ystar.tracers.ρq2))
    @. dρq2 += alpha * ystar.tracers.ρq2

    @. ystar.tracers.ρq3 = hwdiv(hgrad(ρq3 / ρ))
    Spaces.weighted_dss!(ystar.tracers.ρq3)
    @. ystar.tracers.ρq3 = -D₄ * hwdiv(ρ * hgrad(ystar.tracers.ρq3))
    @. dρq3 += alpha * ystar.tracers.ρq3

    @. ystar.tracers.ρq4 = hwdiv(hgrad(ρq4 / ρ))
    Spaces.weighted_dss!(ystar.tracers.ρq4)
    @. ystar.tracers.ρq4 = -D₄ * hwdiv(ρ * hgrad(ystar.tracers.ρq4))
    @. dρq4 += alpha * ystar.tracers.ρq4

    @. ystar.tracers.ρq5 = hwdiv(hgrad(ρq5 / ρ))
    Spaces.weighted_dss!(ystar.tracers.ρq5)
    @. ystar.tracers.ρq5 = -D₄ * hwdiv(ρ * hgrad(ystar.tracers.ρq5))
    @. dρq5 += alpha * ystar.tracers.ρq5

    # 4) Vertical Transport:
    # 4.1) Vertical part of continuity equation
    @. dρ -= alpha * vdivf2c.(Ic2f.(ρ .* uₕ))
    @. dρ -= alpha * vert_flux_wρ

    # 4.2) Vertical advection of tracers equations
    @. dρq1 -= alpha * vdivf2c(Ic2f(ρq1 * uₕ))
    @. dρq1 -= alpha * vert_flux_wρq1

    @. dρq2 -= alpha * vdivf2c(Ic2f(ρq2 * uₕ))
    @. dρq2 -= alpha * vert_flux_wρq2

    @. dρq3 -= alpha * vdivf2c(Ic2f(ρq3 * uₕ))
    @. dρq3 -= alpha * vert_flux_wρq3

    @. dρq4 -= alpha * vdivf2c(Ic2f(ρq4 * uₕ))
    @. dρq4 -= alpha * vert_flux_wρq4

    @. dρq5 -= alpha * vdivf2c(Ic2f(ρq5 * uₕ))
    @. dρq5 -= alpha * vert_flux_wρq5

    # 5) DSS:
    Spaces.weighted_dss!(dρ)
    Spaces.weighted_dss!(dρq)
    return dydt
end

# Set up vectors and parameters needed for the RHS function
ystar = copy(y0)

parameters = (
    lim_flag = lim_flag,
    horzspace = horzspace,
    limiter1 = Limiters.QuasiMonotoneLimiter(ystar.tracers.ρq1, ystar.ρ),
    limiter2 = Limiters.QuasiMonotoneLimiter(ystar.tracers.ρq2, ystar.ρ),
    limiter3 = Limiters.QuasiMonotoneLimiter(ystar.tracers.ρq3, ystar.ρ),
    limiter4 = Limiters.QuasiMonotoneLimiter(ystar.tracers.ρq4, ystar.ρ),
    limiter5 = Limiters.QuasiMonotoneLimiter(ystar.tracers.ρq5, ystar.ρ),
    τ = τ,
    center_coords = center_coords,
    face_coords = face_coords,
)

# Set up the RHS function
# rhs!(ystar, y0, parameters, FT(0.0), dt, 1)

# Solve the ODE
M_inits = [
    sum(y0.ρ),
    sum(y0.tracers.ρq1),
    sum(y0.tracers.ρq2),
    sum(y0.tracers.ρq3),
    sum(y0.tracers.ρq4),
    sum(y0.tracers.ρq5),
]

dts = FT[1.075, 1.05, 1.025, @.(10^(-(0:2) / 4))...] .* (60 * 60)
M_errs = similar(dts, length(dts), 6)
for (index, dt) in enumerate(dts)
    @info dt
    prob = ODEProblem(IncrementingODEFunction(rhs!), copy(y0), (FT(0.0), T), parameters)
    sol = @timev solve(
        prob,
        SSPRK33ShuOsher(),
        dt = dt,
        saveat = T,
        progress = true,
        adaptive = false,
    )
    y = sol.u[end]
    M_errs[index, 1] = abs((sum(y.ρ) - M_inits[1]) / M_inits[1])
    M_errs[index, 2] = abs((sum(y.tracers.ρq1) - M_inits[2]) / M_inits[2])
    M_errs[index, 3] = abs((sum(y.tracers.ρq2) - M_inits[3]) / M_inits[3])
    M_errs[index, 4] = abs((sum(y.tracers.ρq3) - M_inits[4]) / M_inits[4])
    M_errs[index, 5] = abs((sum(y.tracers.ρq4) - M_inits[5]) / M_inits[5])
    M_errs[index, 6] = abs((sum(y.tracers.ρq5) - M_inits[6]) / M_inits[6])
    @info M_errs[index, :]
end
str1 = lim_flag ? "with" : "without"
str2 = lim_flag ? "_lim_" : "_no_lim_"
plt = Plots.plot(
    dts[2:end],
    M_errs[2:end, :];
    line = (3, :solid),
    scale = :log10,
    legend = :topright,
    title = "Conservation Error for Deformation Flow $str1 Limiters ($FT)",
    xlabel = "dt [s]",
    ylabel = "Relative Change: |M_end - M_start| / M_start",
    label = [
        "M = sum(Y.ρ)";;
        "M = sum(Y.tracers.ρq1)";;
        "M = sum(Y.tracers.ρq2)";;
        "M = sum(Y.tracers.ρq3)";;
        "M = sum(Y.tracers.ρq4)";;
        "M = sum(Y.tracers.ρq5)";;
    ],
    titlefontsize = 10,
    guidefontsize = 9,
    background_color_legend = RGBA(1, 1, 1, 0.7),
)
Plots.plot!(
    [dts[end], dts[1]],
    [M_errs[end, 1], M_errs[end, 1] * dts[end] / dts[1]];
    label = "Expected Conservation Error",
    line = (2, :dot, :red)
)
if lim_flag
    Plots.vline!([1.085 * 60 * 60]; label = "", line = (1, :dash, :black))
end
Plots.png(plt, joinpath(path, "new_conservation$str2$FT.png"))

# q1_error =
#     norm(
#         sol.u[end].tracers.ρq1 ./ ρ_ref.(center_coords.z) .-
#         y0.tracers.ρq1 ./ ρ_ref.(center_coords.z),
#     ) / norm(y0.tracers.ρq1 ./ ρ_ref.(center_coords.z))
# @test q1_error ≈ 0.0 atol = 0.75

# q2_error =
#     norm(
#         sol.u[end].tracers.ρq2 ./ ρ_ref.(center_coords.z) .-
#         y0.tracers.ρq2 ./ ρ_ref.(center_coords.z),
#     ) / norm(y0.tracers.ρq2 ./ ρ_ref.(center_coords.z))
# @test q2_error ≈ 0.0 atol = 0.034

# q3_error =
#     norm(
#         sol.u[end].tracers.ρq3 ./ ρ_ref.(center_coords.z) .-
#         y0.tracers.ρq3 ./ ρ_ref.(center_coords.z),
#     ) / norm(y0.tracers.ρq3 ./ ρ_ref.(center_coords.z))
# @test q3_error ≈ 0.0 atol = 0.41

# q4_error =
#     norm(
#         sol.u[end].tracers.ρq4 ./ ρ_ref.(center_coords.z) .-
#         y0.tracers.ρq4 ./ ρ_ref.(center_coords.z),
#     ) / norm(y0.tracers.ρq4 ./ ρ_ref.(center_coords.z))
# @test q4_error ≈ 0.0 atol = 0.03

# q5_error =
#     norm(
#         sol.u[end].tracers.ρq5 ./ ρ_ref.(center_coords.z) .-
#         y0.tracers.ρq5 ./ ρ_ref.(center_coords.z),
#     ) / norm(y0.tracers.ρq5 ./ ρ_ref.(center_coords.z))
# @test q5_error ≈ 0.0 atol = 0.007

# Plots.png(
#     Plots.plot(
#         sol.u[trunc(Int, end / 2)].tracers.ρq3 ./ ρ_ref.(center_coords.z),
#         level = 15,
#         clim = (-1, 1),
#     ),
#     joinpath(path, "q3_6day.png"),
# )

# Plots.png(
#     Plots.plot(
#         sol.u[end].tracers.ρq3 ./ ρ_ref.(center_coords.z),
#         level = 15,
#         clim = (-1, 1),
#     ),
#     joinpath(path, "q3_12day.png"),
# )
