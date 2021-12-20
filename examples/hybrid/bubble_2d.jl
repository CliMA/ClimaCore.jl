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
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# set up function space
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 40,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

# set up rhs!
hv_center_space, hv_face_space = hvspace_2D((-500, 500), (0, 1000))
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

    return (ρ = ρ, ρθ = ρθ, ρuₕ = ρ * Geometry.UVector(0.0))
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coords) do coord
    bubble = init_dry_rising_bubble_2d(coord.x, coord.z)
    bubble
end

ρw = map(face_coords) do coord
    Geometry.WVector(0.0)
end;

Y = Fields.FieldVector(Yc = Yc, ρw = ρw)

function energy(Yc, ρu, z)
    ρ = Yc.ρ
    ρθ = Yc.ρθ
    u = ρu / ρ
    kinetic = ρ * norm(u)^2 / 2
    potential = z * grav * ρ
    internal = C_v * pressure(ρθ) / R_d
    return kinetic + potential + internal
end
function combine_momentum(ρuₕ, ρw)
    Geometry.transform(Geometry.UWAxis(), ρuₕ) +
    Geometry.transform(Geometry.UWAxis(), ρw)
end
function center_momentum(Y)
    If2c = Operators.InterpolateF2C()
    combine_momentum.(Y.Yc.ρuₕ, If2c.(Y.ρw))
end
function total_energy(Y)
    ρ = Y.Yc.ρ
    ρu = center_momentum(Y)
    ρθ = Y.Yc.ρθ
    z = Fields.coordinate_field(axes(ρ)).z
    sum(energy.(Yc, ρu, z))
end

energy_0 = total_energy(Y)
mass_0 = sum(Yc.ρ) # Computes ∫ρ∂Ω such that quadrature weighting is accounted for.

function rhs!(dY, Y, _, t)
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    # spectral horizontal operators
    hdiv = Operators.Divergence()
    hgrad = Operators.Gradient()
    hwdiv = Operators.WeakDivergence()
    hwgrad = Operators.WeakGradient()

    # vertical FD operators with BC's
    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(0.0)),
        top = Operators.SetDivergence(Geometry.WVector(0.0)),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0),
        ),
        top = Operators.SetValue(Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0)),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    ∂f = Operators.GradientC2F()
    ∂c = Operators.GradientF2C()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )

    fcc = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    uₕ = @. Yc.ρuₕ / Yc.ρ
    w = @. ρw / If(Yc.ρ)
    wc = @. Ic(ρw) / Yc.ρ
    p = @. pressure(Yc.ρθ)
    θ = @. Yc.ρθ / Yc.ρ
    Yfρ = @. If(Yc.ρ)

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    @. dYc.ρθ = hdiv(hgrad(θ))
    @. dYc.ρuₕ = hdiv(hgrad(uₕ))
    @. dρw = hdiv(hgrad(w))
    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)

    κ₄ = 100.0 # m^4/s
    @. dYc.ρθ = -κ₄ * hdiv(Yc.ρ * hgrad(dYc.ρθ))
    @. dYc.ρuₕ = -κ₄ * hdiv(Yc.ρ * hgrad(dYc.ρuₕ))
    @. dρw = -κ₄ * hdiv(Yfρ * hgrad(dρw))

    # density
    @. dYc.ρ = -∂(ρw)
    @. dYc.ρ -= hdiv(Yc.ρuₕ)

    # potential temperature
    @. dYc.ρθ += -(∂(ρw * If(Yc.ρθ / Yc.ρ)))
    @. dYc.ρθ -= hdiv(uₕ * Yc.ρθ)

    # horizontal momentum
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.UAxis(), Geometry.UAxis()),
            @SMatrix [1.0]
        ),
    )
    @. dYc.ρuₕ += -uvdivf2c(ρw ⊗ If(uₕ))
    @. dYc.ρuₕ -= hdiv(Yc.ρuₕ ⊗ uₕ + p * Ih)

    # vertical momentum
    @. dρw +=
        B(
            Geometry.transform(
                Geometry.WAxis(),
                -(∂f(p)) - If(Yc.ρ) * ∂f(Φ(coords.z)),
            ) - vvdivc2f(Ic(ρw ⊗ w)),
        ) + fcf(wc, ρw)
    uₕf = @. If(Yc.ρuₕ / Yc.ρ) # requires boundary conditions
    @. dρw -= hdiv(uₕf ⊗ ρw)

    ### UPWIND FLUX CORRECTION
    upwind_correction = true
    if upwind_correction
        @. dYc.ρ += fcc(w, Yc.ρ)
        @. dYc.ρθ += fcc(w, Yc.ρθ)
        @. dYc.ρuₕ += fcc(w, Yc.ρuₕ)
        @. dρw += fcf(wc, ρw)
    end

    ### DIFFUSION
    κ₂ = 0.0 # m^2/s
    #  1a) horizontal div of horizontal grad of horiz momentun
    @. dYc.ρuₕ += hdiv(κ₂ * (Yc.ρ * hgrad(Yc.ρuₕ / Yc.ρ)))
    #  1b) vertical div of vertical grad of horiz momentun
    @. dYc.ρuₕ += uvdivf2c(κ₂ * (Yfρ * ∂f(Yc.ρuₕ / Yc.ρ)))

    #  1c) horizontal div of horizontal grad of vert momentum
    @. dρw += hdiv(κ₂ * (Yfρ * hgrad(ρw / Yfρ)))
    #  1d) vertical div of vertical grad of vert momentun
    @. dρw += vvdivc2f(κ₂ * (Yc.ρ * ∂c(ρw / Yfρ)))

    #  2a) horizontal div of horizontal grad of potential temperature
    @. dYc.ρθ += hdiv(κ₂ * (Yc.ρ * hgrad(Yc.ρθ / Yc.ρ)))
    #  2b) vertical div of vertial grad of potential temperature
    @. dYc.ρθ += ∂(κ₂ * (Yfρ * ∂f(Yc.ρθ / Yc.ρ)))

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

dYdt = similar(Y);
rhs!(dYdt, Y, nothing, 0.0);


# run!
using OrdinaryDiffEq
Δt = 0.025
prob = ODEProblem(rhs!, Y, (0.0, 50.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "bubble_2d"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# post-processing
import Plots
anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρθ ./ u.Yc.ρ, clim = (300.0, 300.8))
end
Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    Plots.plot(If2c.(u.ρw) ./ u.Yc.ρ, clim = (-2, 2))
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρuₕ ./ u.Yc.ρ, clim = (-2, 2))
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

Es = [total_energy(u) for u in sol.u]
Mass = [sum(u.Yc.ρ) for u in sol.u]

Plots.png(
    Plots.plot((Es .- energy_0) ./ energy_0),
    joinpath(path, "energy.png"),
)
Plots.png(Plots.plot((Mass .- mass_0) ./ mass_0), joinpath(path, "mass.png"))
