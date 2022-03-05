push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack
using OrdinaryDiffEq

import ClimaCore:
        ClimaCore,
        slab,
        Spaces,
        Domains,
        Meshes,
        Geometry,
        Topologies,
        Fields,
        Operators
using ClimaCore.Geometry

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    xelem = 20,
    velem = 20,
    npoly = 1
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
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true)
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = xelem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space = 
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

FT = Float64
const atm_T_ini = FT(270.0)
const MSLP = FT(1e5)
const grav = FT(9.8)
const R_d = FT(287.058)
const γ = FT(1.4)
const C_p = FT(R_d * γ / (γ - 1))
const C_v = FT(R_d / (γ - 1))
const R_m = R_d

function pressure(ρθ)
    if ρθ >= 0
        return MSLP * (R_d * ρθ / MSLP)^γ
    else
        return NaN
    end
end

Φ(z) = grav * z

abstract type BCtag end
struct ZeroFlux <: BCtag end

bc_divF2C_bottom!(::ZeroFlux, dY, Y, p, t) = Operators.SetValue(Geometry.WVector(0.0))
bc_divF2C_top!(::ZeroFlux, dY, Y, p, t) = Operators.SetValue(Geometry.WVector(0.0))

struct ZeroFieldFlux <: BCtag end
function bc_divF2C_bottom!(::ZeroFieldFlux, dY, Y, p, t)
    space = axes(Y.Yc.ρθ).horizontal_space
    FT = Spaces.undertype(space)
    zeroflux = Fields.zeros(FT, space)
    return Operators.SetValue(Geometry.WVector.(zeroflux))
end

function init_bubble_2d(x, z)
    cp_d = C_p
    cv_d = C_v
    p₀ = MSLP
    g = grav
    γ = cp_d / cv_d
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = atm_T_ini
    θ_c = 0.5

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - g * z / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p₀ * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    ρθ = ρ * θ # potential temperature density

    return (ρ = ρ, ρθ = ρθ, ρuₕ = ρ * Geometry.UVector(0.0))
end

function rhs!(dY, Y, params, t)
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    center_coords = Fields.coordinate_field(params.domain.hv_center_space)
    face_coords = Fields.coordinate_field(params.domain.hv_face_space)

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
        bottom = Operators.SetValue(Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0)),
    )
    If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
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

    fcc = Operators.FluxCorrectionC2C(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
    fcf = Operators.FluxCorrectionF2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
    ∇_z_ρθ = Operators.DivergenceF2C(
        bottom = bc_divF2C_bottom!(params.bc.ρθ.bottom, dY, Y, params, t),
        top = bc_divF2C_top!(params.bc.ρθ.top, dY, Y, params, t),
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

    κ₄ = 0.0 # m^4/s
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
    Ih = Ref(Geometry.Axis2Tensor((Geometry.UAxis(), Geometry.UAxis()), @SMatrix [1.0]),)
    @. dYc.ρuₕ += -uvdivf2c(ρw ⊗ If(uₕ))
    @. dYc.ρuₕ -= hdiv(Yc.ρuₕ ⊗ uₕ + p * Ih)

    # vertical momentum
    @. dρw +=
        B(Geometry.transform(Geometry.WAxis(), -(∂f(p)) - If(Yc.ρ) * ∂f(Φ(center_coords.z))) - vvdivc2f(Ic(ρw ⊗ w)))
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
    κ₂ = 5.0 # m^2/s
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
    @. dYc.ρθ += ∇_z_ρθ(κ₂ * (Yfρ * ∂f(Yc.ρθ / Yc.ρ)))

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

# init simulation
function init(; xmin = -500, xmax = 500, zmin = 0, zmax = 1000, npoly = 1, xelem = 20, velem = 20, bc = nothing)

    # construct domain spaces
    hv_center_space, hv_face_space = hvspace_2D((xmin, xmax), (zmin, zmax), xelem, velem, npoly) # [m]
    center_coords = Fields.coordinate_field(hv_center_space)
    face_coords = Fields.coordinate_field(hv_face_space)
    domain = (hv_center_space = hv_center_space, hv_face_space = hv_face_space)

    # initialize prognostic variables
    Yc = map(center_coords) do coord
        sea_breeze = init_bubble_2d(coord.x, coord.z)
        sea_breeze
    end

    ρw = map(face_coords) do coord
        Geometry.WVector(0.0)
    end

    Y = Fields.FieldVector(Yc = Yc, ρw = ρw)

    # select boundary conditions
    if bc === nothing
        bc = (
            ρθ = (bottom = ZeroFlux(), top = ZeroFlux()),
            ρu = nothing, # for now BCs are hard coded, except for ρθ
        )
    end

    return Y, bc, domain
end

function run!(Y, bc, domain)
    dYdt = similar(Y)
    params = (aux_params = 0.0, T_sfc = 1.0, bc = bc, domain = domain)
    rhs!(dYdt, Y, params, 0.0)
    prob = ODEProblem(rhs!, Y, (0.0, 100.0), params)
    Δt = 0.025
    sol = solve(prob, SSPRK33(), dt = Δt, saveat = 1.0, progress = true, progress_message = (dt, u, params, t) -> t)
end

# Traditional ZeroFlux BC
sol1 = run!(init()...)

bc = (
    ρθ = (bottom = ZeroFieldFlux(), top = ZeroFlux()),
    ρu = nothing, # for now BCs are hard coded, except for ρθ
)

# ZeroFlux condition set as a Field
sol2 = run!(init(; bc = bc)...)

@test sol1.u == sol2.u
