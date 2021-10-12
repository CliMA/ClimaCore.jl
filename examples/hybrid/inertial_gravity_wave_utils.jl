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

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include("../implicit_solver_utils.jl")
include("../ordinary_diff_eq_bug_fixes.jl")

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
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(-0)..Geometry.YPoint{FT}(0),
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

Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)
Φ(z) = grav * z

function init_inertial_gravity_wave(x, z, is_large_domain)
    p_0 = MSLP
    g = grav
    cp_d = C_p
    x_c = 0.
    θ_0 = 300.
    Δθ = 0.01
    A = is_large_domain ? 100000. : 5000.
    H = 10000.
    NBr = 0.01
    S = NBr * NBr / g

    p_ref = p_0 * (1 - g / (cp_d * θ_0 * S) * (1 - exp(-S * z)))^(cp_d / R_d)
    θ = θ_0 * exp(z * S) + Δθ * sin(pi * z / H) / (1 + ((x - x_c) / A)^2)
    ρ = p_ref / ((p_ref / p_0)^(R_d / cp_d) * R_d * θ)
    ρθ = ρ * θ

    return (ρ = ρ, ρθ = ρθ, ρuₕ = ρ * Geometry.Cartesian1Vector(0.0))
end

function rhs!(dY, Y, p, t)
    @unpack Φf, uₕ, uₕf, P = p
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    # spectral horizontal operators
    hdiv = Operators.Divergence()

    # vertical FD operators with BC's
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
        ),
        top = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
        ),
    )
    If_bc = Operators.InterpolateC2F(
        bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )
    ∂f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )

    @. uₕ = Yc.ρuₕ / Yc.ρ
    @. uₕf = If_bc(uₕ)
    @. P = pressure(Yc.ρθ)

    # density
    @. dYc.ρ = -∂(ρw)
    @. dYc.ρ -= hdiv(Yc.ρuₕ)

    # potential temperature
    @. dYc.ρθ = -(∂(ρw * If(Yc.ρθ / Yc.ρ)))
    @. dYc.ρθ -= hdiv(uₕ * Yc.ρθ)

    # horizontal momentum
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()),
            @SMatrix [1.0]
        ),
    )
    @. dYc.ρuₕ = -hdiv(Yc.ρuₕ ⊗ uₕ + P * Ih)
    @. dYc.ρuₕ -= uvdivf2c(ρw ⊗ uₕf)

    # vertical momentum
    @. dρw = B(
        Geometry.transform(
            Geometry.Cartesian3Axis(),
            -(∂f(P)) - If(Yc.ρ) * Φf,
        ) - vvdivc2f(Ic(ρw ⊗ (ρw / If(Yc.ρ)))),
    )
    @. dρw -= hdiv(uₕf ⊗ ρw)

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

function rhs_vertical!(dY, Y, p, t)
    @unpack Φf, P = p
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    # vertical FD operators with BC's
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

    @. P = pressure(Yc.ρθ)

    # density
    @. dYc.ρ = -∂(ρw)

    # potential temperature
    @. dYc.ρθ = -(∂(ρw * If(Yc.ρθ / Yc.ρ)))

    # horizontal momentum
    parent(dYc.ρuₕ) .= 0.

    # vertical momentum
    @. dρw = B(
        Geometry.transform(
            Geometry.Cartesian3Axis(),
            -(∂f(P)) - If(Yc.ρ) * Φf,
        ),
    )
    return dY
end

#=
ρ(z, t), ρθ(z, t), ρw(z, t)
Δz(zc_n) = zf_{n+1} - zf_n
Δz(zf_n) = zc_n - zc_{n-1}
∂Φ/∂z(zf_n) = (Φ(zc_n) - Φ(zc_{n-1})) / Δz(zf_n) = grav
θ(zf_n, t_i) =
    if n == 1:
        = θ(zc_n, t_i) =
        = ρθ(zc_n, t_i) / ρ(zc_n, t_i)
    elseif n == N + 1:
        = θ(zc_{n-1}, t_i) =
        = ρθ(zc_{n-1}, t_i) / ρ(zc_{n-1}, t_i)
    else:
        = (θ(zc_{n-1}, t_i) + θ(zc_n, t_i)) / 2 =
        = (
            ρθ(zc_{n-1}, t_i) / ρ(zc_{n-1}, t_i) +
            ρθ(zc_n, t_i) / ρ(zc_n, t_i)
          ) / 2
P(zc_n, t_i) = MSLP * (R_d * ρθ(zc_n, t_i) / MSLP)^γ
∂P(zc_n, t_i)/∂ρθ(zc_n, t_i) = γ * R_d * (R_d * ρθ(zc_n, t_i) / MSLP)^(γ - 1) =
    = (γ - 1) * Π(zc_n, t_i)

∂/∂t ρ(zc_n, t_{i+1}) ≈ (ρ(zc_n, t_{i+1}) - ρ(zc_n, t_i)) / Δt =
    = -(ρw(zf_{n+1}, t_{i+1}) - ρw(zf_n, t_{i+1})) / Δz(zc_n) =
    = (ρw(zf_n, t_{i+1}) - ρw(zf_{n+1}, t_{i+1})) / Δz(zc_n)
    # ρw(zf_1, t_{i+1}) and ρw(zf_{N+1}, t_{i+1}) are assumed to be 0
∂/∂t ρθ(zc_n, t_{i+1}) ≈ (ρθ(zc_n, t_{i+1}) - ρθ(zc_n, t_i)) / Δt =
    = -(
        ρw(zf_{n+1}, t_{i+1}) * θ(zf_{n+1}, t_i) -
        ρw(zf_n, t_{i+1}) * θ(zf_n, t_i)
       ) / Δz(zc_n) =
    = (
        ρw(zf_n, t_{i+1}) * θ(zf_n, t_i) -
        ρw(zf_{n+1}, t_{i+1}) * θ(zf_{n+1}, t_i)
       ) / Δz(zc_n)
    # ρw(zf_1, t_{i+1}) and ρw(zf_{N+1}, t_{i+1}) are assumed to be 0
∂/∂t ρw(zf_n, t_{i+1}) ≈ (ρw(zf_n, t_{i+1}) - ρw(zf_n, t_i)) / Δt =
    if n == 1 or n == N + 1:
        = 0
    else:
        = -(P(zc_n, t_{i+1}) - P(zc_{n-1}, t_{i+1})) / Δz(zf_n) -
          (ρ(zc_n, t_{i+1}) + ρ(zc_{n+1}, t_{i+1})) / 2 * ∂Φ/∂z(zf_n) =
        = (P(zc_{n-1}, t_{i+1}) - P(zc_n, t_{i+1})) / Δz(zf_n) -
          (ρ(zc_n, t_{i+1}) + ρ(zc_{n+1}, t_{i+1})) / 2 * grav
=#

function rhs_horizontal!(dY, Y, p, t)
    @unpack uₕ, uₕf, P = p
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    # spectral horizontal operators
    hdiv = Operators.Divergence()

    # vertical FD operators with BC's
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
        ),
        top = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
        ),
    )
    If_bc = Operators.InterpolateC2F(
        bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )

    @. uₕ = Yc.ρuₕ / Yc.ρ
    @. uₕf = If_bc(uₕ)
    @. P = pressure(Yc.ρθ)

    # density
    @. dYc.ρ = -hdiv(Yc.ρuₕ)

    # potential temperature
    @. dYc.ρθ = -hdiv(uₕ * Yc.ρθ)

    # horizontal momentum
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()),
            @SMatrix [1.0]
        ),
    )
    @. dYc.ρuₕ = -hdiv(Yc.ρuₕ ⊗ uₕ + P * Ih)
    @. dYc.ρuₕ -= uvdivf2c(ρw ⊗ uₕf)

    # vertical momentum
    @. dρw = B(-vvdivc2f(Ic(ρw ⊗ (ρw / If(Yc.ρ)))))
    @. dρw -= hdiv(uₕf ⊗ ρw)

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

struct CustomWRepresentation{T,AT1,AT2,AT3,AT4}
    # grid information
    velem::Int
    helem::Int
    npoly::Int

    # whether this struct is used to compute Wfact_t or Wfact
    transform::Bool

    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::T

    # cache for the grid values used to compute the Jacobian
    Δz′::AT1
    Δzf′::AT1

    # cache for the variable values used to compute the Jacobian
    θf::AT2
    ∂P∂ρθ::AT2

    # nonzero blocks of the Jacobian (∂ρₜ/∂ρw, ∂ρθₜ/∂ρw, ∂ρwₜ/∂ρ, and ∂ρwₜ/∂ρθ)
    Jρ_ρw′::AT3
    Jρθ_ρw′::AT3
    Jρw_ρ′::AT3
    Jρw_ρθ′::AT3

    # cache for the Schur complement
    S::AT4
end

function CustomWRepresentation(
    velem,
    helem,
    npoly,
    coords,
    face_coords,
    transform;
    FT = Float64,
)
    N = velem
    M = helem * (npoly + 1)

    dtγ_ref = Ref(zero(FT))

    zf = reshape(parent(face_coords.z), N + 1, M)
    Δz′ = zf[2:N + 1, :] .- zf[1:N, :]
    zc = reshape(parent(coords.z), N , M)
    Δzf′ = zc[2:N, :] .- zc[1:N - 1, :]

    θf = Array{FT}(undef, N + 1)
    ∂P∂ρθ = Array{FT}(undef, N)

    Jρ_ρw′ = [GeneralBidiagonal(Array{FT}, true, N, N + 1) for _ in 1:M]
    Jρθ_ρw′ = [GeneralBidiagonal(Array{FT}, true, N, N + 1) for _ in 1:M]
    Jρw_ρ′ = [GeneralBidiagonal(Array{FT}, false, N + 1, N) for _ in 1:M]
    Jρw_ρθ′ = [GeneralBidiagonal(Array{FT}, false, N + 1, N) for _ in 1:M]

    S = Tridiagonal(
        Array{FT}(undef, N),
        Array{FT}(undef, N + 1),
        Array{FT}(undef, N),
    )

    CustomWRepresentation{
        typeof(dtγ_ref),
        typeof(Δz′),
        typeof(θf),
        typeof(Jρ_ρw′),
        typeof(S),
    }(
        velem,
        helem,
        npoly,
        transform,
        dtγ_ref,
        Δz′,
        Δzf′,
        θf,
        ∂P∂ρθ,
        Jρ_ρw′,
        Jρθ_ρw′,
        Jρw_ρ′,
        Jρw_ρθ′,
        S,
    )
end

import Base: similar
# We only use Wfact, but the Rosenbrock23 solver requires us to pass
# jac_prototype, then calls similar(jac_prototype) to obtain J and Wfact. This
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(cf::CustomWRepresentation{T,AT}) where {T, AT} = cf

# TODO: Automate construction of Jacobian from variables and rhs! specification.
#       For example, if upwinding is specified, construct that block of the
#       Jacobian differently. Also, check whether variables are conservative or
#       non-conservative, and construct different Jacobians for each.
function Wfact!(W, u, p, dtγ, t)
    @unpack velem, helem, npoly, dtγ_ref, Δz′, Δzf′, θf, ∂P∂ρθ, Jρ_ρw′, Jρθ_ρw′,
        Jρw_ρ′, Jρw_ρθ′ = W
    dtγ_ref[] = dtγ

    N = velem
    # TODO: Remove duplicate column computations.
    for i in 1:npoly + 1, h in 1:helem
        m = (h - 1) * (npoly + 1) + i

        Δz = reshape(view(Δz′, :, m), N)
        Δzf = reshape(view(Δzf′, :, m), N - 1)
        ρ = reshape(parent(Spaces.column(u.Yc.ρ, i, 1, h)), N)
        ρθ = reshape(parent(Spaces.column(u.Yc.ρθ, i, 1, h)), N)
        Jρ_ρw = Jρ_ρw′[m]
        Jρθ_ρw = Jρθ_ρw′[m]
        Jρw_ρ = Jρw_ρ′[m]
        Jρw_ρθ = Jρw_ρθ′[m]

        # Compute the variable values

        θf[1] = ρθ[1] / ρ[1]
        @views @. θf[2:N] = (ρθ[1:N - 1] + ρθ[2:N]) / (ρ[1:N - 1] + ρ[2:N])
        θf[N + 1] = ρθ[N] / ρ[N]

        @. ∂P∂ρθ = (γ - 1) * Π(ρθ)

        # Compute the nonzero blocks of the Jacobian
        
        @. Jρ_ρw.d = 1 / Δz
        @. Jρ_ρw.d2 = -1 / Δz
        # This is unnecessary when w = 0 at the boundary.
        # Jρ_ρw.d[1] = 0
        # Jρ_ρw.d2[N] = 0

        @. Jρθ_ρw.d = θf[1:N] / Δz
        @. Jρθ_ρw.d2 = -θf[2:N + 1] / Δz
        # This is unnecessary when w = 0 at the boundary.
        # Jρθ_ρw.d[1] = 0
        # Jρθ_ρw.d2[N] = 0

        Jρw_ρ.d[1] = 0
        @. Jρw_ρ.d[2:N] = -grav / 2
        @. Jρw_ρ.d2[1:N - 1] = Jρw_ρ.d[2:N]
        Jρw_ρ.d2[N] = 0

        Jρw_ρθ.d[1] = 0
        @. Jρw_ρθ.d[2:N] = -∂P∂ρθ[2:N] / Δzf
        @. Jρw_ρθ.d2[1:N - 1] = ∂P∂ρθ[1:N - 1] / Δzf
        Jρw_ρθ.d2[N] = 0

        # @. Jρ_ρw.d = 0
        # @. Jρ_ρw.d2 = 0
        # @. Jρθ_ρw.d = 0
        # @. Jρθ_ρw.d2 = 0
        # @. Jρw_ρ.d = 0
        # @. Jρw_ρ.d2 = 0
        # @. Jρw_ρθ.d = 0
        # @. Jρw_ρθ.d2 = 0
    end
end

function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        @unpack velem, helem, npoly, transform, dtγ_ref, Jρ_ρw′, Jρθ_ρw′,
            Jρw_ρ′, Jρw_ρθ′, S = A
        dtγ = dtγ_ref[]
        
        N = velem
        # TODO: Remove duplicate column computations.
        for i in 1:npoly + 1, h in 1:helem
            m = (h - 1) * (npoly + 1) + i
            schur_solve!(
                reshape(parent(Spaces.column(x.Yc.ρ, i, 1, h)), N),
                reshape(parent(Spaces.column(x.Yc.ρθ, i, 1, h)), N),
                reshape(parent(Spaces.column(x.ρw, i, 1, h)), N + 1),
                Jρ_ρw′[m],
                Jρθ_ρw′[m],
                Jρw_ρ′[m],
                Jρw_ρθ′[m],
                reshape(parent(Spaces.column(b.Yc.ρ, i, 1, h)), N),
                reshape(parent(Spaces.column(b.Yc.ρθ, i, 1, h)), N),
                reshape(parent(Spaces.column(b.ρw, i, 1, h)), N + 1),
                dtγ,
                S,
            )
        end

        parent(x.Yc.ρuₕ) .= -parent(b.Yc.ρuₕ)

        if transform
            parent(x) .*= dtγ
        end
    end
end

using OrdinaryDiffEq
function inertial_gravity_wave_prob(;
    helem,
    velem,
    npoly,
    is_large_domain,
    ode_algorithm,
    is_imex,
    tspan,
)
    xmax = is_large_domain ? 1500000. : 150000.
    hv_center_space, hv_face_space =
        hvspace_2D((-xmax, xmax), (0., 10000.), helem, velem, npoly)
    coords = Fields.coordinate_field(hv_center_space)
    face_coords = Fields.coordinate_field(hv_face_space)
    Yc = map(
        coord -> init_inertial_gravity_wave(coord.x, coord.z, is_large_domain),
        coords,
    )
    ρw = map(coord -> Geometry.Cartesian3Vector(0.0), face_coords)
    Y = Fields.FieldVector(Yc = Yc, ρw = ρw)

    ∂f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Cartesian3Vector(grav)),
        top = Operators.SetGradient(Geometry.Cartesian3Vector(grav)),
    )
    If_bc = Operators.InterpolateC2F(
        bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
    )
    Φf = @. ∂f(Φ(coords.z))
    uₕ = @. Yc.ρuₕ / Yc.ρ
    uₕf = @. If_bc(uₕ)
    P = @. pressure(Yc.ρθ)
    p = (; coords, face_coords, Φf, uₕ, uₕf, P)
    
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    jac_prototype = CustomWRepresentation(
        velem,
        helem,
        npoly,
        coords,
        face_coords,
        use_transform,
    )
    w_kwarg = use_transform ? (; Wfact_t = Wfact!) : (; Wfact = Wfact!)
    if is_imex
        prob = SplitODEProblem(
            ODEFunction(
                rhs_vertical!;
                w_kwarg...,
                jac_prototype = jac_prototype,
                tgrad = (dT, Y, p, t) -> fill!(dT, 0),
            ),
            rhs_horizontal!,
            Y,
            tspan,
            p,
        )
    else
        prob = ODEProblem(
            ODEFunction(
                rhs!;
                w_kwarg...,
                jac_prototype = jac_prototype,
                tgrad = (dT, Y, p, t) -> fill!(dT, 0),
            ),
            Y,
            tspan,
            p,
        )
    end

    return prob
end

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

function inertial_gravity_wave_plots(sol, name)
    dirname = "inertial_gravity_wave"
    path = joinpath(@__DIR__, "output", dirname)
    mkpath(path)

    coords = sol.prob.p.coords
    θ_ref = 300. .* exp.(coords.z .* (0.01 * 0.01 / grav))
    anim = Plots.@animate for u in sol.u
        Plots.plot(u.Yc.ρθ ./ u.Yc.ρ .- θ_ref, clim = (-0.002, 0.012))
    end
    Plots.mp4(anim, joinpath(path, "Δθ_$name.mp4"), fps = 20)
    anim = Plots.@animate for u in sol.u
        Plots.plot(
            pressure.(u.Yc.ρθ) .- pressure.(u.Yc.ρ .* θ_ref),
            clim = (0., 3.),
        )
    end
    Plots.mp4(anim, joinpath(path, "Δp_$name.mp4"), fps = 20)
end

# reltol = 1e-2
# abstol = 1e-8
# use generated functions to build timestepper from Butcher Tableau
#     handle 0, 1, -1, and 1//N nicely
#     automatically unroll stage looops and eliminate tableau dereferences