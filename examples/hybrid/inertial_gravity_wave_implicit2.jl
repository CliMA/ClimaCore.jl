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

helem = 75
velem = 10 # Use 20 if results are poor.
npoly = 4

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

# set up rhs!
hv_center_space, hv_face_space =
    hvspace_2D((-150000, 150000), (0, 10000), helem, velem, npoly)
    # hvspace_2D((-1500000, 1500000), (0, 10000), helem, velem, npoly)

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

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_inertial_gravity_wave(x, z)
    p_0 = MSLP
    g = grav
    cp_d = C_p
    x_c = 0.
    θ_0 = 300.
    Δθ = 0.01
    A = 5000. # 100000.
    H = 10000.
    NBr = 0.01
    S = NBr * NBr / g

    p_ref = p_0 * (1 - g / (cp_d * θ_0 * S) * (1 - exp(-S * z)))^(cp_d / R_d)
    θ = θ_0 * exp(z * S) + Δθ * sin(pi * z / H) / (1 + ((x - x_c) / A)^2)
    ρ = p_ref / ((p_ref / p_0)^(R_d / cp_d) * R_d * θ)
    ρθ = ρ * θ

    return (ρ = ρ, ρθ = ρθ, ρuₕ = ρ * Geometry.Cartesian1Vector(0.0))
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space);
face_coords = Fields.coordinate_field(hv_face_space);
Yc = map(coord -> init_inertial_gravity_wave(coord.x, coord.z), coords);
ρw = map(coord -> Geometry.Cartesian3Vector(0.0), face_coords);
Y = Fields.FieldVector(Yc = Yc, ρw = ρw);

function rhs!(dY, Y, _, t)
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

    uₕ = @. Yc.ρuₕ / Yc.ρ
    w = @. ρw / If(Yc.ρ)
    p = @. pressure(Yc.ρθ)

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
    @. dYc.ρuₕ = -hdiv(Yc.ρuₕ ⊗ uₕ + p * Ih)
    @. dYc.ρuₕ -= uvdivf2c(ρw ⊗ If_bc(uₕ))

    # vertical momentum
    @. dρw = B(
        Geometry.transform(
            Geometry.Cartesian3Axis(),
            -(∂f(p)) - If(Yc.ρ) * ∂f(Φ(coords.z)),
        ) - vvdivc2f(Ic(ρw ⊗ w)),
    )
    uₕf = @. If_bc(Yc.ρuₕ / Yc.ρ) # requires boundary conditions
    @. dρw -= hdiv(uₕf ⊗ ρw)

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

function rhs_vertical!(dY, Y, _, t)
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

    # density
    @. dYc.ρ = -∂(ρw)

    # potential temperature
    @. dYc.ρθ = -(∂(ρw * If(Yc.ρθ / Yc.ρ)))

    # vertical momentum
    @. dρw = B(
        Geometry.transform(
            Geometry.Cartesian3Axis(),
            -(∂f(pressure(Yc.ρθ))) - If(Yc.ρ) * ∂f(Φ(coords.z)),
        )
    )
    return dY
end

function rhs_horizontal!(dY, Y, _, t)
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

    uₕ = @. Yc.ρuₕ / Yc.ρ
    w = @. ρw / If(Yc.ρ)
    p = @. pressure(Yc.ρθ)

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
    @. dYc.ρuₕ = -hdiv(Yc.ρuₕ ⊗ uₕ + p * Ih)
    @. dYc.ρuₕ -= uvdivf2c(ρw ⊗ If_bc(uₕ))

    # vertical momentum
    @. dρw = B(-vvdivc2f(Ic(ρw ⊗ w)))
    uₕf = @. If_bc(Yc.ρuₕ / Yc.ρ) # requires boundary conditions
    @. dρw -= hdiv(uₕf ⊗ ρw)

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

struct GeneralBidiagonal{T,AT<:AbstractVector{T}} <: AbstractMatrix{T}
    d::AT
    d2::AT
    isUpper::Bool
    nrows::Int
    ncols::Int
end
function GeneralBidiagonal(
    ::Type{AT},
    isUpper::Bool,
    nrows::Int,
    ncols::Int,
) where {AT}
    nd = min(nrows, ncols)
    nd2 = (isUpper ? ncols : nrows) > nd ? nd : nd - 1
    @assert nd2 > 0
    d = AT(undef, nd)
    d2 = AT(undef, nd2)
    return GeneralBidiagonal{eltype(d), typeof(d)}(d, d2, isUpper, nrows, ncols)
end

import Base: size, getindex, setindex!
size(A::GeneralBidiagonal) = (A.nrows, A.ncols)
function getindex(A::GeneralBidiagonal, i::Int, j::Int)
    @boundscheck 1 <= i <= A.nrows && 1 <= j <= A.ncols
    if i == j
        return A.d[i]
    elseif A.isUpper && j == i + 1
        return A.d2[i]
    elseif !A.isUpper && i == j + 1
        return A.d2[j]
    else
        return zero(eltype(A))
    end
end
function setindex!(A::GeneralBidiagonal, v, i::Int, j::Int)
    @boundscheck 1 <= i <= A.nrows && 1 <= j <= A.ncols
    if i == j
        A.d[i] = v
    elseif A.isUpper && j == i + 1
        A.d2[i] = v
    elseif !A.isUpper && i == j + 1
        A.d2[j] = v
    elseif !iszero(v)
        throw(ArgumentError(
            "Setting A[$i, $j] to $v will make A no longer be GeneralBidiagonal"
        ))
    end
end

import LinearAlgebra: mul!
function mul!(
    C::AbstractVector,
    A::GeneralBidiagonal,
    B::AbstractVector,
    α::Number,
    β::Number,
)
    if A.nrows != length(C)
        throw(DimensionMismatch(
            "A has $(A.nrows) rows, but C has length $(length(C))"
        ))
    end
    if A.ncols != length(B)
        throw(DimensionMismatch(
            "A has $(A.ncols) columns, but B has length $(length(B))"
        ))
    end
    if iszero(α)
        return LinearAlgebra._rmul_or_fill!(C, β)
    end
    nd = length(A.d)
    nd2 = length(A.d2)
    @inbounds if A.isUpper
        if nd2 == nd
            @views @. C = α * (A.d * B[1:nd] + A.d2 * B[2:nd + 1]) + β * C
        else
            @views @. C[1:nd - 1] =
                α * (A.d[1:nd - 1] * B[1:nd - 1] + A.d2 * B[2:nd]) +
                β * C[1:nd - 1]
            C[nd] = α * A.d[nd] * B[nd] + β * C[nd]
        end
    else
        C[1] = α * A.d[1] * B[1] + β * C[1]
        @views @. C[2:nd] =
            α * (A.d[2:nd] * B[2:nd] + A.d2[1:nd - 1] * B[1:nd - 1]) +
            β * C[2:nd]
        if nd2 == nd
            C[nd + 1] = α * A.d2[nd] * B[nd] + β * C[nd + 1]
        end
    end
    C[nd2 + 2:end] .= zero(eltype(C))
    return C
end
function mul!(
    C::Tridiagonal,
    A::GeneralBidiagonal,
    B::GeneralBidiagonal,
    α::Number,
    β::Number,
)
    if A.nrows != B.ncols || A.nrows != size(C, 1)
        throw(DimensionMismatch(string(
            "A has $(A.nrows) rows, B has $(B.ncols) columns, and C has ",
            "$(size(C, 1)) rows/columns, but all three must match"
        )))
    end
    if A.ncols != B.nrows
        throw(DimensionMismatch(
            "A has $(A.ncols) columns, but B has $(B.rows) rows"
        ))
    end
    if A.isUpper && B.isUpper
        throw(ArgumentError(
            "A and B are both upper bidiagonal, so C is not tridiagonal"
        ))
    end
    if !A.isUpper && !B.isUpper
        throw(ArgumentError(
            "A and B are both lower bidiagonal, so C is not tridiagonal"
        ))
    end
    if iszero(α)
        return LinearAlgebra._rmul_or_fill!(C, β)
    end
    nd = length(A.d) # == length(B.d)
    nd2 = length(A.d2) # == length(B.d2)
    @inbounds if A.isUpper # && !B.isUpper
        if nd2 == nd
            #                   3 . .
            # 1 2 . . .         4 3 .         13+24 23    .
            # . 1 2 . .    *    . 4 3    =    14    13+24 23
            # . . 1 2 .         . . 4         .     14    13+24
            #                   . . .
            @. C.d = α * (A.d * B.d + A.d2 * B.d2) + β * C.d
        else
            # 1 2 .                           13+24 23    .     .     .
            # . 1 2         3 . . . .         14    13+24 23    .     .
            # . . 1    *    4 3 . . .    =    .     14    13    .     .
            # . . .         . 4 3 . .         .     .     .     .     .
            # . . .                           .     .     .     .     .
            @views @. C.d[1:nd - 1] =
                α * (A.d[1:nd - 1] * B.d[1:nd - 1] + A.d2 * B.d2) +
                β * C.d[1:nd - 1]
            C.d[nd] = α * A.d[nd] * B.d[nd] + β * C.d[nd]
        end
        @views @. C.du[1:nd - 1] =
            α * A.d2[1:nd - 1] * B.d[2:nd] + β * C.du[1:nd - 1]
        @views @. C.dl[1:nd - 1] =
            α * A.d[2:nd] * B.d2[1:nd - 1] + β * C.dl[1:nd - 1]
    else # !A.isUpper && B.isUpper
        C.d[1] = α * A.d[1] * B.d[1] + β * C.d[1]
        @views @. C.d[2:nd] =
            α * (A.d[2:nd] * B.d[2:nd] + A.d2[1:nd - 1] * B.d2[1:nd - 1]) +
            β * C.d[2:nd]
        if nd2 == nd
            # 1 . .                           13    14    .     .     .
            # 2 1 .         3 4 . . .         23    13+24 14    .     .
            # . 2 1    *    . 3 4 . .    =    .     23    13+24 14    .
            # . . 2         . . 3 4 .         .     .     23    24    .
            # . . .                           .     .     .     .     .
            C.d[nd + 1] = α * A.d2[nd] * B.d2[nd] + β * C.d[nd + 1]
        # else
            #                   3 4 .
            # 1 . . . .         . 3 4         13    14    .
            # 2 1 . . .    *    . . 3    =    23    13+24 14
            # . 2 1 . .         . . .         .     23    13+24
            #                   . . .
        end
        @views @. C.du[1:nd2] =
            α * A.d[1:nd2] * B.d2 + β * C.du[1:nd2]
        @views @. C.dl[1:nd2] =
            α * A.d2 * B.d[1:nd2] + β * C.dl[1:nd2]
    end
    C.d[nd2 + 2:end] .= zero(eltype(C))
    C.du[nd2 + 1:end] .= zero(eltype(C))
    C.dl[nd2 + 1:end] .= zero(eltype(C))
    return C
end

struct CustomWRepresentation{T,AT1,AT2,AT3,AT4}
    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::T

    # cache for the grid values used to compute the Jacobian
    Δz′::AT1
    Δzh′::AT1

    # cache for the face values used to compute the Jacobian
    ρh′::AT2
    ρθh′::AT2
    Πh′::AT2

    # nonzero blocks of the Jacobian (∂ρₜ/∂w, ∂ρθₜ/∂w, ∂wₜ/∂ρ, and ∂wₜ/∂ρθ)
    Jρ_w::AT3
    Jρθ_w::AT3
    Jw_ρ::AT3
    Jw_ρθ::AT3

    # cache for the Schur complement
    S::AT4
end

function CustomWRepresentation(; FT = Float64)
    N = velem
    M = helem * (npoly + 1)

    dtγ_ref = Ref(zero(FT))

    zf = reshape(parent(face_coords.z), N + 1, M)
    Δz′ = zf[2:end, :] .- zf[1:end - 1, :]
    Δzh′ = similar(zf)
    Δzh′[2:N, :] .= (Δz′[2:end, :] .+ Δz′[1:end - 1, :]) ./ 2

    ρh′ = [Array{FT}(undef, N + 1) for _ in 1:M]
    ρθh′ = [Array{FT}(undef, N + 1) for _ in 1:M]
    Πh′ = [Array{FT}(undef, N + 1) for _ in 1:M]

    Jρ_w = [GeneralBidiagonal(Array{FT}, true, N, N + 1) for _ in 1:M]
    Jρθ_w = [GeneralBidiagonal(Array{FT}, true, N, N + 1) for _ in 1:M]
    Jw_ρ = [GeneralBidiagonal(Array{FT}, false, N + 1, N) for _ in 1:M]
    Jw_ρθ = [GeneralBidiagonal(Array{FT}, false, N + 1, N) for _ in 1:M]

    S = Tridiagonal(
        Array{FT}(undef, N),
        Array{FT}(undef, N + 1),
        Array{FT}(undef, N),
    )

    CustomWRepresentation{
        typeof(dtγ_ref),
        typeof(Δz′),
        typeof(ρh′),
        typeof(Jρ_w),
        typeof(S),
    }(
        dtγ_ref,
        Δz′,
        Δzh′,
        ρh′,
        ρθh′,
        Πh′,
        Jρ_w,
        Jρθ_w,
        Jw_ρ,
        Jw_ρθ,
        S,
    )
end

import Base: similar, deepcopy
# We only use Wfact, but the Rosenbrock23 solver requires us to pass
# jac_prototype, then calls similar(jac_prototype) to obtain J and Wfact. This
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(cf::CustomWRepresentation{T,AT}) where {T, AT} = cf
Base.deepcopy(cf::CustomWRepresentation{T,AT}) where {T, AT} = cf

function Wfact!(W, u, p, dtγ, t)
    @unpack dtγ_ref, Δz′, Δzh′, ρh′, ρθh′, Πh′, Jρ_w, Jρθ_w, Jw_ρ, Jw_ρθ = W
    dtγ_ref[] = dtγ

    N = velem
    for i in 1:npoly + 1, h in 1:helem
        m = (h - 1) * (npoly + 1) + i

        Δz = reshape(view(Δz′, :, m), N)
        Δzh = reshape(view(Δzh′, :, m), N + 1)
        ρ = reshape(parent(Spaces.column(u.Yc.ρ, i, 1, h)), N)
        ρθ = reshape(parent(Spaces.column(u.Yc.ρθ, i, 1, h)), N)
        ρh = ρh′[m]
        ρθh = ρθh′[m]
        Πh = Πh′[m]

        # Compute the cell-face values
        
        ρh[1] = ρ[1]
        @views @. ρh[2:N] = (ρ[1:N - 1] + ρ[2:N]) / 2
        ρh[N + 1] = ρ[N]

        ρθh[1] = ρθ[1]
        @views @. ρθh[2:N] = (ρθ[1:N - 1] + ρθ[2:N]) / 2
        ρθh[N + 1] = ρθ[N]

        @views @. Πh[1:N] = Π(ρθ) # Temporarily store cell-center values in Πh
        @views @. Πh[2:N] = (Πh[1:N - 1] + Πh[2:N]) / 2

        # Compute the nonzero blocks of the Jacobian
        
        @views @. Jρ_w[m].d = ρh[1:N] / Δz
        @views @. Jρ_w[m].d2 = -ρh[2:N + 1] / Δz

        @views @. Jρθ_w[m].d = ρθh[1:N] / Δz
        @views @. Jρθ_w[m].d2 = -ρθh[2:N + 1] / Δz

        Jw_ρ[m].d[1] = 0
        Jw_ρ[m].d2[N] = 0
        @views @. Jw_ρ[m].d[2:N] = -grav / (2 * ρh[2:N])
        @views @. Jw_ρ[m].d2[1:N - 1] = Jw_ρ[m].d[2:N]

        Jw_ρθ[m].d[1] = 0
        Jw_ρθ[m].d2[N] = 0
        @views @. Jw_ρθ[m].d[2:N] = -(γ - 1) * Πh[2:N] / (ρh[2:N] * Δzh[2:N])
        @views @. Jw_ρθ[m].d2[1:N - 1] = -Jw_ρθ[m].d[2:N]

        Πh .= reshape(parent(Spaces.column(u.ρw, i, 1, h)), N + 1) ./ ρh
    end
end

function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        # A represents the matrix W = -I + dtγ * J, which can be expressed as
        #     [-I        0          dtγ Jρ_w ;
        #      0         -I         dtγ Jρθ_w;
        #      dtγ Jw_ρ  dtγ Jw_ρθ  -I       ] =
        #     [-I    0      Aρ_w ;
        #      0     -I     Aρθ_w;
        #      Aw_ρ  Aw_ρθ  -I   ]
        # b represents the vector [bρ; bρθ; bw]
        # x represents the vector [xρ; xρθ; xw]

        # Solving A x = b:
        #     -xρ + Aρ_w xw = bρ ==> xρ = -bρ + Aρ_w xw  (1)
        #     -xρθ + Aρθ_w xw = bρθ ==> xρθ = -bρθ + Aρθ_w xw  (2)
        #     Aw_ρ xρ + Aw_ρθ xρθ - xw = bw  (3)
        # Substitute (1) and (2) into (3):
        #     Aw_ρ (-bρ + Aρ_w xw) + Aw_ρθ (-bρθ + Aρθ_w xw) - xw = bw ==>
        #     (-I + Aw_ρ Aρ_w + Aw_ρθ Aρθ_w) xw = bw + Aw_ρ bρ + Aw_ρθ bρθ ==>
        #     xw = (-I + Aw_ρ Aρ_w + Aw_ρθ Aρθ_w) \ (bw + Aw_ρ bρ + Aw_ρθ bρθ)
        # Finally, use (1) and (2) to get xρ and xρθ.

        # Note: The tridiagonal matrix (-I + Aw_ρ Aρ_w + Aw_ρθ Aρθ_w) is the
        #       "Schur complement" of [-I 0; 0 -I] (the top-left 4 blocks) in A.
    
        @unpack dtγ_ref, ρh′, Πh′, S, Jρ_w, Jρθ_w, Jw_ρ, Jw_ρθ = A
        dtγ = dtγ_ref[]

        N = velem
        for i in 1:npoly + 1, h in 1:helem
            m = (h - 1) * (npoly + 1) + i

            xρ = reshape(parent(Spaces.column(x.Yc.ρ, i, 1, h)), N)
            xρθ = reshape(parent(Spaces.column(x.Yc.ρθ, i, 1, h)), N)
            xρw = reshape(parent(Spaces.column(x.ρw, i, 1, h)), N + 1)
            bρ = reshape(parent(Spaces.column(b.Yc.ρ, i, 1, h)), N)
            bρθ = reshape(parent(Spaces.column(b.Yc.ρθ, i, 1, h)), N)
            bρw = reshape(parent(Spaces.column(b.ρw, i, 1, h)), N + 1)

            ρ = ρh′[m]
            w = Πh′[m]
            xw = xρw # Temporarily store w values in xρw

            # LHS = -I + dtγ^2 Jw_ρ Jρ_w + dtγ^2 Jw_ρθ Jρθ_w
            S.dl .= 0
            S.d .= -1
            S.du .= 0
            mul!(S, Jw_ρ[m], Jρ_w[m], dtγ^2, 1)
            mul!(S, Jw_ρθ[m], Jρθ_w[m], dtγ^2, 1)

            # RHS = bw + dtγ Jw_ρ bρ + dtγ Jw_ρθ bρθ
            xw .= bρw ./ ρ
            mul!(xw, Jw_ρ[m], bρ, dtγ, 1)
            mul!(xw, Jw_ρθ[m], bρθ, dtγ, 1)

            # xw = LHS \ RHS
            # TODO: LinearAlgebra will compute lu! and then ldiv! in seperate steps.
            #       The Thomas algorithm can do this in one step:
            #       https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm.
            ldiv!(lu!(S), xw)

            # xρ = -bρ + dtγ Jρ_w xw
            xρ .= bρ
            mul!(xρ, Jρ_w[m], xw, dtγ, -1)

            # xρθ = -bρθ + dtγ Jρθ_w xw
            xρθ .= bρθ
            mul!(xρθ, Jρθ_w[m], xw, dtγ, -1)

            # Update w values to ρw values
            xρh = Array{Float64}(undef, N + 1)
            xρh[1] = xρ[1]
            xρh[2:N] .= (xρ[1:end - 1] .+ xρ[2:end]) ./ 2
            xρh[N + 1] = xρ[N]
            xρw .= ρ .* xw .+ w .* xρh

            xρ .*= dtγ
            xρθ .*= dtγ
            xρw .*= dtγ
        end
    end
end

using OrdinaryDiffEq
Δt = 20.
tspan = (0., 2000.)
# prob = ODEProblem(
#     ODEFunction(
#         rhs!,
#         Wfact = Wfact!,
#         jac_prototype = CustomWRepresentation(),
#         tgrad = (dT, Y, p, t) -> fill!(dT, 0),
#     ),
#     Y,
#     tspan,
# )
# sol = solve(
#     prob,
#     Rosenbrock23(linsolve = linsolve!),
#     dt = Δt,
#     adaptive = false,
#     saveat = 10.,
#     progress = true,
#     progress_steps = 1,
#     progress_message = (dt, u, p, t) -> t,
# );
prob = SplitODEProblem(
    ODEFunction(
        rhs_vertical!,
        Wfact_t = Wfact!,
        jac_prototype = CustomWRepresentation(),
        tgrad = (dT, Y, p, t) -> fill!(dT, 0),
    ),
    rhs_horizontal!,
    Y,
    tspan,
)
sol = solve(
    prob,
    KenCarp4(linsolve = linsolve!),
    dt = Δt,
    saveat = 10.0,
    progress = true,
    progress_steps = 1,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "inertial_gravity_wave"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# post-processing
import Plots
θ_ref = 300. .* exp.(coords.z .* (0.01 * 0.01 / grav))
anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρθ ./ u.Yc.ρ .- θ_ref, clim = (-0.002, 0.012))
end
Plots.mp4(anim, joinpath(path, "wave_Δθ_implicit2.mp4"), fps = 20)
anim = Plots.@animate for u in sol.u
    Plots.plot(pressure.(u.Yc.ρθ) .- pressure.(u.Yc.ρ .* θ_ref), clim = (0., 3.))
end
Plots.mp4(anim, joinpath(path, "wave_Δp_implicit2.mp4"), fps = 20)

# reltol = 1e-2
# abstol = 1e-8