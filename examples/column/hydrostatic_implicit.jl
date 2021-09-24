push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

import ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

using OrdinaryDiffEq:
    OrdinaryDiffEq,
    ODEProblem,
    ODEFunction,
    solve,
    SSPRK33,
    Rosenbrock23,
    ImplicitEuler

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using LinearAlgebra
global_logger(TerminalLogger())

using UnPack

const FT = Float64

# https://github.com/CliMA/CLIMAParameters.jl/blob/master/src/Planet/planet_parameters.jl#L5
const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacit at constant volume
const R_m = R_d # moist R, assumed to be dry


domain = Domains.IntervalDomain(0.0, 30e3, x3boundary = (:bottom, :top))
#mesh = Meshes.IntervalMesh(domain, Meshes.ExponentialStretching(7.5e3); nelems = 30)
mesh = Meshes.IntervalMesh(domain; nelems = 30)

cspace = Spaces.CenterFiniteDifferenceSpace(mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

# https://github.com/CliMA/Thermodynamics.jl/blob/main/src/TemperatureProfiles.jl#L115-L155
# https://clima.github.io/Thermodynamics.jl/dev/TemperatureProfiles/#DecayingTemperatureProfile
function decaying_temperature_profile(z; T_virt_surf = 280.0, T_min_ref = 230.0)
    # Scale height for surface temperature
    H_sfc = R_d * T_virt_surf / grav
    H_t = H_sfc

    z′ = z / H_t
    tanh_z′ = tanh(z′)

    ΔTv = T_virt_surf - T_min_ref
    Tv = T_virt_surf - ΔTv * tanh_z′

    ΔTv′ = ΔTv / T_virt_surf
    p =
        MSLP * exp(
            (
                -H_t *
                (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
            ) / (H_sfc * (1 - ΔTv′^2)),
        )
    ρ = p / (R_d * Tv)
    ρθ = ρ * Tv * (MSLP / p)^(R_m / C_p)
    return (ρ = ρ, ρθ = ρθ)
end

Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)
Φ(z) = grav * z

function discrete_hydrostatic_balance!(
    ρ,
    w,
    ρθ,
    Δz::Float64,
    _grav::Float64,
    Π::Function,
)
    # compute θ such that
    #   I(θ)[i+1/2] = -g / ∂f(Π(ρθ))
    # discretely, then set
    #   ρ = ρθ/θ
    for i in 1:(length(ρ) - 1)
        #  ρ[i+1] = ρθ[i+1]/(-2Δz*_grav/(Π(ρθ[i+1]) - Π(ρθ[i])) - ρθ[i]/ρ[i])
        ρ[i + 1] =
            ρθ[i + 1] /
            (-2 * _grav / ((Π(ρθ[i + 1]) - Π(ρθ[i])) / Δz) - ρθ[i] / ρ[i])

        ρ[i + 1] =
            ρθ[i + 1] /
            (1 / ((-2 * _grav) * (Π(ρθ[i + 1]) - Π(ρθ[i]))Δz) - ρθ[i] / ρ[i])

        ∂Π∂z = (Π(ρθ[i + 1]) - Π(ρθ[i])) / Δz
    end
end

zc = Fields.coordinate_field(cspace)
Yc = decaying_temperature_profile.(zc)
w = Geometry.Cartesian3Vector.(zeros(FT, fspace))
zf = parent(Fields.coordinate_field(fspace))
Δz = zf[2:end] - zf[1:(end - 1)]
Y_init = copy(Yc)
w_init = copy(w)
Y = Fields.FieldVector(Yc = Yc, w = w)

function tendency!(dY, Y, _, t)
    Yc = Y.Yc
    w = Y.w
    dYc = dY.Yc
    dw = dY.w

    If = Operators.InterpolateC2F()
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
    )
    ∂f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
    )

    @. dYc.ρ = -(∂(w * If(Yc.ρ)))
    @. dYc.ρθ = -(∂(w * If(Yc.ρθ)))
    @. dw = B(
        Geometry.CartesianVector(
            -(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - ∂f(Φ(zc)),
        ),
    )
    return dY
end

dY = tendency!(similar(Y), Y, nothing, 0.0)

function jacobian!(J, Y, p, t)
    # copyto!(J, LinearAlgebra.I)
    # @info length(Y)
    # @info Y[1]

    J .= 0.0

    @info "Jacobian computation!!!!", t
    # N cells
    N = div(length(Y) - 1, 3)

    ρ, ρθ, w = Y[1:N], Y[(N + 1):(2N)], Y[(2N + 1):(3N + 1)]

    # construct cell center
    ρh = [ρ[1]; (ρ[1:(N - 1)] + ρ[2:N]) / 2.0; ρ[N]]
    ρθh = [ρθ[1]; (ρθ[1:(N - 1)] + ρθ[2:N]) / 2.0; ρθ[N]]


    Πc = Π.(ρθ)
    Πh = [NaN64; (Πc[1:(N - 1)] + Πc[2:N]) / 2.0; NaN64]
    Δzh = [NaN64; (Δz[1:(N - 1)] + Δz[2:N]) / 2.0; NaN64]


    for i in 1:N
        J[i, i + 2N] = ρh[i] / Δz[i]
        J[i, i + 2N + 1] = -ρh[i + 1] / Δz[i]
    end

    for i in 1:N
        J[i + N, i + 2N] = ρθh[i] / Δz[i]
        J[i + N, i + 2N + 1] = -ρθh[i + 1] / Δz[i]
    end

    # 0 for i = 1, N+1
    for i in 2:N
        J[i + 2N, (i - 1)] = -grav / (2 * ρh[i])
        J[i + 2N, (i - 1) + 1] = -grav / (2 * ρh[i])

        J[i + 2N, (i - 1) + N] = (γ - 1) * Πh[i] ./ (ρh[i] * Δzh[i])
        J[i + 2N, (i - 1) + 1 + N] = -(γ - 1) * Πh[i] ./ (ρh[i] * Δzh[i])
    end


    return J

    # D_ρ = diagm(0=>-ρh/Δz, -1=>ρh/Δz)[1:N, 1:N-1]
    # D_Θ = diagm(0=>-ρθh/Δz, -1=>ρθh/Δz)[1:N, 1:N-1]
    # G_W = (γ - 1) * diagm(0=>Πh./ρh/Δz, 1=>-Πh./ρh/Δz)[1:N-1, 1:N]
    # A_W = diagm(0=>-ones(N-1)./ρh/2, 1=>-ones(N-1)./ρh/2)[1:N-1, 1:N]

    # P = ([zeros(N,N)     zeros(N,N)      D_ρ;
    #       zeros(N,N)     zeros(N,N)      D_Θ
    #       A_W*_grav        G_W          zeros(N+1,N+1)])

end

#=
There are 4 possible types of general bidiagonal matrices:

1 2 . . .          1 . . . .
. 1 2 . .    or    2 1 . . .
. . 1 2 .          . 2 1 . .

or

1 2 .          1 . .
. 1 2          2 1 .
. . 1    or    . 2 1
. . .          . . 2
. . .          . . .
=#
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
#=
Other possible GeneralBidiagonal multiplications:

U * U:

1 2 .         3 4 .         13    14+23 24
. 1 2    *    . 3 4    =    .     13    14+23
. . 1         . . 3         .     .     13

                  3 4 .
1 2 . . .         . 3 4         13    14+23 24
. 1 2 . .    *    . . 3    =    .     13    14+23
. . 1 2 .         . . .         .     .     13
                  . . .

1 2 .                           13    14+23 24    .     .
. 1 2         3 4 . . .         .     13    14+23 24    .
. . 1    *    . 3 4 . .    =    .     .     13    14    .
. . .         . . 3 4 .         .     .     .     .     .
. . .                           .     .     .     .     .

L * L:

1 . .         3 . .         13    .     .
2 1 .    *    4 3 .    =    14+23 13    .
. 2 1         . 4 3         24    14+23 13

                  3 . .
1 . . . .         4 3 .         13    .     .
2 1 . . .    *    . 4 3    =    14+23 13    .
. 2 1 . .         . . 4         24    14+23 13
                  . . .

1 . .                           13    .     .     .     .
2 1 .         3 . . . .         14+23 13    .     .     .
. 2 1    *    4 3 . . .    =    24    14+23 13    .     .
. . 2         . 4 3 . .         .     24    23    .     .
. . .                           .     .     .     .     .
=#

struct CustomFactorization{T, AT}
    dtgamma_ref::T # Reference, so that we can modify it.
    Jρ_w::AT
    Jρθ_w::AT
    Jw_ρ::AT
    Jw_ρθ::AT
    S::AT
    W_test::AT
end
CustomFactorization(n::Integer; FT = Float64) =
    CustomFactorization{Base.RefValue{FT}, Array{FT}}(
        Ref(zero(FT)),
        zeros(FT, n, n + 1),
        zeros(FT, n, n + 1),
        zeros(FT, n + 1, n),
        zeros(FT, n + 1, n),
        zeros(FT, n + 1, n + 1),
        zeros(FT, 3n + 1, 3n + 1),
    )
import Base: similar
Base.similar(cf::CustomFactorization{T, AT}) where {T, AT} = # cf
    CustomFactorization{T, AT}(
        deepcopy(cf.dtgamma_ref),
        deepcopy(cf.Jρ_w),
        deepcopy(cf.Jρθ_w),
        deepcopy(cf.Jw_ρ),
        deepcopy(cf.Jw_ρθ),
        deepcopy(cf.S),
        deepcopy(cf.W_test),
    )

function Wfact!(W, u, p, dtgamma, t)
    @unpack dtgamma_ref, Jρ_w, Jρθ_w, Jw_ρ, Jw_ρθ, S, W_test = W

    dtgamma_ref[] = dtgamma

    N = div(length(Y) - 1, 3)

    ρ, ρθ, w = u[1:N], u[(N + 1):(2N)], u[(2N + 1):(3N + 1)]
    # construct cell center
    ρh = [ρ[1]; (ρ[1:(N - 1)] + ρ[2:N]) / 2.0; ρ[N]]
    ρθh = [ρθ[1]; (ρθ[1:(N - 1)] + ρθ[2:N]) / 2.0; ρθ[N]]


    Πc = Π.(ρθ)
    Πh = [NaN; (Πc[1:(N - 1)] + Πc[2:N]) / 2.0; NaN]
    Δzh = [NaN; (Δz[1:(N - 1)] + Δz[2:N]) / 2.0; NaN]

    # (dρₜ/dw)
    # Bidiagonal
    for i in 1:N
        Jρ_w[i, i] = ρh[i] / Δz[i]
        Jρ_w[i, i + 1] = -ρh[i + 1] / Δz[i]
    end

    # D_Θ = diagm(0=>-ρθh/Δz, -1=>ρθh/Δz)[1:N, 1:N-1]
    # (dρΘₜ/dw)
    # Bidiagonal
    for i in 1:N
        Jρθ_w[i, i] = ρθh[i] / Δz[i]
        Jρθ_w[i, i + 1] = -ρθh[i + 1] / Δz[i]
    end

    # (dwₜ/dρ) = A_W*_grav
    # A_W = diagm(0=>-ones(N-1)./ρh/2, 1=>-ones(N-1)./ρh/2)[1:N-1, 1:N]
    # Bidiagonal
    for i in 2:N
        Jw_ρ[i, (i - 1)] = -grav / (2 * ρh[i])
        Jw_ρ[i, (i - 1) + 1] = -grav / (2 * ρh[i])
    end

    # G_W = (γ - 1) * diagm(0=>Πh./ρh/Δz, 1=>-Πh./ρh/Δz)[1:N-1, 1:N]
    # (dwₜ/dρΘ) = G_W
    # Bidiagonal
    for i in 2:N
        Jw_ρθ[i, (i - 1)] = (γ - 1) * Πh[i] ./ (ρh[i] * Δzh[i])
        Jw_ρθ[i, (i - 1) + 1] = -(γ - 1) * Πh[i] ./ (ρh[i] * Δzh[i])
    end

    jacobian!(W_test, u, p, t)
    W_test .= -(I - dtgamma .* W_test)
end
function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        x_test = deepcopy(x)
        b_test = deepcopy(b)
        

        @unpack dtgamma_ref, Jρ_w, Jρθ_w, Jw_ρ, Jw_ρθ, S, W_test = A
        
        dtgamma = dtgamma_ref[]

        x1 = parent(x_test.Yc.ρ)
        x2 = parent(x_test.Yc.ρθ)
        x3 = parent(x_test.w)
        b1 = parent(b_test.Yc.ρ)
        b2 = parent(b_test.Yc.ρθ)
        b3 = parent(b_test.w)
        

        # A = -I + dtgamma J

        # J = ([zeros(N,N)           zeros(N,N)      D_ρ (dρₜ/dw);
        #       zeros(N,N)           zeros(N,N)      D_Θ (dρΘₜ/dw)
        #       A_W*_grav (dwₜ/dρ)  G_W (dwₜ/dρΘ)  zeros(N+1,N+1)])

        # A = ([-I               0               dtgamma*(dρₜ/dw);
        #       0               -I               dtgamma*(dρΘₜ/dw)
        #       dtgamma*(dwₜ/dρ)   dtgamma*(dwₜ/dρΘ)       -I           ])


        # A = ([-I               0           A13;
        #       0               -I           A23
        #       A31             A32         -I    ])
        # b = ([b1;
        #       b2
        #       b3])

        # solve for x
        # A* [x1; x2; x3] = [b1; b2; b3]

        # x1 = -b1 + A13 * x3  (1)
        # x2 = -b2 + A23 * x3  (2)
        # A31 x1 + A32 x2 - x3 = b3 (3)
        # bring x1 and x2 into eq(3)
        # A31 ( -b1 + A13 * x3) + A32 (-b2 + A23 * x3) - x3 = b3
        # S:= -I + A31*A13 + A32*A23 "Schur complement" =>  Tridiagonal
        # RHS = b3 +
        # 1) Form tridiagonal matrix
        S .= 0
        S[diagind(S)] .= -1
        # S = S + dtgamma^2 * Jw_ρ * Jρ_w
        mul!(S, Jw_ρ, Jρ_w, dtgamma^2, 1)
        # S = S + dtgamma^2 * Jw_ρθ * Jρθ_w
        mul!(S, Jw_ρθ, Jρθ_w, dtgamma^2, 1)
        # S * x3 = b3 - A31 *b1 - A32 * b2

        # 2) form RHS
        # x3 = S\(b3 - A31 *b1 - A32 * b2)
        # x3 = b3 + dtgamma * Jw_ρ *b1 + dtgamma * Jw_ρθ * b2
        x3 .= b3
        mul!(x3, Jw_ρ, b1, dtgamma, 1)
        mul!(x3, Jw_ρθ, b2, dtgamma, 1)

        S_test = copy(S)

        # 3) solve for x3
        # TODO: LinearAlgebra will compute the LU factorization, then solve
        # Thomas' algorithm can do this in one step:
        # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        # can also reuse x3 storage for t3, and solve in-place
        ldiv!(lu!(S), x3)


        # 4) compute x1
        # use (1) & (2) to get x1, x2
        # x1 = -b1 + A13 * x3
        # x1 .= -b1 + dtgamma * Jρ_w * x3
        x1 .= -b1
        mul!(x1, Jρ_w, x3, dtgamma, 1)

        # 5) compute x2
        # x2 .= b2 + dtgamma * Jρθ_w * x3
        x2 .= -b2
        mul!(x2, Jρθ_w, x3, dtgamma, 1)
        
        x .= copy(b)
        N = div(length(x) - 1, 3)
        J = W_test[(2N + 1):(3N + 1), (2N + 1):(3N + 1)]
        J +=
            -W_test[(2N + 1):(3N + 1), 1:N] *
            (Diagonal(W_test[1:N, 1:N]) \ W_test[1:N, (2N + 1):(3N + 1)])
        J +=
            -W_test[(2N + 1):(3N + 1), (N + 1):(2N)] * (
                Diagonal(W_test[(N + 1):(2N), (N + 1):(2N)]) \
                W_test[(N + 1):(2N), (2N + 1):(3N + 1)]
            )

        x[(2N + 1):(3N + 1)] +=
            -W_test[(2N + 1):(3N + 1), 1:N] * (Diagonal(W_test[1:N, 1:N]) \ b[1:N])
        x[(2N + 1):(3N + 1)] +=
            -W_test[(2N + 1):(3N + 1), (N + 1):(2N)] *
            (Diagonal(W_test[(N + 1):(2N), (N + 1):(2N)]) \ b[(N + 1):(2N)])

        x[(2N + 1):(3N + 1)] .= Tridiagonal(J) \ x[(2N + 1):(3N + 1)]
        x[1:N] .=
            Diagonal(W_test[1:N, 1:N]) \
            (b[1:N] - W_test[1:N, (2N + 1):(3N + 1)] * x[(2N + 1):(3N + 1)])
        x[(N + 1):(2N)] .=
            Diagonal(W_test[(N + 1):(2N), (N + 1):(2N)]) \ (
                b[(N + 1):(2N)] -
                W_test[(N + 1):(2N), (2N + 1):(3N + 1)] * x[(2N + 1):(3N + 1)]
            )

        @assert J ≈ S_test
        @assert x ≈ x_test
        @assert b ≈ b_test
    end
end

Δt = 5.0
ndays = 1.0

# Solve the ODE operator
prob = ODEProblem(
    ODEFunction(
        tendency!,
        # jac = jacobian!,
        # jac_prototype = zeros(length(Y), length(Y)),
        Wfact = Wfact!,
        jac_prototype = CustomFactorization((length(Y) - 1) ÷ 3),
        tgrad = (dT, Y, p, t) -> fill!(dT, 0),
    ),
    Y,
    (0.0, 60 * 60 * 24 * ndays),
)
# 60 * 60 * 24 * ndays
sol = solve(
    prob,
    # ImplicitEuler(linsolve = linsolve!),
    Rosenbrock23(linsolve = linsolve!),
    dt = Δt,
    adaptive = false,
    saveat = 60 * 60, # save every hour
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "hydrostatic_implicit"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

z_centers = parent(Fields.coordinate_field(cspace))
z_faces = parent(Fields.coordinate_field(fspace))

function hydrostatic_plot(u, Yc_init, w_init; title = "", size = (1024, 600))
    sub_plt1 = Plots.plot(
        parent(Yc_init.ρ),
        z_centers,
        marker = :circle,
        xlabel = "ρ",
        label = "T=0",
    )
    sub_plt1 = Plots.plot!(sub_plt1, parent(u.Yc.ρ), z_centers, label = "T")

    sub_plt2 = Plots.plot(
        parent(w_init),
        z_faces,
        marker = :circle,
        xlim = (-0.2, 0.2),
        xlabel = "ω",
        label = "T=0",
    )
    sub_plt2 = Plots.plot!(sub_plt2, parent(u.w), z_faces, label = "T")

    sub_plt3 = Plots.plot(
        parent(Yc_init.ρθ),
        z_centers,
        marker = :circle,
        xlabel = "ρθ",
        label = "T=0",
    )
    sub_plt3 = Plots.plot!(sub_plt3, parent(u.Yc.ρθ), z_centers, label = "T")

    return Plots.plot(
        sub_plt1,
        sub_plt2,
        sub_plt3,
        title = title,
        layout = (1, 3),
        size = size,
    )
end

# anim = Plots.@animate for (i, u) in enumerate(sol.u)
#     hydrostatic_plot(u, Y_init, w_init, title = "Hour $(i)")
# end
# Plots.mp4(anim, joinpath(path, "hydrostatic.mp4"), fps = 10)

Plots.png(
    hydrostatic_plot(sol[end], Y_init, w_init),
    joinpath(path, "hydrostatic_end.png"),
)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/hydrostatic_end.png", "Hydrostatic End")
