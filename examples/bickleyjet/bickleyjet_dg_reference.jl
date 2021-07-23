# For analysis
#=
include("../../../perf/IntelITT.jl/src/IntelITT.jl")
using Main.IntelITT
include("../../../perf/SDE.jl/sde.jl")
=#

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Base.Threads
using Profile
using IntervalSets
using LinearAlgebra
using Logging: global_logger
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using StaticArrays
using TerminalLoggers: TerminalLogger
using UnPack

import ClimateMachineCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimateMachineCore: slab
import ClimateMachineCore.Operators
using ClimateMachineCore.Geometry
using ClimateMachineCore.RecursiveApply
using ClimateMachineCore.RecursiveApply: rdiv, rmap

global_logger(TerminalLogger())

const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
)

numflux_name = get(ARGS, 1, "roe")
boundary_name = get(ARGS, 2, "")

# standard implementation
# ========

domain = Domains.RectangleDomain(
    -2π..2π,
    -2π..2π,
    x1periodic = true,
    x2periodic = boundary_name != "noslip",
)

#n1, n2 = 16, 16
n1, n2 = 512, 512
Nq = 4
mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

function init_state(x, p)
    @unpack x1, x2 = x
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(x2)^(-2)

    # Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x1) * cos(p.k * x2)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(x2 + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (x2 + p.l / 10) / p.l^2 * cos(p.k * x1) * cos(p.k * x2)
    u₁′ += p.k * gaussian * cos(p.k * x1) * sin(p.k * x2)
    u₂′ = -p.k * gaussian * sin(p.k * x1) * cos(p.k * x2)

    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * x2)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, p)
    @unpack ρ, ρu, ρθ = state
    u = ρu ./ ρ
    return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * I), ρθ = ρθ .* u)
end

function energy(state, p)
    @unpack ρ, ρu = state
    u = ρu ./ ρ
    return ρ * (u.u1^2 + u.u2^2) / 2 + p.g * ρ^2 / 2
end

function total_energy(y, parameters)
    sum(state -> energy(state, parameters), y)
end

# numerical fluxes
wavespeed(y, parameters) = sqrt(parameters.g)

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function roeflux(n, (y⁻, parameters⁻), (y⁺, parameters⁺))
    Favg = rdiv(flux(y⁻, parameters⁻) ⊞ flux(y⁺, parameters⁺), 2)

    λ = sqrt(parameters⁻.g)

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n

    # in general thermodynamics, (pressure, soundspeed)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (w1 * (u - c * n) + w2 * (u + c * n) + w3 * u + w4 * (Δu - Δuₙ * n)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5

    Δf = (ρ = -fluxᵀn_ρ, ρu = -fluxᵀn_ρu, ρθ = -fluxᵀn_ρθ)
    rmap(f -> f' * n, Favg) ⊞ Δf
end

numflux = if numflux_name == "central"
    Operators.CentralNumericalFlux(flux)
elseif numflux_name == "rusanov"
    Operators.RusanovNumericalFlux(flux, wavespeed)
elseif numflux_name == "roe"
    roeflux
end

function rhs!(dydt, y, (parameters, numflux), t)

    # ϕ' K' W J K dydt =  -ϕ' K' I' [DH' WH JH flux.(I K y)]
    #  =>   K dydt = - K inv(K' WJ K) K' I' [DH' WH JH flux.(I K y)]

    # where:
    #  ϕ = test function
    #  K = DSS scatter (i.e. duplicates points at element boundaries)
    #  K y = stored input vector (with duplicated values)
    #  I = interpolation to higher-order space
    #  D = derivative operator
    #  H = suffix for higher-order space operations
    #  W = Quadrature weights
    #  J = Jacobian determinant of the transformation `ξ` to `x`
    #
    Nh = Topologies.nlocalelems(y)

    F = flux.(y, Ref(parameters))
    # TODO: get this to work
    #   F = Base.Broadcast.broadcasted(flux, y, Ref(parameters))
    Operators.slab_weak_divergence!(dydt, F)

    Operators.add_numerical_flux_internal!(numflux, dydt, y, parameters)

    Operators.add_numerical_flux_boundary!(
        dydt,
        y,
        parameters,
    ) do normal, (y⁻, parameters)
        y⁺ = (ρ = y⁻.ρ, ρu = y⁻.ρu .- dot(y⁻.ρu, normal) .* normal, ρθ = y⁻.ρθ)
        numflux(normal, (y⁻, parameters), (y⁺, parameters))
    end

    # 6. Solve for final result
    dydt_data = Fields.field_values(dydt)
    dydt_data .= rdiv.(dydt_data, space.local_geometry.WJ)

    # 7. cutoff filter
    M = Spaces.Quadratures.cutoff_filter_matrix(
        Float64,
        space.quadrature_style,
        3,
    )
    Operators.tensor_product!(dydt_data, M)

    return dydt
end

#dydt = Fields.Field(similar(Fields.field_values(y0)), space)
#rhs!(dydt, y0, (parameters, numflux), 0.0);

# Solve the ODE operator
#prob = ODEProblem(rhs!, y0, (0.0, 200.0), (parameters, numflux))

#__sde_start()
#__itt_resume()

#sol = @time solve(prob, SSPRK33(), dt = 0.02)
    #saveat = 1.0,
    #progress = true,
    #progress_message = (dt, u, p, t) -> t,

#__itt_pause()
#__sde_stop()

# reference implementation
# ========

const Nstate = 4

# data layout: i, j, k, n1, n2
#  n1,n2 topology
#  i,j,k datalayouts
X = reshape(parent(Fields.coordinate_field(space)), (Nq,Nq,2,n1*n2))
y0_ref = Array{Float64}(undef, (Nq,Nq,Nstate,n1*n2))

function init_y0_ref!(y0_ref, X, parameters)
    for h = 1:n1*n2
        for j = 1:Nq, i = 1:Nq
            x = (x1=X[i,j,1,h], x2=X[i,j,2,h])
            y = init_state(x, parameters)
            y0_ref[i,j,1,h] = y.ρ
            y0_ref[i,j,2,h] = y.ρu.u1
            y0_ref[i,j,3,h] = y.ρu.u2
            y0_ref[i,j,4,h] = y.ρθ
        end
    end
    return y0_ref
end

init_y0_ref!(y0_ref, X, parameters)

Nqf = 3
_,W = Spaces.Quadratures.quadrature_points(Float64, quad)
D = Spaces.Quadratures.differentiation_matrix(Float64, quad)
M = Spaces.Quadratures.cutoff_filter_matrix(Float64, quad, Nqf)

dydt_ref = similar(y0_ref)

Geometry.:⊗(u::SVector, v::SVector) = u*v'

getval(::Val{V}) where {V} = V

struct TendencyState{DT, WJT, ST}
    ∂ξ∂x::DT
    WJv¹::WJT
    WJv²::WJT
    scratch::ST

    function TendencyState(n1, n2, Nstate, Nq)
        ∂ξ∂x = @SMatrix [n1/(2pi) 0; 0 n2/(2pi)]
        WJv¹ = MArray{Tuple{Nstate, Nq, Nq}, Float64, 3, Nstate * Nq * Nq}(undef)
        WJv² = MArray{Tuple{Nstate, Nq, Nq}, Float64, 3, Nstate * Nq * Nq}(undef)
        scratch = MArray{Tuple{Nq, Nq, Nstate}, Float64, 3, Nq * Nq * Nstate}(undef)
        return new{typeof(∂ξ∂x), typeof(WJv¹), typeof(scratch)}(∂ξ∂x, WJv¹, WJv², scratch)
    end
end

const tendency_states = [TendencyState(n1, n2, Nstate, Nq) for _ in 1:nthreads()]

function tendency_test1(dydt_ref, y0_ref, (parameters, numflux, W, D, M, valNq, states), t)
    global Nstate
    Nq = getval(valNq)
    n1 = size(y0_ref,4)
    n2 = size(y0_ref,5)

    J = 2pi/n1*2pi/n2

    # "Volume" part
    @threads for h = 1:n1*n2
        g = parameters.g

        @inbounds begin
            state = states[threadid()]
            WJv¹ = state.WJv¹
            WJv² = state.WJv²
            ∂ξ∂x = state.∂ξ∂x

            # compute volume flux
            for j = 1:Nq, i = 1:Nq
                # 1. evaluate flux function at the point
                ρ    = y0_ref[i,j,1,h]
                ρu1  = y0_ref[i,j,2,h]
                ρu2  = y0_ref[i,j,3,h]
                ρθ   = y0_ref[i,j,4,h]
                u1   = ρu1/ρ
                u2   = ρu2/ρ
                Fρ   = SVector(ρu1, ρu2)
                Fρu1 = SVector(ρu1*u1 + g * ρ^2 / 2, ρu1*u2)
                Fρu2 = SVector(ρu2*u1,               ρu2*u2 + g * ρ^2 / 2)
                Fρθ  = SVector(ρθ*u1, ρθ*u2)

                # 2. Convert to contravariant coordinates and store in work array
                WJ = W[i]*W[j]*J
                WJv¹[1,i,j], WJv²[1,i,j] = WJ * ∂ξ∂x * Fρ
                WJv¹[2,i,j], WJv²[2,i,j] = WJ * ∂ξ∂x * Fρu1
                WJv¹[3,i,j], WJv²[3,i,j] = WJ * ∂ξ∂x * Fρu2
                WJv¹[4,i,j], WJv²[4,i,j] = WJ * ∂ξ∂x * Fρθ
            end

            # weak derivatives
            for j = 1:Nq, i = 1:Nq
                WJ = W[i]*W[j]*J
                for s = 1:Nstate
                    adj = 0.0
                    @simd for k = 1:Nq
                        # D'[i,:]*WJv¹[:,j]
                        adj += D[k, i] * WJv¹[s, k, j]
                    end
                    @simd for k = 1:Nq
                        # D'[j,:]*WJv²[i,:]
                        adj += D[k, j] * WJv²[s, i, k]
                    end
                    dydt_ref[i,j,s,h] = adj/WJ
                end
            end
        end
    end

    return dydt_ref
end

function tendency_test2(dydt_ref, y0_ref, (parameters, numflux, W, D, M, valNq, states), t)
    global Nstate
    Nq = getval(valNq)
    n1 = size(y0_ref,4)
    n2 = size(y0_ref,5)

    J = 2pi/n1*2pi/n2

    # "Face" part
    sJ1 = 2pi/n1
    sJ2 = 2pi/n2
    @threads for h2 = 1:n2
        g = parameters.g
        @inbounds begin
            for h1 = 1:n1
                h = h1*h2
                # direction 1
                g1 = mod1(h1-1,n1)
                g2 = h2
                g = g1*g2
                normal = SVector(-1.0,0.0)
                for j = 1:Nq
                    sWJ = W[j]*sJ2
                    WJ⁻ = W[1]*W[j]*J
                    WJ⁺ = W[Nq]*W[j]*J

                    y⁻ = (ρ=y0_ref[1 ,j,1,h], ρu=SVector(y0_ref[1 ,j,2,h], y0_ref[1 ,j,3,h]), ρθ=y0_ref[1 ,j,4,h])
                    y⁺ = (ρ=y0_ref[Nq,j,1,g], ρu=SVector(y0_ref[Nq,j,2,g], y0_ref[Nq,j,3,g]), ρθ=y0_ref[Nq,j,4,g])
                    nf = numflux(normal, (y⁻, parameters), (y⁺, parameters))

                    dydt_ref[1 ,j,1,h] -= sWJ/WJ⁻ * nf.ρ
                    dydt_ref[1 ,j,2,h] -= sWJ/WJ⁻ * nf.ρu[1]
                    dydt_ref[1 ,j,3,h] -= sWJ/WJ⁻ * nf.ρu[2]
                    dydt_ref[1 ,j,4,h] -= sWJ/WJ⁻ * nf.ρθ

                    dydt_ref[Nq,j,1,g] += sWJ/WJ⁺ * nf.ρ
                    dydt_ref[Nq,j,2,g] += sWJ/WJ⁺ * nf.ρu[1]
                    dydt_ref[Nq,j,3,g] += sWJ/WJ⁺ * nf.ρu[2]
                    dydt_ref[Nq,j,4,g] += sWJ/WJ⁺ * nf.ρθ
                end
                # direction 2
                g1 = h1
                g2 = mod1(h2-1,n2)
                g = g1*g2
                normal = SVector(0.0,-1.0)
                for i = 1:Nq
                    sWJ = W[i]*sJ1
                    WJ⁻ = W[i]*W[1]*J
                    WJ⁺ = W[i]*W[Nq]*J

                    y⁻ = (ρ=y0_ref[i,1 ,1,h], ρu=SVector(y0_ref[i,1 ,2,h], y0_ref[i,1 ,3,h]), ρθ=y0_ref[i,1 ,4,h])
                    y⁺ = (ρ=y0_ref[i,Nq,1,g], ρu=SVector(y0_ref[i,Nq,2,g], y0_ref[i,Nq,3,g]), ρθ=y0_ref[i,Nq,4,g])
                    nf = numflux(normal, (y⁻, parameters), (y⁺, parameters))

                    dydt_ref[i,1 ,1,h] -= sWJ/WJ⁻ * nf.ρ
                    dydt_ref[i,1 ,2,h] -= sWJ/WJ⁻ * nf.ρu[1]
                    dydt_ref[i,1 ,3,h] -= sWJ/WJ⁻ * nf.ρu[2]
                    dydt_ref[i,1 ,4,h] -= sWJ/WJ⁻ * nf.ρθ

                    dydt_ref[i,Nq,1,g] += sWJ/WJ⁺ * nf.ρ
                    dydt_ref[i,Nq,2,g] += sWJ/WJ⁺ * nf.ρu[1]
                    dydt_ref[i,Nq,3,g] += sWJ/WJ⁺ * nf.ρu[2]
                    dydt_ref[i,Nq,4,g] += sWJ/WJ⁺ * nf.ρθ
                end
            end
        end
    end

    return dydt_ref
end

function tendency_test3(dydt_ref, y0_ref, (parameters, numflux, W, D, M, valNq, states), t)
    global Nstate
    Nq = getval(valNq)
    n1 = size(y0_ref,4)
    n2 = size(y0_ref,5)

    # apply filter
    for h2 = 1:n2
        @inbounds begin
            state = states[threadid()]
            scratch = state.scratch
            for h1 = 1:n1
                for j in 1:Nq, i in 1:Nq
                    for s = 1:Nstate
                        scratch[i,j,s] = 0.0
                        for k = 1:Nq
                            scratch[i,j,s] += M[i,k] * dydt_ref[k,j,s,h1,h2]
                        end
                    end
                end
                for j in 1:Nq, i in 1:Nq
                    for s = 1:Nstate
                        dydt_ref[i,j,s,h1,h2] = 0.0
                        for k = 1:Nq
                            dydt_ref[i,j,s,h1,h2] += M[j,k] * scratch[i,k,s]
                        end
                    end
                end
            end
        end
    end

    return dydt_ref
end

function tendency_ref!(dydt_ref, y0_ref, (parameters, numflux, W, D, M, valNq, states), t)
    global Nstate
    Nq = getval(valNq)
    n1 = size(y0_ref,4)
    n2 = size(y0_ref,5)

    J = 2pi/n1*2pi/n2

    # "Volume" part
    @threads for h = 1:n1*n2
        g = parameters.g
        @inbounds begin
            state = states[threadid()]
            WJv¹ = state.WJv¹
            WJv² = state.WJv²
            ∂ξ∂x = state.∂ξ∂x

            # compute volume flux
            for j = 1:Nq, i = 1:Nq
                # 1. evaluate flux function at the point
                ρ = y0_ref[i,j,1,h]
                ρu1 = y0_ref[i,j,2,h]
                ρu2 = y0_ref[i,j,3,h]
                ρθ = y0_ref[i,j,4,h]
                u1 = ρu1/ρ
                u2 = ρu2/ρ
                Fρ = SVector(ρu1, ρu2)
                Fρu1 = SVector(ρu1*u1 + g * ρ^2 / 2, ρu1*u2)
                Fρu2 = SVector(ρu2*u1,               ρu2*u2 + g * ρ^2 / 2)
                Fρθ = SVector(ρθ*u1, ρθ*u2)

                # 2. Convert to contravariant coordinates and store in work array

                WJ = W[i]*W[j]*J
                WJv¹[1,i,j], WJv²[1,i,j] = WJ * ∂ξ∂x * Fρ
                WJv¹[2,i,j], WJv²[2,i,j] = WJ * ∂ξ∂x * Fρu1
                WJv¹[3,i,j], WJv²[3,i,j] = WJ * ∂ξ∂x * Fρu2
                WJv¹[4,i,j], WJv²[4,i,j] = WJ * ∂ξ∂x * Fρθ
            end

            # weak derivatives
            for j = 1:Nq, i = 1:Nq
                WJ = W[i]*W[j]*J
                for s = 1:Nstate
                    adj = 0.0
                    for k = 1:Nq
                        # D'[i,:]*WJv¹[:,j]
                        adj += D[k, i] * WJv¹[s, k, j]
                    end
                    for k = 1:Nq
                        # D'[j,:]*WJv²[i,:]
                        adj += D[k, j] * WJv²[s, i, k]
                    end
                    dydt_ref[i,j,s,h] = adj/WJ
                end
            end
        end
    end

    # "Face" part
    sJ1 = 2pi/n1
    sJ2 = 2pi/n2
    @threads for h2 = 1:n2
        @inbounds begin
            for h1 = 1:n1
                h = h1*h2
                # direction 1
                g1 = mod1(h1-1,n1)
                g2 = h2
                g = g1*g2
                normal = SVector(-1.0,0.0)
                for j = 1:Nq
                    sWJ = W[j]*sJ2
                    WJ⁻ = W[1]*W[j]*J
                    WJ⁺ = W[Nq]*W[j]*J

                    y⁻ = (ρ=y0_ref[1 ,j,1,h], ρu=SVector(y0_ref[1 ,j,2,h], y0_ref[1 ,j,3,h]), ρθ=y0_ref[1 ,j,4,h])
                    y⁺ = (ρ=y0_ref[Nq,j,1,g], ρu=SVector(y0_ref[Nq,j,2,g], y0_ref[Nq,j,3,g]), ρθ=y0_ref[Nq,j,4,g])
                    nf = numflux(normal, (y⁻, parameters), (y⁺, parameters))

                    dydt_ref[1 ,j,1,h] -= sWJ/WJ⁻ * nf.ρ
                    dydt_ref[1 ,j,2,h] -= sWJ/WJ⁻ * nf.ρu[1]
                    dydt_ref[1 ,j,3,h] -= sWJ/WJ⁻ * nf.ρu[2]
                    dydt_ref[1 ,j,4,h] -= sWJ/WJ⁻ * nf.ρθ

                    dydt_ref[Nq,j,1,g] += sWJ/WJ⁺ * nf.ρ
                    dydt_ref[Nq,j,2,g] += sWJ/WJ⁺ * nf.ρu[1]
                    dydt_ref[Nq,j,3,g] += sWJ/WJ⁺ * nf.ρu[2]
                    dydt_ref[Nq,j,4,g] += sWJ/WJ⁺ * nf.ρθ
                end
                # direction 2
                g1 = h1
                g2 = mod1(h2-1,n2)
                g = g1*g2
                normal = SVector(0.0,-1.0)
                for i = 1:Nq
                    sWJ = W[i]*sJ1
                    WJ⁻ = W[i]*W[1]*J
                    WJ⁺ = W[i]*W[Nq]*J

                    y⁻ = (ρ=y0_ref[i,1 ,1,h], ρu=SVector(y0_ref[i,1 ,2,h], y0_ref[i,1 ,3,h]), ρθ=y0_ref[i,1 ,4,h])
                    y⁺ = (ρ=y0_ref[i,Nq,1,g], ρu=SVector(y0_ref[i,Nq,2,g], y0_ref[i,Nq,3,g]), ρθ=y0_ref[i,Nq,4,g])
                    nf = numflux(normal, (y⁻, parameters), (y⁺, parameters))

                    dydt_ref[i,1 ,1,h] -= sWJ/WJ⁻ * nf.ρ
                    dydt_ref[i,1 ,2,h] -= sWJ/WJ⁻ * nf.ρu[1]
                    dydt_ref[i,1 ,3,h] -= sWJ/WJ⁻ * nf.ρu[2]
                    dydt_ref[i,1 ,4,h] -= sWJ/WJ⁻ * nf.ρθ

                    dydt_ref[i,Nq,1,g] += sWJ/WJ⁺ * nf.ρ
                    dydt_ref[i,Nq,2,g] += sWJ/WJ⁺ * nf.ρu[1]
                    dydt_ref[i,Nq,3,g] += sWJ/WJ⁺ * nf.ρu[2]
                    dydt_ref[i,Nq,4,g] += sWJ/WJ⁺ * nf.ρθ
                end
            end
        end
    end

    # apply filter
    @threads for h2 = 1:n2
        @inbounds begin
            state = states[threadid()]
            scratch = state.scratch
            for h1 = 1:n1
                for j in 1:Nq, i in 1:Nq
                    for s = 1:Nstate
                        scratch[i,j,s] = 0.0
                        for k = 1:Nq
                            scratch[i,j,s] += M[i,k] * dydt_ref[k,j,s,h1,h2]
                        end
                    end
                end
                for j in 1:Nq, i in 1:Nq
                    for s = 1:Nstate
                        dydt_ref[i,j,s,h1,h2] = 0.0
                        for k = 1:Nq
                            dydt_ref[i,j,s,h1,h2] += M[j,k] * scratch[i,k,s]
                        end
                    end
                end
            end
        end
    end

    return dydt_ref
end

tendency_ref!(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);
@timev tendency_ref!(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);

#__itt_resume()
#@time for n = 1:1
#    tendency_ref!(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);
#end
#__itt_pause()

#tendency_test1(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);
#tendency_test2(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);
#tendency_test3(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);
#Profile.clear_malloc_data()
#@timev tendency_test3(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);

#__itt_resume()
#@time for n = 1:1
#    tendency!(dydt_ref, y0_ref, (parameters, numflux, W, D, M, Val(Nq), tendency_states), 0.0);
#end
#__itt_pause()

#prob_ref = ODEProblem(tendency_ref!, y0_ref, (0.0, 200.0), (parameters, numflux, W, D, M, Val(Nq), tendency_states))

#__sde_start()
#__itt_resume()

#sol_ref = @time solve(prob_ref, SSPRK33(), dt = 0.02)

#__itt_pause()
#__sde_stop()

#dydt_reshaped = reshape(parent(dydt),(Nq,Nq,4,n1,n2))
#dydt_ref ≈ dydt_reshaped

#=
using Plots
ENV["GKSwstype"] = "nul"

anim = @animate for u in sol.u
    heatmap(u.ρθ, clim = (-1, 1), color = :balance)
end
mp4(anim, joinpath(@__DIR__, "bickleyjet_dg_$numflux_name.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
png(plot(Es), joinpath(@__DIR__, "energy_dg_$numflux_name.png"))
=#

#=
# # figpath = joinpath(figure_save_directory, "posterior_$(param)_T_$(T)_w_$(ω_true).png")
# linkfig(figpath)
function linkfig(figpath)
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url =
            "artifact://" * join(split(figpath, '/')[(end - 3):end], '/')
        alt = split(splitdir(figpath)[2], '.')[1]
        @info "Linking Figure: $artifact_url"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end
=#

