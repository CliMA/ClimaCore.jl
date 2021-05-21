push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using LinearAlgebra
using Logging: global_logger
using UnPack, StaticArrays

import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore: slab
import ClimateMachineCore.Operators
using ClimateMachineCore.Geometry
import ClimateMachineCore.Geometry: Abstract2DPoint
using ClimateMachineCore.RecursiveOperators
using ClimateMachineCore.RecursiveOperators: rdiv, rmap


const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
)


domain = Domains.RectangleDomain(
    x1min = -2π,
    x1max = 2π,
    x2min = -2π,
    x2max = 2π,
    x1periodic = true,
    x2periodic = true,
)

n1, n2 = 16, 16
Nq = 4
Nqh = 7
discretization = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
grid_topology = Topologies.GridTopology(discretization)
quad = Meshes.Quadratures.GLL{Nq}()
mesh = Meshes.Mesh2D(grid_topology, quad)


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

y0 = init_state.(Fields.coordinate_field(mesh), Ref(parameters))

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


function rhs!(dydt, y, (parameters, numflux), t)
    Nh = Topologies.nlocalelems(y)

    F = flux.(y, Ref(parameters))
    # TODO: get this to work
    #   F = Base.Broadcast.broadcasted(flux, y, Ref(parameters))
    Operators.slab_weak_divergence!(dydt, F)

    Operators.add_numerical_flux_internal!(numflux, dydt, y, parameters)

    # 6. Solve for final result
    dydt_data = Fields.field_values(dydt)
    dydt_data .= rdiv.(dydt_data, mesh.local_geometry.WJ)

    # 7. cutoff filter
    #=
    M = Meshes.Quadratures.cutoff_filter_matrix(
        Float64,
        mesh.quadrature_style,
        3,
    )
    Operators.tensor_product!(dydt_data, M)
    =#
    return dydt
end

dydt = Fields.Field(similar(Fields.field_values(y0)), mesh)
rhs!(dydt, y0, (parameters, roeflux), 0.0);


# "Reference" implementation: just operates on Arrays

# data layout: i, j, k, n1, n2
#  n1,n2 topology
#  i,j,k datalayouts
X = reshape(parent(Fields.coordinate_field(mesh)), (Nq, Nq, 2, n1, n2))
Y = Array{Float64}(undef, (Nq, Nq, 4, n1, n2))

function init_Y!(Y, X, parameters)
    Nq, _, _, n1, n2 = size(X)
    for h2 in 1:n2, h1 in 1:n1
        for j in 1:Nq, i in 1:Nq
            x = (x1 = X[i, j, 1, h1, h2], x2 = X[i, j, 2, h1, h2])
            y = init_state(x, parameters)
            Y[i, j, 1, h1, h2] = y.ρ
            Y[i, j, 2, h1, h2] = y.ρu.u1
            Y[i, j, 3, h1, h2] = y.ρu.u2
            Y[i, j, 4, h1, h2] = y.ρθ
        end
    end
    return Y
end

init_Y!(Y, X, parameters)

Nqf = 3
_, W = Meshes.Quadratures.quadrature_points(Float64, quad)
D = Meshes.Quadratures.differentiation_matrix(Float64, quad)
M = Meshes.Quadratures.cutoff_filter_matrix(Float64, quad, Nqf)

dYdt = similar(Y)

Geometry.:⊗(u::SVector, v::SVector) = u * v'

getval(::Val{V}) where {V} = V

function tendency_ref!(dYdt, Y, (parameters, W, D, M, valNq), t)
    # specialize on Nq
    # allocate per thread?
    Nq = getval(valNq)
    Nstate = 4
    WJv¹ = MArray{Tuple{Nstate, Nq, Nq}, Float64, 3, Nstate * Nq * Nq}(undef)
    WJv² = MArray{Tuple{Nstate, Nq, Nq}, Float64, 3, Nstate * Nq * Nq}(undef)
    n1 = size(Y, 4)
    n2 = size(Y, 5)

    J = 2pi / n1 * 2pi / n2
    ∂ξ∂x = @SMatrix [n1/(2pi) 0; 0 n2/(2pi)]

    g = parameters.g

    # "Volume" part
    for h2 in 1:n2, h1 in 1:n1
        # compute volume flux
        for j in 1:Nq, i in 1:Nq
            # 1. evaluate flux function at the point
            ρ = Y[i, j, 1, h1, h2]
            ρu1 = Y[i, j, 2, h1, h2]
            ρu2 = Y[i, j, 3, h1, h2]
            ρθ = Y[i, j, 4, h1, h2]
            u1 = ρu1 / ρ
            u2 = ρu2 / ρ
            Fρ = SVector(ρu1, ρu2)
            Fρu1 = SVector(ρu1 * u1 + g * ρ^2 / 2, ρu1 * u2)
            Fρu2 = SVector(ρu2 * u1, ρu2 * u2 + g * ρ^2 / 2)
            Fρθ = SVector(ρθ * u1, ρθ * u2)

            # 2. Convert to contravariant coordinates and store in work array

            WJ = W[i] * W[j] * J
            WJv¹[1, i, j], WJv²[1, i, j] = WJ * ∂ξ∂x * Fρ
            WJv¹[2, i, j], WJv²[2, i, j] = WJ * ∂ξ∂x * Fρu1
            WJv¹[3, i, j], WJv²[3, i, j] = WJ * ∂ξ∂x * Fρu2
            WJv¹[4, i, j], WJv²[4, i, j] = WJ * ∂ξ∂x * Fρθ
        end

        # weak derivatives
        for j in 1:Nq, i in 1:Nq
            WJ = W[i] * W[j] * J
            for s in 1:Nstate
                t = 0.0
                for k in 1:Nq
                    # D'[i,:]*WJv¹[:,j]
                    t += D[k, i] * WJv¹[s, k, j]
                end
                for k in 1:Nq
                    # D'[j,:]*WJv²[i,:]
                    t += D[k, j] * WJv²[s, i, k]
                end
                dYdt[i, j, s, h1, h2] = t / WJ
            end
        end
    end


    # "Face" part
    sJ1 = 2pi / n1
    sJ2 = 2pi / n2
    for h2 in 1:n2, h1 in 1:n1
        # direction 1
        g1 = mod1(h1 - 1, n1)
        g2 = h2
        normal = SVector(-1.0, 0.0)
        for j in 1:Nq
            sWJ = W[j] * sJ2
            WJ⁻ = W[1] * W[j] * J
            WJ⁺ = W[Nq] * W[j] * J

            y⁻ = (
                ρ = Y[1, j, 1, h1, h2],
                ρu = SVector(Y[1, j, 2, h1, h2], Y[1, j, 3, h1, h2]),
                ρθ = Y[1, j, 4, h1, h2],
            )
            y⁺ = (
                ρ = Y[Nq, j, 1, g1, g2],
                ρu = SVector(Y[Nq, j, 2, g1, g2], Y[Nq, j, 3, g1, g2]),
                ρθ = Y[Nq, j, 4, g1, g2],
            )
            nf = roeflux(normal, (y⁻, parameters), (y⁺, parameters))

            dYdt[1, j, 1, h1, h2] -= sWJ / WJ⁻ * nf.ρ
            dYdt[1, j, 2, h1, h2] -= sWJ / WJ⁻ * nf.ρu[1]
            dYdt[1, j, 3, h1, h2] -= sWJ / WJ⁻ * nf.ρu[2]
            dYdt[1, j, 4, h1, h2] -= sWJ / WJ⁻ * nf.ρθ

            dYdt[Nq, j, 1, g1, g2] += sWJ / WJ⁺ * nf.ρ
            dYdt[Nq, j, 2, g1, g2] += sWJ / WJ⁺ * nf.ρu[1]
            dYdt[Nq, j, 3, g1, g2] += sWJ / WJ⁺ * nf.ρu[2]
            dYdt[Nq, j, 4, g1, g2] += sWJ / WJ⁺ * nf.ρθ
        end
        # direction 2
        g1 = h1
        g2 = mod1(h2 - 1, n2)
        normal = SVector(0.0, -1.0)
        for i in 1:Nq
            sWJ = W[i] * sJ1
            WJ⁻ = W[i] * W[1] * J
            WJ⁺ = W[i] * W[Nq] * J

            y⁻ = (
                ρ = Y[i, 1, 1, h1, h2],
                ρu = SVector(Y[i, 1, 2, h1, h2], Y[i, 1, 3, h1, h2]),
                ρθ = Y[i, 1, 4, h1, h2],
            )
            y⁺ = (
                ρ = Y[i, Nq, 1, g1, g2],
                ρu = SVector(Y[i, Nq, 2, g1, g2], Y[i, Nq, 3, g1, g2]),
                ρθ = Y[i, Nq, 4, g1, g2],
            )
            nf = roeflux(normal, (y⁻, parameters), (y⁺, parameters))

            dYdt[i, 1, 1, h1, h2] -= sWJ / WJ⁻ * nf.ρ
            dYdt[i, 1, 2, h1, h2] -= sWJ / WJ⁻ * nf.ρu[1]
            dYdt[i, 1, 3, h1, h2] -= sWJ / WJ⁻ * nf.ρu[2]
            dYdt[i, 1, 4, h1, h2] -= sWJ / WJ⁻ * nf.ρθ

            dYdt[i, Nq, 1, g1, g2] += sWJ / WJ⁺ * nf.ρ
            dYdt[i, Nq, 2, g1, g2] += sWJ / WJ⁺ * nf.ρu[1]
            dYdt[i, Nq, 3, g1, g2] += sWJ / WJ⁺ * nf.ρu[2]
            dYdt[i, Nq, 4, g1, g2] += sWJ / WJ⁺ * nf.ρθ
        end
    end


    # apply filter
    # temporary storage: match the layout of dYdt?
    #=
    scratch = MArray{Tuple{Nstate, Nq, Nq}, Float64, 3, Nstate * Nq * Nq}(undef)
    for h2 = 1:n2, h1 = 1:n1
        for j in 1:Nq, i in 1:Nq
            for s = 1:Nstate
                scratch[s,i,j] = 0.0
                for k = 1:Nq
                    scratch[s,i,j] += M[i,k] * dYdt[k,j,s,h1,h2]
                end
            end
        end
        for j in 1:Nq, i in 1:Nq
            for s = 1:Nstate
                dYdt[i,j,s,h1,h2] = 0.0
                for k = 1:Nq
                    dYdt[i,j,s,h1,h2] += M[j,k] * scratch[s,i,k]
                end
            end
        end
    end
    =#
    return dYdt
end

tendency_ref!(dYdt, Y, (parameters, W, D, M, Val(Nq)), 0.0);

dYdt_ref = reshape(parent(dydt), (Nq, Nq, 4, n1, n2))
@assert dYdt ≈ dYdt_ref


using BenchmarkTools


Nqs = 2:7
Ts = Float64[]
Rs = Float64[]

for Nq in Nqs
    global quad, mesh, y0, dydt, X, Y, dYdt, Nqf, W, D, M
    quad = Meshes.Quadratures.GLL{Nq}()
    mesh = Meshes.Mesh2D(grid_topology, quad)
    y0 = init_state.(Fields.coordinate_field(mesh), Ref(parameters))
    dydt = Fields.Field(similar(Fields.field_values(y0)), mesh)
    push!(Ts, @belapsed rhs!($dydt, $y0, ($parameters, $roeflux), 0.0))

    X = reshape(parent(Fields.coordinate_field(mesh)), (Nq, Nq, 2, n1, n2))
    Y = Array{Float64}(undef, (Nq, Nq, 4, n1, n2))
    init_Y!(Y, X, parameters)
    dYdt = similar(Y)

    Nqf = 3
    _, W = Meshes.Quadratures.quadrature_points(Float64, quad)
    D = Meshes.Quadratures.differentiation_matrix(Float64, quad)
    M = Meshes.Quadratures.cutoff_filter_matrix(Float64, quad, Nqf)

    push!(
        Rs,
        @belapsed tendency_ref!(
            $dYdt,
            $Y,
            ($parameters, $W, $D, $M, $(Val(Nq))),
            0.0,
        )
    )
end

using Plots
ENV["GKSwstype"] = "nul"

plt = plot(ylims = (0, Inf), xlabel = "Nq", ylabel = "Time (ms)")
plot!(plt, Nqs, 1e3 .* Ts, label = "ClimateMachineCore")
plot!(plt, Nqs, 1e3 .* Rs, label = "Reference")

png(plt, joinpath(@__DIR__, "times.png"))
