using Base.Threads
import ClimaCore.Spaces

function spaceconfig(::Val{Nq}) where {Nq}
    quad = Spaces.Quadratures.GLL{Nq}()
    ξ, W = Spaces.Quadratures.quadrature_points(Float64, quad)
    D = Spaces.Quadratures.differentiation_matrix(Float64, quad)
    return (ξ, W, D)
end

# construct the coordinate array
function coordinates(::Val{Nq}, n1, n2) where {Nq}
    (ξ, W, D) = spaceconfig(Val(Nq))
    X = Array{Float64}(undef, (Nq, Nq, 2, n1, n2))
    for h2 in 1:n2, h1 in 1:n1
        x1_lo = 2pi * (2h1 - 2 - n1) / n1
        x1_hi = 2pi * (2h1 - n1) / n1
        x2_lo = 2pi * (2h2 - 2 - n2) / n2
        x2_hi = 2pi * (2h2 - n2) / n2
        for j in 1:Nq, i in 1:Nq
            X[i, j, 1, h1, h2] = (1 - ξ[i]) / 2 * x1_lo + (1 + ξ[i]) / 2 * x1_hi
            X[i, j, 2, h1, h2] = (1 - ξ[j]) / 2 * x2_lo + (1 + ξ[j]) / 2 * x2_hi
        end
    end
    return X
end

const Nstate = 4

function init_y0_ref!(y0_ref, X, ::Val{Nq}, parameters) where {Nq}
    @assert Nq == size(X, 1) == size(X, 2) == size(y0_ref, 1) == size(y0_ref, 2)
    n1 = size(X, 4)
    n2 = size(X, 5)
    @assert size(y0_ref, 4) == n1
    @assert size(y0_ref, 5) == n2

    @threads for h2 in 1:n2
        for h1 in 1:n1
            @inbounds for j in 1:Nq
                @simd for i in 1:Nq
                    x = (x1 = X[i, j, 1, h1, h2], x2 = X[i, j, 2, h1, h2])
                    y = init_state(x, parameters)
                    y0_ref[i, j, 1, h1, h2] = y.ρ
                    y0_ref[i, j, 2, h1, h2] = y.ρu.u1
                    y0_ref[i, j, 3, h1, h2] = y.ρu.u2
                    y0_ref[i, j, 4, h1, h2] = y.ρθ
                end
            end
        end
    end
    return y0_ref
end

function init_y0_ref(X, ::Val{Nq}, parameters) where {Nq}
    global Nstate
    n1 = size(X, 4)
    n2 = size(X, 5)
    y0_ref = Array{Float64}(undef, (Nq, Nq, Nstate, n1, n2))
    init_y0_ref!(y0_ref, X, Val(Nq), parameters)
    return y0_ref
end

struct TendencyState{DT, WJT, ST}
    ∂ξ∂x::DT
    WJv¹::WJT
    WJv²::WJT
    scratch::ST

    function TendencyState(n1, n2, ::Val{Nq}) where {Nq}
        global Nstate
        ∂ξ∂x = @SMatrix [n1/(2pi) 0; 0 n2/(2pi)]
        WJv¹ =
            MArray{Tuple{Nstate, Nq, Nq}, Float64, 3, Nstate * Nq * Nq}(undef)
        WJv² =
            MArray{Tuple{Nstate, Nq, Nq}, Float64, 3, Nstate * Nq * Nq}(undef)
        scratch =
            MArray{Tuple{Nq, Nq, Nstate}, Float64, 3, Nq * Nq * Nstate}(undef)
        return new{typeof(∂ξ∂x), typeof(WJv¹), typeof(scratch)}(
            ∂ξ∂x,
            WJv¹,
            WJv²,
            scratch,
        )
    end
end

function init_tendency_states(n1, n2, ::Val{Nq}) where {Nq}
    global Nstate
    return [TendencyState(n1, n2, Val(Nq)) for _ in 1:nthreads()]
end

getval(::Val{V}) where {V} = V

function volume_ref!(dydt_ref, y0_ref, (n1, n2, parameters, valNq, states), t)
    global Nstate
    # specialize on Nq
    Nq = getval(valNq)
    (_, W, D) = spaceconfig(Val(Nq))
    J = 2pi / n1 * 2pi / n2

    # "Volume" part
    @threads for h2 in 1:n2
        @inbounds begin
            g = parameters.g
            state = states[threadid()]
            WJv¹ = state.WJv¹
            WJv² = state.WJv²
            ∂ξ∂x = state.∂ξ∂x

            for h1 in 1:n1
                # compute volume flux
                for j in 1:Nq, i in 1:Nq
                    # 1. evaluate flux function at the point
                    y = (
                        ρ = y0_ref[i, j, 1, h1, h2],
                        ρu = Cartesian12Vector(
                            y0_ref[i, j, 2, h1, h2],
                            y0_ref[i, j, 3, h1, h2],
                        ),
                        ρθ = y0_ref[i, j, 4, h1, h2],
                    )
                    F = flux(y, parameters)
                    Fρ = F.ρ
                    Fρu1 = SVector(F.ρu[1, 1], F.ρu[1, 2])
                    Fρu2 = SVector(F.ρu[2, 1], F.ρu[2, 2])
                    Fρθ = F.ρθ

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
                        adj = 0.0
                        @simd for k in 1:Nq
                            # D'[i,:]*WJv¹[:,j]
                            adj += D[k, i] * WJv¹[s, k, j]
                            # D'[j,:]*WJv²[i,:]
                            adj += D[k, j] * WJv²[s, i, k]
                        end
                        dydt_ref[i, j, s, h1, h2] = adj / WJ
                    end
                end
            end
        end
    end

    return dydt_ref
end


function add_face_ref!(dydt_ref, y0_ref, (n1, n2, parameters, valNq), t)
    Nq = getval(valNq)
    (ξ, W, D) = spaceconfig(Val(Nq))
    J = 2pi / n1 * 2pi / n2

    # "Face" part
    sJ1 = 2pi / n1
    sJ2 = 2pi / n2

    @threads for h2 in 1:n2
        @inbounds begin
            for h1 in 1:n1
                # direction 1
                m1 = mod1(h1 - 1, n1)
                m2 = h2
                normal = SVector(-1.0, 0.0)
                for j in 1:Nq
                    sWJ = W[j] * sJ2
                    WJ⁻ = W[1] * W[j] * J
                    WJ⁺ = W[Nq] * W[j] * J

                    y⁻ = (
                        ρ = y0_ref[1, j, 1, h1, h2],
                        ρu = SVector(
                            y0_ref[1, j, 2, h1, h2],
                            y0_ref[1, j, 3, h1, h2],
                        ),
                        ρθ = y0_ref[1, j, 4, h1, h2],
                    )
                    y⁺ = (
                        ρ = y0_ref[Nq, j, 1, m1, m2],
                        ρu = SVector(
                            y0_ref[Nq, j, 2, m1, m2],
                            y0_ref[Nq, j, 3, m1, m2],
                        ),
                        ρθ = y0_ref[Nq, j, 4, m1, m2],
                    )
                    nf = roeflux(normal, (y⁻, parameters), (y⁺, parameters))

                    dydt_ref[1, j, 1, h1, h2] -= sWJ / WJ⁻ * nf.ρ
                    dydt_ref[1, j, 2, h1, h2] -= sWJ / WJ⁻ * nf.ρu[1]
                    dydt_ref[1, j, 3, h1, h2] -= sWJ / WJ⁻ * nf.ρu[2]
                    dydt_ref[1, j, 4, h1, h2] -= sWJ / WJ⁻ * nf.ρθ

                    dydt_ref[Nq, j, 1, m1, m2] += sWJ / WJ⁺ * nf.ρ
                    dydt_ref[Nq, j, 2, m1, m2] += sWJ / WJ⁺ * nf.ρu[1]
                    dydt_ref[Nq, j, 3, m1, m2] += sWJ / WJ⁺ * nf.ρu[2]
                    dydt_ref[Nq, j, 4, m1, m2] += sWJ / WJ⁺ * nf.ρθ
                end

                # direction 2
                m1 = h1
                m2 = mod1(h2 - 1, n2)
                normal = SVector(0.0, -1.0)
                for i in 1:Nq
                    sWJ = W[i] * sJ1
                    WJ⁻ = W[i] * W[1] * J
                    WJ⁺ = W[i] * W[Nq] * J

                    y⁻ = (
                        ρ = y0_ref[i, 1, 1, h1, h2],
                        ρu = SVector(
                            y0_ref[i, 1, 2, h1, h2],
                            y0_ref[i, 1, 3, h1, h2],
                        ),
                        ρθ = y0_ref[i, 1, 4, h1, h2],
                    )
                    y⁺ = (
                        ρ = y0_ref[i, Nq, 1, m1, m2],
                        ρu = SVector(
                            y0_ref[i, Nq, 2, m1, m2],
                            y0_ref[i, Nq, 3, m1, m2],
                        ),
                        ρθ = y0_ref[i, Nq, 4, m1, m2],
                    )
                    nf = roeflux(normal, (y⁻, parameters), (y⁺, parameters))

                    dydt_ref[i, 1, 1, h1, h2] -= sWJ / WJ⁻ * nf.ρ
                    dydt_ref[i, 1, 2, h1, h2] -= sWJ / WJ⁻ * nf.ρu[1]
                    dydt_ref[i, 1, 3, h1, h2] -= sWJ / WJ⁻ * nf.ρu[2]
                    dydt_ref[i, 1, 4, h1, h2] -= sWJ / WJ⁻ * nf.ρθ

                    dydt_ref[i, Nq, 1, m1, m2] += sWJ / WJ⁺ * nf.ρ
                    dydt_ref[i, Nq, 2, m1, m2] += sWJ / WJ⁺ * nf.ρu[1]
                    dydt_ref[i, Nq, 3, m1, m2] += sWJ / WJ⁺ * nf.ρu[2]
                    dydt_ref[i, Nq, 4, m1, m2] += sWJ / WJ⁺ * nf.ρθ
                end
            end
        end
    end

    return dydt_ref
end
