import ClimateMachineCore.Meshes

function meshconfig(::Val{Nq}) where {Nq}
    quad = Meshes.Quadratures.GLL{Nq}()
    ξ, W = Meshes.Quadratures.quadrature_points(Float64, quad)
    D = Meshes.Quadratures.differentiation_matrix(Float64, quad)
    return (ξ, W, D)
end

# construct the coordinate array
function coordinates(::Val{Nq}, n1, n2) where {Nq}
    (ξ, W, D) = meshconfig(Val(Nq))
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

function init_Y!(Y, X, ::Val{Nq}, parameters) where {Nq}
    @assert Nq == size(X, 1) == size(X, 2) == size(Y, 1) == size(Y, 2)
    n1 = size(X, 4)
    n2 = size(X, 5)
    @assert size(Y, 4) == n1
    @assert size(Y, 5) == n2

    for h2 in 1:n2, h1 in 1:n1
        x1lo = h1 / (n1 + 1)
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

function init_Y(X, ::Val{Nq}, parameters) where {Nq}
    n1 = size(X, 4)
    n2 = size(X, 5)
    Nstate = 4
    Y = Array{Float64}(undef, (Nq, Nq, Nstate, n1, n2))
    init_Y!(Y, X, Val(Nq), parameters)
end

getval(::Val{V}) where {V} = V

function volume_ref!(dYdt, Y, (parameters, valNq), t)
    # specialize on Nq
    Nq = getval(valNq)
    (ξ, W, D) = meshconfig(Val(Nq))
    # allocate per thread?
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
            y = (
                ρ = Y[i, j, 1, h1, h2],
                ρu = Cartesian12Vector(Y[i, j, 2, h1, h2], Y[i, j, 3, h1, h2]),
                ρθ = Y[i, j, 4, h1, h2],
            )
            F = flux(y, parameters)
            Fρ = F.ρ
            Fρu1 = SVector(F.ρu.matrix[1, 1], F.ρu.matrix[1, 2])
            Fρu2 = SVector(F.ρu.matrix[2, 1], F.ρu.matrix[2, 2])
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
    return dYdt
end


function add_face_ref!(dYdt, Y, (parameters, valNq), t)
    Nq = getval(valNq)
    (ξ, W, D) = meshconfig(Val(Nq))
    n1 = size(Y, 4)
    n2 = size(Y, 5)

    # "Face" part
    sJ1 = 2pi / n1
    sJ2 = 2pi / n2

    J = 2pi / n1 * 2pi / n2

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
                ρu = Cartesian12Vector(Y[1, j, 2, h1, h2], Y[1, j, 3, h1, h2]),
                ρθ = Y[1, j, 4, h1, h2],
            )
            y⁺ = (
                ρ = Y[Nq, j, 1, g1, g2],
                ρu = Cartesian12Vector(
                    Y[Nq, j, 2, g1, g2],
                    Y[Nq, j, 3, g1, g2],
                ),
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
                ρu = Cartesian12Vector(Y[i, 1, 2, h1, h2], Y[i, 1, 3, h1, h2]),
                ρθ = Y[i, 1, 4, h1, h2],
            )
            y⁺ = (
                ρ = Y[i, Nq, 1, g1, g2],
                ρu = Cartesian12Vector(
                    Y[i, Nq, 2, g1, g2],
                    Y[i, Nq, 3, g1, g2],
                ),
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
    return dYdt
end
