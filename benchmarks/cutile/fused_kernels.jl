#=
Host-side pieces of the single-kernel ("fully fused") cuTile spectral
operators, with no cuTile dependency so they can be validated on the CPU:

  - `spectral_weights`: the four Nq² × Nq² Kronecker weights that turn the
    strong gradient and the weak divergence into row-batched GEMMs on the
    (Nv, Nq², Nh) reshapes of VIJFH parents.
  - `horizontal_metrics`: the contravariant metric block g¹¹, g¹² (= g²¹),
    g²² and the weighted Jacobian WJ as contiguous (Mv, Nq², Nh) arrays, with
    Mv = 1 when the metrics are the same on every level.
  - `cpu_scalar_laplacian!`, `cpu_pressure_gradient!`: plain-array reference
    implementations of exactly the arithmetic the cuTile kernels perform.
    The GPU kernels in fused_kernels_cutile.jl are these loops with the
    per-element GEMMs on tensor cores; test_fused_cpu.jl checks them
    against ClimaCore's operators.

Flattened node index p = i + Nq (j - 1) for node (i, j), and a scalar field
becomes F3[v, p, h].

Strong gradient (covariant components):
    ∂₁f = F3 * Wg1,  Wg1 = kron(I, D)'      ∂₁f[i, j] = Σₖ D[i, k] f[k, j]
    ∂₂f = F3 * Wg2,  Wg2 = kron(D, I)'      ∂₂f[i, j] = Σₗ D[j, l] f[i, l]

Weak divergence of a vector with WJ-weighted contravariant components
w¹, w²:  θ = -(WJ)⁻¹ (Σₖ D[k, i] w¹[k, j] + Σₗ D[l, j] w²[i, l]), so
    θ = (W¹ * Wd1 + W² * Wd2) ./ WJ,  Wd1 = -kron(I, D),  Wd2 = -kron(D, I)
with the sign folded into the weights.

Scalar Laplacian ∇²χ = wdiv(grad χ):
    g = grad χ;  uⁱ = gⁱʲ gⱼ;  θ = ((WJ u¹) * Wd1 + (WJ u²) * Wd2) ./ WJ
=#

import LinearAlgebra
import StaticArrays: SMatrix
import ClimaCore: Fields, Spaces

"""
    spectral_weights(D::SMatrix{Nq, Nq})

The GEMM weights `(; Wg1, Wg2, Wd1, Wd2)` described at the top of this file,
as `Nq² × Nq²` matrices with the element type of `D`.
"""
function spectral_weights(D::SMatrix{Nq, Nq, FT}) where {Nq, FT}
    Dm = Matrix(D)
    Id = Matrix{FT}(LinearAlgebra.I, Nq, Nq)
    kID = LinearAlgebra.kron(Id, Dm)
    kDI = LinearAlgebra.kron(Dm, Id)
    return (; Wg1 = Matrix(kID'), Wg2 = Matrix(kDI'), Wd1 = -kID, Wd2 = -kDI)
end

# (Nv, Nq², Nh) copy of component `f` of a VIJFH parent, or of its first
# `levels` levels.
function component3(A::AbstractArray{<:Any, 5}, f::Int; levels = size(A, 1))
    (_, Nqi, Nqj, _, Nh) = size(A)
    return reshape(A[1:levels, :, :, f, :], levels, Nqi * Nqj, Nh)
end

"""
    metrics_are_level_uniform(space; rtol = 8 * eps(FT))

Whether the horizontal metric block of `space` is the same on every level up
to a level-wise factor: g¹¹, g¹², g²² independent of `v`, and WJ[v, ·, h] a
`v`-dependent multiple of WJ[1, ·, h]. True for a shallow-atmosphere sphere
without topography, where the level factor is the vertical Jacobian and
cancels in the weak divergence, so the kernels can read one level of
metrics instead of Nv.
"""
function metrics_are_level_uniform(space; rtol = nothing)
    lg = Fields.local_geometry_field(space)
    g = Array(parent(Fields.field_values(lg.gⁱʲ)))
    wj = Array(parent(Fields.field_values(lg.WJ)))
    FT = eltype(g)
    tol = isnothing(rtol) ? 8 * eps(FT) : rtol
    relvar(A) =
        maximum(abs.(A .- A[1:1, :, :, :])) / max(maximum(abs, A), floatmin(FT))
    for k in (1, 2, 5)
        relvar(g[:, :, :, k, :]) <= tol || return false
    end
    ratio = wj[:, :, :, 1, :] ./ wj[1:1, :, :, 1, :]
    return maximum(abs.(ratio .- ratio[:, 1:1, 1:1, :])) <= tol * maximum(abs, ratio)
end

"""
    horizontal_metrics(space; levels)

`(; g11, g12, g22, WJ)` as contiguous `(levels, Nq², Nh)` arrays on the
device of `space`, `levels` being 1 or Nv. The metric tensor is symmetric, so
g²¹ = g¹²; the horizontal-vertical entries do not enter a `Covariant12Vector`
conversion and are not needed.
"""
function horizontal_metrics(space; levels)
    lg = Fields.local_geometry_field(space)
    pg = parent(Fields.field_values(lg.gⁱʲ))
    pwj = parent(Fields.field_values(lg.WJ))
    Nv = size(pg, 1)
    levels in (1, Nv) || error("levels must be 1 or Nv = $Nv; got $levels")
    return (;
        g11 = component3(pg, 1; levels),
        g12 = component3(pg, 2; levels),
        g22 = component3(pg, 5; levels),
        WJ = component3(pwj, 1; levels),
    )
end

##### CPU reference implementations (the kernel arithmetic, per element)

"""
    cpu_scalar_laplacian!(out, x, w, m; p, ρ, weighted, scale, accumulate)

`out .= (accumulate ? out : 0) .+ scale .* ∇²(χ)` on `(Nv, Nq², Nh)` arrays,
where `χ = x` or, when `p` is given, `χ = (x + p) / ρ` (the specific-enthalpy
prologue of the energy hyperdiffusion), and `∇²χ = wdiv(ρ grad χ)` when
`weighted`. `w` from [`spectral_weights`](@ref), `m` from
[`horizontal_metrics`](@ref).
"""
function cpu_scalar_laplacian!(
    out,
    x,
    w,
    m;
    p = nothing,
    ρ = nothing,
    weighted = false,
    scale = one(eltype(out)),
    accumulate = false,
)
    Nh = size(x, 3)
    for h in 1:Nh
        χ = x[:, :, h]
        if !isnothing(p)
            χ = (χ .+ p[:, :, h]) ./ ρ[:, :, h]
        end
        g1 = χ * w.Wg1
        g2 = χ * w.Wg2
        if weighted
            g1 = g1 .* ρ[:, :, h]
            g2 = g2 .* ρ[:, :, h]
        end
        g11 = m.g11[:, :, h]
        g12 = m.g12[:, :, h]
        g22 = m.g22[:, :, h]
        wj = m.WJ[:, :, h]
        u1 = wj .* (g11 .* g1 .+ g12 .* g2)
        u2 = wj .* (g12 .* g1 .+ g22 .* g2)
        lap = (u1 * w.Wd1 .+ u2 * w.Wd2) ./ wj
        out[:, :, h] .= (accumulate ? out[:, :, h] : 0) .+ scale .* lap
    end
    return out
end

"""
    cpu_pressure_gradient!(out1, out2, p, K, Φ, ρ, w; scale, accumulate)

Covariant components of `scale * (grad(p) / ρ + grad(K + Φ))`, added to
`out1`, `out2` when `accumulate`. With `scale = -1` and `accumulate = true`
this is the horizontal pressure-gradient force on `Yₜ.c.uₕ`.
"""
function cpu_pressure_gradient!(
    out1,
    out2,
    p,
    K,
    Φ,
    ρ,
    w;
    scale = one(eltype(out1)),
    accumulate = false,
)
    Nh = size(p, 3)
    for h in 1:Nh
        ps = p[:, :, h]
        b = K[:, :, h] .+ Φ[:, :, h]
        ρs = ρ[:, :, h]
        du1 = (ps * w.Wg1) ./ ρs .+ b * w.Wg1
        du2 = (ps * w.Wg2) ./ ρs .+ b * w.Wg2
        out1[:, :, h] .= (accumulate ? out1[:, :, h] : 0) .+ scale .* du1
        out2[:, :, h] .= (accumulate ? out2[:, :, h] : 0) .+ scale .* du2
    end
    return (out1, out2)
end
