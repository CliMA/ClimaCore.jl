#=
Single-kernel cuTile spectral operators: each production expression that
contains a horizontal gradient runs as ONE kernel, with the pointwise work
(metric conversion, weights, the enthalpy prologue, the tendency update) as
prologue and epilogue around per-element GEMMs. Nothing is materialized
between the two contractions of a Laplacian, so the intermediate-traffic
penalty that made the split cuTile path lose on the Laplacian is gone.

  scalar_laplacian_kernel!   OUT = [OUT +] scale * wdiv([ρ] grad(χ)),
                             χ = X or (X + P) / R (energy hyperdiffusion)
  pressure_gradient_kernel!  UT = [UT +] scale * (grad(P) / R + grad(KE + PHI))

One CTA per (v-tile, element): grid = (cld(Nv, tv), 1, Nh). Tiles are
(tv, Nq²) slabs; the v dimension is zero-padded on load and clipped on
store, and every operation is row-wise (per v) or a contraction over the
node index, so padded rows never contaminate valid ones. Metric tiles are
(mtv, Nq²) with mtv = tv, or mtv = 1 broadcast down the tile when the
metrics are level-uniform (`metrics_are_level_uniform`), which removes four
field reads per Laplacian.

Arrays reach the kernels as (Nv, Nq², Nh) strided views of VIJFH parents
(`tile3`), including components of `FieldVector` blocks, so tendencies are
updated in place with no scratch fields. Weights come from
`spectral_weights`, metrics from `horizontal_metrics` (fused_kernels.jl).
=#

import CUDA
import cuTile
import ClimaCore: Fields, Quadratures, Spaces
const ct = cuTile

include(joinpath(@__DIR__, "fused_kernels.jl"))

"""
    tile3(A::AbstractArray{T, 5}, f = 1)

`(Nv, Nq², Nh)` cuTile view of component `f` of a VIJFH parent `A`, which may
be a `CuArray` or a strided `SubArray` of one (a `FieldVector` block). No
copy: the node dimensions (v, i, j) of a VIJFH parent are contiguous, so
the view has strides `(1, Nv, stride_h)`.
"""
function tile3(A::AbstractArray{T, 5}, f::Int = 1) where {T}
    (Nv, Nqi, Nqj, Nf, Nh) = size(A)
    1 <= f <= Nf || error("component $f out of range 1:$Nf")
    s = strides(A)
    (s[1] == 1 && s[2] == Nv && s[3] == Nv * Nqi) ||
        error("VIJFH parent is not contiguous in (v, i, j); strides $s")
    ptr = reinterpret(Ptr{T}, pointer(A) + (f - 1) * s[4] * sizeof(T))
    sizes = (Nv, Nqi * Nqj, Nh)
    strd = (1, Nv, s[5])
    I = all(x -> x <= typemax(Int32), (sizes..., strd...)) ? Int32 : Int64
    return ct.TileArray(ptr, I.(sizes), I.(strd))
end

tile3(field::Fields.Field, f::Int = 1) =
    tile3(parent(Fields.field_values(field)), f)

# Load the (tv, K) slab of v-tile `bv` of element `h`.
@inline function load_slab(A, bv, h, tv::Int, K::Int)
    t = ct.load(
        A;
        index = (bv, 1, h),
        shape = (tv, K, 1),
        padding_mode = ct.PaddingMode.Zero,
    )
    return reshape(t, (tv, K))
end

@inline function store_slab(A, bv, h, tile, tv::Int, K::Int)
    ct.store(A; index = (bv, 1, h), tile = reshape(tile, (tv, K, 1)))
    return nothing
end

@inline load_weight(W, K::Int) = ct.load(W; index = (1, 1), shape = (K, K))

function scalar_laplacian_kernel!(
    X::ct.TileArray{T, 3},
    P::ct.TileArray{T, 3},
    R::ct.TileArray{T, 3},
    G11::ct.TileArray{T, 3},
    G12::ct.TileArray{T, 3},
    G22::ct.TileArray{T, 3},
    WJ::ct.TileArray{T, 3},
    Wg1::ct.TileArray{T, 2},
    Wg2::ct.TileArray{T, 2},
    Wd1::ct.TileArray{T, 2},
    Wd2::ct.TileArray{T, 2},
    OUT::ct.TileArray{T, 3},
    scale::T,
    tv::Int,
    K::Int,
    mtv::Int,
    energy::Bool,
    weighted::Bool,
    accumulate::Bool,
) where {T}
    bv = ct.bid(1)
    h = ct.bid(3)
    mv = mtv == 1 ? one(bv) : bv

    x = load_slab(X, bv, h, tv, K)
    # `r` is ρ when the prologue or the weight needs it; otherwise a stand-in
    # that keeps the variable defined on every path (the branches are
    # compile-time constants, so nothing is loaded that is not used).
    r = (energy || weighted) ? load_slab(R, bv, h, tv, K) : x
    if energy
        x = (x .+ load_slab(P, bv, h, tv, K)) ./ r
    end

    # Strong gradient, covariant components.
    zero_acc = zeros(T, tv, K)
    g1 = muladd(x, load_weight(Wg1, K), zero_acc)
    g2 = muladd(x, load_weight(Wg2, K), zero_acc)
    if weighted
        g1 = g1 .* r
        g2 = g2 .* r
    end

    # WJ-weighted contravariant components.
    g11 = load_slab(G11, mv, h, mtv, K)
    g12 = load_slab(G12, mv, h, mtv, K)
    g22 = load_slab(G22, mv, h, mtv, K)
    wj = load_slab(WJ, mv, h, mtv, K)
    u1 = wj .* (g11 .* g1 .+ g12 .* g2)
    u2 = wj .* (g12 .* g1 .+ g22 .* g2)

    # Weak divergence; the minus sign is folded into Wd1, Wd2.
    acc = muladd(u1, load_weight(Wd1, K), zero_acc)
    acc = muladd(u2, load_weight(Wd2, K), acc)
    out = (acc ./ wj) * scale
    if accumulate
        out = load_slab(OUT, bv, h, tv, K) .+ out
    end
    store_slab(OUT, bv, h, out, tv, K)
    return
end

function pressure_gradient_kernel!(
    P::ct.TileArray{T, 3},
    KE::ct.TileArray{T, 3},
    PHI::ct.TileArray{T, 3},
    R::ct.TileArray{T, 3},
    Wg1::ct.TileArray{T, 2},
    Wg2::ct.TileArray{T, 2},
    UT1::ct.TileArray{T, 3},
    UT2::ct.TileArray{T, 3},
    scale::T,
    tv::Int,
    K::Int,
    accumulate::Bool,
) where {T}
    bv = ct.bid(1)
    h = ct.bid(3)

    p = load_slab(P, bv, h, tv, K)
    b = load_slab(KE, bv, h, tv, K) .+ load_slab(PHI, bv, h, tv, K)
    r = load_slab(R, bv, h, tv, K)

    zero_acc = zeros(T, tv, K)
    wg1 = load_weight(Wg1, K)
    wg2 = load_weight(Wg2, K)
    du1 = muladd(p, wg1, zero_acc) ./ r .+ muladd(b, wg1, zero_acc)
    du2 = muladd(p, wg2, zero_acc) ./ r .+ muladd(b, wg2, zero_acc)
    out1 = du1 * scale
    out2 = du2 * scale
    if accumulate
        out1 = load_slab(UT1, bv, h, tv, K) .+ out1
        out2 = load_slab(UT2, bv, h, tv, K) .+ out2
    end
    store_slab(UT1, bv, h, out1, tv, K)
    store_slab(UT2, bv, h, out2, tv, K)
    return
end

##### Host side

"""
    CuTileSpectral(space; tv = 64)

Everything the fused kernels need for the horizontal operators on `space`:
GEMM weights and metrics on the device, the v-tile size `tv` (power of two,
at most `nextpow(2, Nv)`), and the metric tile size `mtv` (1 when the metrics
are level-uniform, else `tv`). Build once and keep in the model cache.
"""
struct CuTileSpectral{W, M}
    weights::W
    metrics::M
    tv::Int
    K::Int
    mtv::Int
    Nv::Int
end

function CuTileSpectral(space; tv::Int = 64)
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)
    K = Nq * Nq
    ispow2(K) || error("cuTile tile extents must be powers of two; Nq² = $K")
    Nv = Spaces.nlevels(space)
    tv = min(tv, nextpow(2, Nv))
    ispow2(tv) || error("tv must be a power of two; got $tv")
    D = Quadratures.differentiation_matrix(FT, quad)
    w = spectral_weights(D)
    weights = map(CUDA.CuArray, w)
    uniform = metrics_are_level_uniform(space)
    metrics = horizontal_metrics(space; levels = uniform ? 1 : Nv)
    return CuTileSpectral(weights, metrics, tv, K, uniform ? 1 : tv, Nv)
end

grid_dims(s::CuTileSpectral, Nh) = (cld(s.Nv, s.tv), 1, Nh)

"""
    cutile_scalar_laplacian!(out, χ, s::CuTileSpectral;
                             energy = nothing, weight = nothing,
                             scale = 1, accumulate = false)

`out .= [out .+] scale .* wdiv([weight .*] grad(χ′))` in one kernel, where
`χ′ = χ`, or `(χ + p) / ρ` when `energy = (p, ρ)`. `weight` is a scalar
field; when both are given it must be the same field as `ρ` (the energy
hyperdiffusion uses ρ for both).
"""
function cutile_scalar_laplacian!(
    out,
    χ,
    s::CuTileSpectral;
    energy = nothing,
    weight = nothing,
    scale = 1,
    accumulate = false,
)
    T = eltype(parent(Fields.field_values(out)))
    X = tile3(χ)
    if !isnothing(energy)
        (p, ρ) = energy
        (isnothing(weight) || weight === ρ) ||
            error("the weight must be the same field as the energy prologue's ρ")
        P = tile3(p)
        R = tile3(ρ)
    elseif !isnothing(weight)
        P = X
        R = tile3(weight)
    else
        P = X
        R = X
    end
    (; g11, g12, g22, WJ) = s.metrics
    (; Wg1, Wg2, Wd1, Wd2) = s.weights
    Nh = size(parent(Fields.field_values(χ)), 5)
    CUDA.@cuda backend = ct blocks = grid_dims(s, Nh) scalar_laplacian_kernel!(
        X,
        P,
        R,
        g11,
        g12,
        g22,
        WJ,
        Wg1,
        Wg2,
        Wd1,
        Wd2,
        tile3(out),
        T(scale),
        ct.Constant(s.tv),
        ct.Constant(s.K),
        ct.Constant(s.mtv),
        ct.Constant(!isnothing(energy)),
        ct.Constant(!isnothing(weight)),
        ct.Constant(accumulate),
    )
    return out
end

"""
    cutile_pressure_gradient!(uₜ, p, K, Φ, ρ, s::CuTileSpectral;
                              scale = -1, accumulate = true)

`uₜ .= [uₜ .+] scale .* (grad(p) / ρ + grad(K + Φ))` in one kernel; `uₜ` is a
`Covariant12Vector` field. The defaults apply the pressure-gradient force to
the momentum tendency in place.
"""
function cutile_pressure_gradient!(
    uₜ,
    p,
    K,
    Φ,
    ρ,
    s::CuTileSpectral;
    scale = -1,
    accumulate = true,
)
    pu = parent(Fields.field_values(uₜ))
    T = eltype(pu)
    size(pu, 4) == 2 || error("uₜ must have two covariant components")
    (; Wg1, Wg2) = s.weights
    Nh = size(pu, 5)
    CUDA.@cuda backend = ct blocks = grid_dims(s, Nh) pressure_gradient_kernel!(
        tile3(p),
        tile3(K),
        tile3(Φ),
        tile3(ρ),
        Wg1,
        Wg2,
        tile3(pu, 1),
        tile3(pu, 2),
        T(scale),
        ct.Constant(s.tv),
        ct.Constant(s.K),
        ct.Constant(accumulate),
    )
    return uₜ
end
