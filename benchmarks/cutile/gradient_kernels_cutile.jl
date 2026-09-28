#=
cuTile.jl contender: the whole gradient (both covariant components) as one
batched GEMM per element ("KronGEMM").

Contiguous reshapes of the VIJFH parents:
    F3 :: (Nv, K, Nh)   with K = Nq²   (scalar input)
    O3 :: (Nv, C, Nh)   with C = 2Nq²  (Covariant12Vector output)
and a precomputed K×C weight Wt (see `gradient_weight` in gradient_kernels.jl):
    O3[:, :, h] = F3[:, :, h] * Wt

One CTA per (v-tile, element): grid = (cld(Nv, tv), 1, Nh). The v dimension is
padded to the tile size via PaddingMode.Zero on load; the store is masked to
the array bounds by cuTile. Wt is 4 KB at Nq = 4 / Float64 and stays cached.

Only loaded when the `cutile` contender is requested, so the rest of the
benchmark runs on nodes without cuTile support.
=#

import cuTile
const ct = cuTile

function grad_krongemm_kernel!(
    F3::ct.TileArray{T, 3},
    Wt::ct.TileArray{T, 2},
    O3::ct.TileArray{T, 3},
    tv::Int,
    K::Int,
    C::Int,
) where {T}
    bv = ct.bid(1)
    h = ct.bid(3)
    f = ct.load(
        F3;
        index = (bv, 1, h),
        shape = (tv, K, 1),
        padding_mode = ct.PaddingMode.Zero,
    )
    w = ct.load(Wt; index = (1, 1), shape = (K, C))
    # NB: no TF32 conversion for Float32 (it is opt-in in cuTile); the tile
    # matmul runs at full precision, matching the correctness gate's rtol.
    acc = zeros(T, tv, C)
    acc = muladd(reshape(f, (tv, K)), w, acc)
    ct.store(O3; index = (bv, 1, h), tile = reshape(acc, (tv, C, 1)))
    return
end

function launch_grad_cutile!(F3, Wt, O3; tv, K, C)
    (Nv, _, Nh) = size(F3)
    grid = (cld(Nv, tv), 1, Nh)
    CUDA.@cuda backend = ct blocks = grid grad_krongemm_kernel!(
        F3,
        Wt,
        O3,
        ct.Constant(tv),
        ct.Constant(K),
        ct.Constant(C),
    )
    return nothing
end
