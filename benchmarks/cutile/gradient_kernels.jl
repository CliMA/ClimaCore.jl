#=
CUDA.jl contenders for the spectral-element gradient benchmark
(no cuTile dependency in this file):

  - `launch_grad_cuda_ref!`: hand-written, flop-matched replication of the
    strong-form Gradient contraction, coalesced along `v`.
  - `launch_copy_floor!`: bandwidth floor with one read and `nwrites` writes
    per point (the gradient is `nwrites = 2`) and zero flops.
  - `gradient_weight`: the fused 16x32 "KronGEMM" weight used by the cuTile
    contender, built from ClimaCore's differentiation matrix.

Parent-array layout is VIJFH: input (Nv, Nq, Nq, 1, Nh), output (Nv, Nq, Nq, 2, Nh),
with the covariant components (∂₁f, ∂₂f) in the F dimension.
=#

import CUDA
import LinearAlgebra
import StaticArrays: SMatrix

"""
    gradient_weight(D::SMatrix{Nq, Nq, FT})

Fused weight for computing both covariant gradient components as a single
GEMM on the contiguous reshapes F3 = (Nv, Nq², Nh), O3 = (Nv, 2Nq², Nh):

    O3[:, :, h] = F3[:, :, h] * Wt

With flattened indices p = k + Nq*(l-1) (input) and c = i + Nq*(j-1) + Nq²*(F-1)
(output), the strong-form gradient
∂₁f[i, j] = Σₖ D[i, k] f[k, j]
∂₂f[i, j] = Σₗ D[j, l] f[i, l]
corresponds to W = vcat(kron(I, D), kron(D, I)) (Julia `kron` puts the second
factor on the fast index), and Wt = W'.
"""
function gradient_weight(D::SMatrix{Nq, Nq, FT}) where {Nq, FT}
    Dm = Matrix(D)
    Id = Matrix{FT}(LinearAlgebra.I, Nq, Nq)
    W = vcat(LinearAlgebra.kron(Id, Dm), LinearAlgebra.kron(Dm, Id)) # (2Nq²)×(Nq²)
    return Matrix(W') # (Nq²)×(2Nq²)
end

# One thread per (v, i, j, h) point; blockIdx().y enumerates the Nq² nodes of
# the slab so that consecutive threads are consecutive in `v` (coalesced).
function grad_cuda_ref_kernel!(out, f, D, ::Val{Nq}) where {Nq}
    v = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    q = CUDA.blockIdx().y
    h = CUDA.blockIdx().z
    Nv = size(f, 1)
    if v <= Nv
        i = (q - 1) % Nq + 1
        j = (q - 1) ÷ Nq + 1
        FT = eltype(out)
        g1 = zero(FT)
        g2 = zero(FT)
        @inbounds for k in 1:Nq
            g1 = muladd(D[i, k], f[v, k, j, 1, h], g1)
            g2 = muladd(D[j, k], f[v, i, k, 1, h], g2)
        end
        @inbounds out[v, i, j, 1, h] = g1
        @inbounds out[v, i, j, 2, h] = g2
    end
    return nothing
end

function copy_floor_kernel!(out, f, nwrites, ::Val{Nq}) where {Nq}
    v = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    q = CUDA.blockIdx().y
    h = CUDA.blockIdx().z
    Nv = size(f, 1)
    if v <= Nv
        i = (q - 1) % Nq + 1
        j = (q - 1) ÷ Nq + 1
        @inbounds x = f[v, i, j, 1, h]
        for w in 1:nwrites
            @inbounds out[v, i, j, w, h] = x
        end
    end
    return nothing
end

function pointwise_launch_dims(f, Nq)
    Nv = size(f, 1)
    Nh = size(f, 5)
    threads = min(nextpow(2, Nv), 256)
    blocks = (cld(Nv, threads), Nq * Nq, Nh)
    return (threads, blocks)
end

function launch_grad_cuda_ref!(out, f, D::SMatrix{Nq, Nq}) where {Nq}
    (threads, blocks) = pointwise_launch_dims(f, Nq)
    CUDA.@cuda threads = threads blocks = blocks grad_cuda_ref_kernel!(
        out,
        f,
        D,
        Val(Nq),
    )
    return nothing
end

function launch_copy_floor!(out, f, ::Val{Nq}, nwrites) where {Nq}
    (threads, blocks) = pointwise_launch_dims(f, Nq)
    CUDA.@cuda threads = threads blocks = blocks copy_floor_kernel!(
        out,
        f,
        nwrites,
        Val(Nq),
    )
    return nothing
end
