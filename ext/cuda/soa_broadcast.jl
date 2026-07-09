function soa_broadcast_kernel!(
    f::F,
    outs::NTuple{Nf, Any},
    ins::NTuple{Nin, Any},
    n,
) where {F, Nf, Nin}
    i = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    if i ≤ n
        # `map` over the input tuple unrolls for any arity and stays type-stable
        # for heterogeneous array types; `ntuple(_, Val(N))` falls back to a
        # runtime loop for N > 10, which boxes and triggers gpu_gc_pool_alloc.
        nt = f(map(a -> @inbounds(a[i]), ins)...)
        ntuple(k -> (@inbounds(outs[k][i] = nt[k]); Int32(0)), Val(Nf))
    end
    return nothing
end

function ClimaCore.Fields._soa_copyto!(
    ::ClimaCore.DataLayouts.ToCUDA,
    f::F,
    outs::NTuple{Nf, Any},
    ins::NTuple{Nin, Any},
    n::Integer,
) where {F, Nf, Nin}
    n > 0 || return nothing
    # A plain launch (no forced `always_inline`): forcing inlining of the per-point
    # function balloons the IR and the LLVM compile for many-call-site functions.
    kernel = CUDA.@cuda launch = false soa_broadcast_kernel!(f, outs, ins, n)
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, outs, ins, n; threads, blocks)
    return nothing
end
