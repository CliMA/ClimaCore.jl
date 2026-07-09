import UnrolledUtilities

"""
    soa_broadcast!(f, out::Field, args::Field...)

!!! warning "Experimental"
    Experimental, intentionally-restricted fast path; the interface may change or be
    removed without notice.

Apply the point-wise function `f` to `args` and write the result into the pre-existing
field `out` in a single device pass (struct-of-arrays). `f` maps the per-point scalar
values of `args` to either a scalar (for a scalar `out`) or a `NamedTuple` matching
`eltype(out)`. `out` and all `args` must share the same space, and `args` must be scalar
fields.

This is a fast-compiling stand-in for `Base.copyto!` of the equivalent broadcast. It
indexes the underlying arrays linearly, avoiding the `CartesianIndex` code generation that
makes many-call-site point-wise kernels (e.g. quadrature integrals) compile slowly on the
GPU. It is restricted to uniform same-space, unmasked, point-wise use and is not a general
broadcast.
"""
function soa_broadcast!(f::F, out::Field, args::Vararg{Field, N}) where {F, N}
    pns = propertynames(out)
    outs =
        isempty(pns) ? (parent(out),) :
        UnrolledUtilities.unrolled_map(nm -> parent(getproperty(out, nm)), pns)
    ins = UnrolledUtilities.unrolled_map(parent, args)
    n = length(ins[1])
    _soa_copyto!(DataLayouts.device_dispatch(parent(out)), f, outs, ins, n)
    return out
end

# CPU path: a plain serial loop. The GPU method is added in ClimaCoreCUDAExt.
function _soa_copyto!(
    ::DataLayouts.ToCPU,
    f::F,
    outs::NTuple{Nf, Any},
    ins::NTuple{Nin, Any},
    n::Integer,
) where {F, Nf, Nin}
    @inbounds for i in 1:n
        # `map` (not `ntuple(_, Val(N))`) so the input extraction unrolls and
        # stays type-stable for N > 10 heterogeneous array types.
        nt = f(map(a -> a[i], ins)...)
        ntuple(k -> (outs[k][i] = nt[k]; nothing), Val(Nf))
    end
    return nothing
end
