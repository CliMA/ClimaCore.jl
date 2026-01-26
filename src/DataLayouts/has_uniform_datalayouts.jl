@inline function first_datalayout_in_bc(args::Tuple, rargs...)
    idx = unrolled_findfirst(Base.Fix2(isa, AbstractData), args)
    return isnothing(idx) ? nothing : args[idx]
end

@inline first_datalayout_in_bc(bc::Base.Broadcast.Broadcasted) =
    first_datalayout_in_bc(bc.args)

@inline _has_uniform_datalayouts_args(start, args::Tuple, rargs...) =
    unrolled_all(args) do arg
        _has_uniform_datalayouts(start, arg, rargs...)
    end
@inline function _has_uniform_datalayouts(
    start,
    bc::Base.Broadcast.Broadcasted,
)
    return _has_uniform_datalayouts_args(start, bc.args)
end
for DL in (
    :IJKFVH,
    :IJFH,
    :IJHF,
    :IFH,
    :IHF,
    :DataF,
    :IJF,
    :IF,
    :VF,
    :VIJFH,
    :VIJHF,
    :VIFH,
    :VIHF,
)
    @eval begin
        @inline _has_uniform_datalayouts(::$(DL), ::$(DL)) = true
    end
end
@inline _has_uniform_datalayouts(_, x::AbstractData) = false
@inline _has_uniform_datalayouts(_, x) = true

"""
    has_uniform_datalayouts
Find the first datalayout in the broadcast expression (BCE),
and compares against every other datalayout in the BCE. Returns
 - `true` if the broadcasted object has only a single kind of datalayout (e.g. VF,VF, VIJFH,VIJFH)
 - `false` if the broadcasted object has multiple kinds of datalayouts (e.g. VIJFH, VIFH)
Note: a broadcasted object can have different _types_,
      e.g., `VIFJH{Float64}` and `VIFJH{Tuple{Float64,Float64}}`
      but not different kinds, e.g., `VIFJH{Float64}` and `VF{Float64}`.
"""
function has_uniform_datalayouts end

@inline has_uniform_datalayouts(bc::Base.Broadcast.Broadcasted) =
    _has_uniform_datalayouts_args(first_datalayout_in_bc(bc), bc.args)

@inline has_uniform_datalayouts(bc::AbstractData) = true
