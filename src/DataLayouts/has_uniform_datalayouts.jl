@inline function first_datalayout_in_bc(args::Tuple, rargs...)
    return UnrolledUtilities.unrolled_argfirst(arg -> first_datalayout_in_bc(arg, rargs...) isa AbstractData, args)
end

@inline first_datalayout_in_bc(args::Tuple{Any}, rargs...) =
    first_datalayout_in_bc(args[1], rargs...)
@inline first_datalayout_in_bc(args::Tuple{}, rargs...) = nothing
@inline first_datalayout_in_bc(x) = nothing
@inline first_datalayout_in_bc(x::AbstractData) = x

@inline first_datalayout_in_bc(bc::Base.Broadcast.Broadcasted) =
    first_datalayout_in_bc(bc.args)

@inline function _has_uniform_datalayouts(
    start,
    args::Tuple,
)
    return UnrolledUtilities.unrolled_all(arg -> _has_uniform_datalayouts(start, arg), args)
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
    _has_uniform_datalayouts(first_datalayout_in_bc(bc), bc.args)

@inline has_uniform_datalayouts(bc::AbstractData) = true
