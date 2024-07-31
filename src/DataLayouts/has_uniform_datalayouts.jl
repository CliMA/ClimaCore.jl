@inline function first_datalayout_in_bc(args::Tuple, rargs...)
    x1 = first_datalayout_in_bc(args[1], rargs...)
    x1 isa AbstractData && return x1
    return first_datalayout_in_bc(Base.tail(args), rargs...)
end

@inline first_datalayout_in_bc(args::Tuple{Any}, rargs...) =
    first_datalayout_in_bc(args[1], rargs...)
@inline first_datalayout_in_bc(args::Tuple{}, rargs...) = nothing
@inline first_datalayout_in_bc(x) = nothing
@inline first_datalayout_in_bc(x::AbstractData) = x

@inline first_datalayout_in_bc(bc::Base.Broadcast.Broadcasted) =
    first_datalayout_in_bc(bc.args)

@inline _has_uniform_datalayouts_args(truesofar, start, args::Tuple, rargs...) =
    truesofar &&
    _has_uniform_datalayouts(truesofar, start, args[1], rargs...) &&
    _has_uniform_datalayouts_args(truesofar, start, Base.tail(args), rargs...)

@inline _has_uniform_datalayouts_args(
    truesofar,
    start,
    args::Tuple{Any},
    rargs...,
) = truesofar && _has_uniform_datalayouts(truesofar, start, args[1], rargs...)
@inline _has_uniform_datalayouts_args(truesofar, _, args::Tuple{}, rargs...) =
    truesofar

@inline function _has_uniform_datalayouts(
    truesofar,
    start,
    bc::Base.Broadcast.Broadcasted,
)
    return truesofar && _has_uniform_datalayouts_args(truesofar, start, bc.args)
end
for DL in (:IJKFVH, :IJFH, :IFH, :DataF, :IJF, :IF, :VF, :VIJFH, :VIFH)
    @eval begin
        @inline _has_uniform_datalayouts(truesofar, ::$(DL), ::$(DL)) = true
    end
end
@inline _has_uniform_datalayouts(truesofar, _, x::AbstractData) = false
@inline _has_uniform_datalayouts(truesofar, _, x) = truesofar

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
    _has_uniform_datalayouts_args(true, first_datalayout_in_bc(bc), bc.args)

@inline has_uniform_datalayouts(bc::AbstractData) = true
