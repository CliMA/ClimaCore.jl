@inline function first_mask_in_bc(args::Tuple)
    x1 = first_mask_in_bc(args[1])
    x1 isa DataLayouts.AbstractMask && return x1
    return first_mask_in_bc(Base.tail(args))
end

@inline first_mask_in_bc(args::Tuple{Any}) = first_mask_in_bc(args[1])
@inline first_mask_in_bc(args::Tuple{}) = nothing
@inline first_mask_in_bc(x) = nothing
@inline first_mask_in_bc(x::Field) = get_mask(axes(x))
@inline first_mask_in_bc(x::DataLayouts.AbstractMask) = x

@inline first_mask_in_bc(bc::Base.AbstractBroadcasted) =
    first_mask_in_bc(bc.args)

@inline _has_uniform_mask_args(truesofar, start, args::Tuple, rargs...) =
    truesofar &&
    _has_uniform_mask(truesofar, start, args[1], rargs...) &&
    _has_uniform_mask_args(truesofar, start, Base.tail(args), rargs...)

@inline _has_uniform_mask_args(truesofar, start, args::Tuple{Any}, rargs...) =
    truesofar && _has_uniform_mask(truesofar, start, args[1], rargs...)
@inline _has_uniform_mask_args(truesofar, _, args::Tuple{}, rargs...) =
    truesofar

@inline function _has_uniform_mask(
    truesofar,
    start,
    bc::Base.Broadcast.Broadcasted,
)
    return truesofar && _has_uniform_mask_args(truesofar, start, bc.args)
end

@inline _has_uniform_mask(
    truesofar,
    ::DataLayouts.NoMask,
    ::DataLayouts.NoMask,
) = true
@inline _has_uniform_mask(
    truesofar,
    ::DataLayouts.IJHMask,
    ::DataLayouts.IJHMask,
) = true
@inline _has_uniform_mask(
    truesofar,
    ::DataLayouts.AbstractMask,
    x::DataLayouts.AbstractMask,
) = false
@inline _has_uniform_mask(truesofar, _, x) = truesofar

"""
    has_uniform_mask
Find the first datalayout in the broadcast expression (BCE),
and compares against every other datalayout in the BCE. Returns
 - `true` if the broadcasted object has only a single kind of datalayout (e.g. VF,VF, VIJFH,VIJFH)
 - `false` if the broadcasted object has multiple kinds of datalayouts (e.g. VIJFH, VIFH)
Note: a broadcasted object can have different _types_,
      e.g., `VIFJH{Float64}` and `VIFJH{Tuple{Float64,Float64}}`
      but not different kinds, e.g., `VIFJH{Float64}` and `VF{Float64}`.
"""
function has_uniform_mask end

@inline has_uniform_mask(bc::Base.Broadcast.Broadcasted) =
    _has_uniform_mask_args(true, first_mask_in_bc(bc), bc.args)

@inline has_uniform_mask(bc::AbstractData) = true
