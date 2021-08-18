import RecursiveArrayTools, DiffEqBase

# for compatibility with OrdinaryDiffEq
# Based on ApproxFun definitions
#  https://github.com/SciML/RecursiveArrayTools.jl/blob/6e779acb321560c75e27739a89ae553cd0f332f1/src/init.jl#L8-L12
# and
#  https://discourse.julialang.org/t/shallow-water-equations-with-differentialequations-jl/2691/16?u=simonbyrne

# This one is annoying
#  https://github.com/SciML/OrdinaryDiffEq.jl/blob/181dcf265351ed3c02437c89a8d2af3f6967fa85/src/initdt.jl#L80
Base.any(f, field::Field) = any(f, parent(field))

Base.muladd(x, field::Field) = muladd.(x, field)

Base.similar(field::F, ::Type{F}) where {F <: Field} = similar(field)

Base.vec(field::Field) = vec(parent(field))


RecursiveArrayTools.recursive_unitless_eltype(field::Field) = typeof(field)

RecursiveArrayTools.recursive_unitless_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_unitless_bottom_eltype(parent(field))

RecursiveArrayTools.recursive_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_bottom_eltype(parent(field))

function RecursiveArrayTools.recursivecopy!(dest::F, src::F) where {F <: Field}
    copy!(parent(dest), parent(src))
    return dest
end

RecursiveArrayTools.recursivecopy(field::Field) = copy(field)
# avoid call to deepcopy
RecursiveArrayTools.recursivecopy(a::AbstractArray{<:Field}) =
    map(RecursiveArrayTools.recursivecopy, a)


@inline RecursiveArrayTools.unpack(
    bc::Base.Broadcast.Broadcasted{
        RecursiveArrayTools.ArrayPartitionStyle{Style},
    },
    i,
) where {Style <: Fields.FieldStyle} = Base.Broadcast.broadcasted(
    bc.f,
    RecursiveArrayTools.unpack_args(i, bc.args)...,
)


function DiffEqBase.calculate_residuals!(
    out::Fields.Field,
    ũ::Fields.Field,
    u₀::Fields.Field,
    u₁::Fields.Field,
    α,
    ρ,
    internalnorm,
    t,
)
    DiffEqBase.calculate_residuals!(
        parent(out),
        parent(ũ),
        parent(u₀),
        parent(u₁),
        α,
        ρ,
        internalnorm,
        t,
    )
end

@inline function calculate_residuals!(out, ũ, u₀, u₁, α, ρ, internalnorm, t)
    @. out = calculate_residuals(ũ, u₀, u₁, α, ρ, internalnorm, t)
    nothing
end


# Play nice with DiffEq ArrayPartition
DiffEqBase.UNITLESS_ABS2(field::Field) =
    sum(DiffEqBase.UNITLESS_ABS2, parent(field))
DiffEqBase.recursive_length(field::Field) = 1
DiffEqBase.NAN_CHECK(field::Field) = DiffEqBase.NAN_CHECK(parent(field))
