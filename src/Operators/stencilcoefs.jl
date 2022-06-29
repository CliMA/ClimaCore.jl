# Wrapper for a set of coefficients and their corresponding bandwidths (the
# lower bandwidth `lbw` and the upper bandwidth `ubw`).
# The bandwidths are stored as type parameters to avoid allocating unnecessary
# memory and to ensure that all the StencilCoefs in a Field have identical
# bandwidths.
# For simplicity, coefficients outside the matrix represented by a Field of
# StencilCoefs can be set to 0 (or, rather, zero(T) for coefficient type T).
struct StencilCoefs{lbw, ubw, C <: NTuple}
    coefs::C
end

function StencilCoefs{lbw, ubw}(coefs::NTuple{N, T}) where {lbw, ubw, N, T}
    if !(
        (lbw isa Integer && ubw isa Integer) ||
        (lbw isa PlusHalf && ubw isa PlusHalf)
    )
        error("Invalid stencil bandwidths $lbw and $ubw (inconsistent types)")
    end
    if N != ubw - lbw + 1
        error("Stencil bandwidth ($(ubw - lbw + 1)) and number of stencil \
              coefficients ($N) are not equal")
    end
    if N == 0 # no reason to support edge case of an empty stencil
        error("Stencil cannot be empty")
    end
    if !isconcretetype(T) # must be compatible with DataLayouts
        error("Stencil coefficients must all have the same concrete type")
    end
    return StencilCoefs{lbw, ubw, typeof(coefs)}(coefs)
end

bandwidths(::Type{<:StencilCoefs{lbw, ubw}}) where {lbw, ubw} = (lbw, ubw)

bandwidth(::Type{<:StencilCoefs{lbw, ubw}}) where {lbw, ubw} = ubw - lbw + 1

Base.eltype(::Type{StencilCoefs{lbw, ubw, C}}) where {lbw, ubw, C} = eltype(C)

Base.@propagate_inbounds Base.getindex(sc::StencilCoefs, i) = sc.coefs[i]

Base.map(f, a::StencilCoefs{lbw, ubw}) where {lbw, ubw} =
    StencilCoefs{lbw, ubw}(map(f, a.coefs))
Base.map(
    f,
    a::StencilCoefs{lbw, ubw},
    b::StencilCoefs{lbw, ubw},
) where {lbw, ubw} = StencilCoefs{lbw, ubw}(map(f, a.coefs, b.coefs))
Base.map(f, a::StencilCoefs{lbwa}, b::StencilCoefs{lbwb}) where {lbwa, lbwb} =
    lbwa <= lbwb ? map_unaligned_coefs(f, a, b) : map_unaligned_coefs(f, b, a)

# TODO: Should this be a generated function?
function map_unaligned_coefs(f, a, b)
    lbwa, ubwa = bandwidths(typeof(a))
    lbwb, ubwb = bandwidths(typeof(b))
    zeroa = zero(eltype(a))
    zerob = zero(eltype(b))
    f_zeroa = coefb -> f(zeroa, coefb)
    f_zerob = coefa -> f(coefa, zerob)
    if ubwa < lbwb # lbwa < ubwa < lbwb < ubwb
        zerof = f(zeroa, zerob)
        coefs1 = map(f_zerob, a.coefs)
        coefs2 = ntuple(_ -> zerof, lbwb - ubwa - 1)
        coefs3 = map(f_zeroa, b.coefs)
        ubw = ubwb
    elseif ubwa < ubwb # lbwa <= lbwb <= ubwa < ubwb
        coefs1 = map(f_zerob, a.coefs[1:(lbwb - lbwa)])
        coefs2 =
            map(f, a.coefs[(lbwb - lbwa + 1):end], b.coefs[1:(ubwa - lbwb + 1)])
        coefs3 = map(f_zeroa, b.coefs[(ubwa - lbwb + 2):end])
        ubw = ubwb
    else # lbwa <= lbwb < ubwb <= ubwa
        coefs1 = map(f_zerob, a.coefs[1:(lbwb - lbwa)])
        coefs2 = map(f, a.coefs[(lbwb - lbwa + 1):(ubwb - lbwa + 1)], b.coefs)
        coefs3 = map(f_zerob, a.coefs[(ubwb - lbwa + 2):end])
        ubw = ubwa
    end
    return StencilCoefs{lbwa, ubw}((coefs1..., coefs2..., coefs3...))
end

@inline Base.:(==)(
    a::StencilCoefs{lbw, ubw},
    b::StencilCoefs{lbw, ubw},
) where {lbw, ubw} = a.coefs == b.coefs

# Automatically map arithmetic functions over stencil coefficients. Makes it so
# that broadcasting over stencil fields is just like broadcasting over matrices;
# e.g., `stencil_field1.^2 .+ stencil_field2 .* scalar_field` should just work.
# TODO: Give stencil fields their own broadcast style, so that this happens for
# arbitrary functions. Better yet, store stencils as matrices using a new data
# structure, instead of using fields to store them as vectors of StencilCoefs.
for op in (:+, :-)
    @eval begin
        import Base: $op
        ($op)(a::StencilCoefs) = map($op, a)
    end
end
for op in (:+, :-, :*, :/, :÷, :\, :^, :%)
    @eval begin
        import Base: $op
        ($op)(a::StencilCoefs, b::StencilCoefs) = map($op, a, b)
        ($op)(a::StencilCoefs, b) = map(c -> ($op)(c, b), a)
        ($op)(a, b::StencilCoefs) = map(c -> ($op)(a, c), b)
    end
end
