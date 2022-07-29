@inline function ref_transform(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {Ato <: AbstractAxis{I}, Afrom <: AbstractAxis{I}} where {I, T, N}
    x
end

@inline function ref_project(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {Ato <: AbstractAxis{I}, Afrom <: AbstractAxis{I}} where {I, T, N}
    x
end

@generated function ref_transform(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
} where {Ito, Ifrom, T, N}
    errcond = false
    for n in 1:N
        i = Ifrom[n]
        if i ∉ Ito
            errcond = :($errcond || x[$n] != zero(T))
        end
    end
    vals = []
    for i in Ito
        val = :(zero(T))
        for n in 1:N
            if i == Ifrom[n]
                val = :(x[$n])
                break
            end
        end
        push!(vals, val)
    end
    quote
        Base.@_propagate_inbounds_meta
        if $errcond
            throw(InexactError(:transform, Ato, x))
        end
        @inbounds AxisVector(ato, SVector($(vals...)))
    end
end

@generated function ref_project(
    ato::Ato,
    x::AxisVector{T, Afrom, SVector{N, T}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
} where {Ito, Ifrom, T, N}
    vals = []
    for i in Ito
        val = :(zero(T))
        for n in 1:N
            if i == Ifrom[n]
                val = :(x[$n])
                break
            end
        end
        push!(vals, val)
    end
    return :(@inbounds AxisVector(ato, SVector($(vals...))))
end

@inline function ref_transform(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{I},
    Afrom <: AbstractAxis{I},
    A2 <: AbstractAxis{J},
} where {I, J, T}
    x
end

@inline function ref_project(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{I},
    Afrom <: AbstractAxis{I},
    A2 <: AbstractAxis{J},
} where {I, J, T}
    x
end

@generated function ref_transform(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
    A2 <: AbstractAxis{J},
} where {Ito, Ifrom, J, T}
    N = length(Ifrom)
    M = length(J)
    errcond = false
    for n in 1:N
        i = Ifrom[n]
        if i ∉ Ito
            for m in 1:M
                errcond = :($errcond || x[$n, $m] != zero(T))
            end
        end
    end
    vals = []
    for m in 1:M
        for i in Ito
            val = :(zero(T))
            for n in 1:N
                if i == Ifrom[n]
                    val = :(x[$n, $m])
                    break
                end
            end
            push!(vals, val)
        end
    end
    quote
        Base.@_propagate_inbounds_meta
        if $errcond
            throw(InexactError(:transform, Ato, x))
        end
        @inbounds Axis2Tensor(
            (ato, axes(x, 2)),
            SMatrix{$(length(Ito)), $M}($(vals...)),
        )
    end
end

@generated function ref_project(
    ato::Ato,
    x::Axis2Tensor{T, Tuple{Afrom, A2}},
) where {
    Ato <: AbstractAxis{Ito},
    Afrom <: AbstractAxis{Ifrom},
    A2 <: AbstractAxis{J},
} where {Ito, Ifrom, J, T}
    N = length(Ifrom)
    M = length(J)
    vals = []
    for m in 1:M
        for i in Ito
            val = :(zero(T))
            for n in 1:N
                if i == Ifrom[n]
                    val = :(x[$n, $m])
                    break
                end
            end
            push!(vals, val)
        end
    end
    return :(@inbounds Axis2Tensor(
        (ato, axes(x, 2)),
        SMatrix{$(length(Ito)), $M}($(vals...)),
    ))
end
