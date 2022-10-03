"""
    ColumnIndex(ij,h)

An index into a column of a field. This can be used as an argument to `getindex`
of a `Field`, to return a field on that column.

# Example
```julia
colidx = ColumnIndex((1,1),1)
field[colidx]
```
"""
struct ColumnIndex{N}
    ij::NTuple{N, Int}
    h::Int
end

Base.@propagate_inbounds Base.getindex(field::Field, colidx::ColumnIndex) =
    column(field, colidx)

Base.@propagate_inbounds function column(
    field::SpectralElementField1D,
    colidx::ColumnIndex{1},
)
    column(field, colidx.ij[1], colidx.h)
end
Base.@propagate_inbounds function column(
    field::ExtrudedFiniteDifferenceField,
    colidx::ColumnIndex{1},
)
    column(field, colidx.ij[1], colidx.h)
end
Base.@propagate_inbounds function column(
    field::SpectralElementField2D,
    colidx::ColumnIndex{2},
)
    column(field, colidx.ij[1], colidx.ij[2], colidx.h)
end
Base.@propagate_inbounds function column(
    field::ExtrudedFiniteDifferenceField,
    colidx::ColumnIndex{2},
)
    column(field, colidx.ij[1], colidx.ij[2], colidx.h)
end

"""
    Fields.bycolumn(fn, space)

Call `fn(colidx)` to every [`ColumnIndex`](@ref) `colidx` of `space`. This can be used to apply
multiple column-wise operations in a single pass, making use of multiple threads.

# Example

```julia
∇ = GradientF2C()
div = DivergenceC2F()

bycolumn(axes(f)) do colidx
    @. ∇f[colidx] = ∇(f[colidx])
    @. df[colidx] = div(∇f[colidx])
end
```
"""
function bycolumn(fn, space::Spaces.SpectralElementSpace1D)
    Nh = Topologies.nlocalelems(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    @inbounds begin
        if enable_threading()
            Threads.@threads for h in 1:Nh
                for i in 1:Nq
                    fn(ColumnIndex((i,), h))
                end
            end
        else
            for h in 1:Nh
                for i in 1:Nq
                    fn(ColumnIndex((i,), h))
                end
            end
        end
    end
    return nothing
end
function bycolumn(fn, space::Spaces.SpectralElementSpace2D)
    Nh = Topologies.nlocalelems(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    @inbounds begin
        if enable_threading()
            Threads.@threads for h in 1:Nh
                for j in 1:Nq, i in 1:Nq
                    fn(ColumnIndex((i, j), h))
                end
            end
        else
            for h in 1:Nh
                for j in 1:Nq, i in 1:Nq
                    fn(ColumnIndex((i, j), h))
                end
            end
        end
    end
    return nothing
end
bycolumn(fn, space::Spaces.ExtrudedFiniteDifferenceSpace) =
    bycolumn(fn, space.horizontal_space)

"""
    ncolumns(::Field)
    ncolumns(::Space)

Number of columns in a given space.
"""
ncolumns(field::Field) = ncolumns(axes(field))

ncolumns(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    ncolumns(space.horizontal_space)

function ncolumns(space::Spaces.SpectralElementSpace1D)
    Nh = Topologies.nlocalelems(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    return Nh * Nq
end
function ncolumns(space::Spaces.SpectralElementSpace2D)
    Nh = Topologies.nlocalelems(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    return Nh * Nq * Nq
end

# potential TODO:
# - define a ColumnIndices type, make it work with https://github.com/JuliaFolds/FLoops.jl
