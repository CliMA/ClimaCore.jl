module Fields

import ..slab, ..column



"""
    Field(values, grid)

A set of `values` defined at each point of a `grid`.
"""
struct Field{V <: AbstractData, G}
    values::V
    grid::G
end

slab(field::Field, args...) = Field(
    slab(getfield(field, :values), args...),
    slab(getfield(field, :grid), args...),
)

column(field::Field, args...) = Field(
    column(getfield(field, :values), args...),
    column(getfield(field, :grid), args...),
)


end # module
