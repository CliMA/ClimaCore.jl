"""
    PlaceholderGrid()

Singleton value that represents an [`AbstractGrid`](@ref). Replacing grids with
`PlaceholderGrid`s allows larger broadcasts to be passed into GPU kernels
without hitting a parameter memory limit, since the `Field`s in a broadcast
expression all contain pointers to the same grid data as the broadcast's
destination. Once inside a GPU kernel, every `PlaceholderGrid` is replaced by
its original grid, with redundant pointers stored in each thread's registers.
"""
struct PlaceholderGrid <: AbstractGrid end

struct PlaceholderGridAdaptor{G <: AbstractGrid}
    grid::G
end

# DataLayouts do not contain grids, so they do not need to be rebuilt by adapt.
Adapt.adapt_structure(::PlaceholderGridAdaptor, data::DataLayouts.DataLayout) = data

# Adapt.jl's default method for Broadcasted does not rebuild instantiated axes.
Adapt.adapt_structure(to::PlaceholderGridAdaptor, bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.Broadcasted(
        bc.style,
        Adapt.adapt(to, bc.f),
        Adapt.adapt(to, bc.args),
        Adapt.adapt(to, bc.axes),
    )

# Avoid ambiguity for grids that have adapt_structure methods by forwarding to
# adapt_storage, which replaces the stored grid data with a PlaceholderGrid.
for G in (
    :SpectralElementGrid1D,
    :SpectralElementGrid2D,
    :MultiPointGrid,
    :FiniteDifferenceGrid,
    :ExtrudedFiniteDifferenceGrid,
)
    @eval Adapt.adapt_structure(to::PlaceholderGridAdaptor, grid::$G) =
        Adapt.adapt_storage(to, grid)
end

# Replace every grid with the same type as the adaptor's grid, leaving grids of
# other types unchanged. Distinct grids with the same type cannot be skipped, as
# an adaptor that compares grids by value (=== or ==) would not be type-stable.
Adapt.adapt_storage((; grid)::PlaceholderGridAdaptor, other_grid::AbstractGrid) =
    other_grid isa typeof(grid) ? PlaceholderGrid() : other_grid
Adapt.adapt_storage((; grid)::PlaceholderGridAdaptor, ::PlaceholderGrid) = grid

"""
    toggle_placeholder_grid(arg, grid)

Replace every copy of `grid` in `arg` with a [`PlaceholderGrid`](@ref), or turn
every placeholder back into `grid`. Level and column views are unwrapped, so
their full grids get toggled. This is only applied to arguments of GPU kernels
launched from a host device, and it is reversed at the start of each kernel.
"""
function toggle_placeholder_grid(arg, grid::AbstractGrid)
    full_grid = grid isa Union{LevelGrid, ColumnGrid} ? grid.full_grid : grid
    return Adapt.adapt(PlaceholderGridAdaptor(full_grid), arg)
end

"""
    toggle_compact_args(args...)

Modify `args` to take up as few bytes as possible, or undo that transformation.
For example, this can call [`Grids.toggle_placeholder_grid`](@ref) to eliminate
duplicate grid pointers. This is only applied to arguments of GPU kernels
launched from a host device, and it is reversed at the start of each kernel.
Specific data structures like `Field`s should extend this in their own modules.
"""
toggle_compact_args(args...) = args
