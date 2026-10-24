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

full_grid(grid::AbstractGrid) =
    grid isa Union{LevelGrid, ColumnGrid} ? grid.full_grid : grid

"""
    toggle_placeholder_grid(arg, grid)

Replace every copy of `grid` in `arg` with a [`PlaceholderGrid`](@ref), or turn
every placeholder back into `grid`. Level and column views are unwrapped, so
their full grids get toggled. This is only applied to arguments of GPU kernels
launched from a host device, and it is reversed at the start of each kernel.
"""
function toggle_placeholder_grid(arg, grid::AbstractGrid)
    return Adapt.adapt(PlaceholderGridAdaptor(full_grid(grid)), arg)
end

# Hooks extended by Fields to discover and compact grids across arguments and
# inside Tuple/NamedTuple bundles. When both 2D horizontal grids and 3D extruded
# grids appear in the same kernel launch, the first 3D extruded grid is chosen
# as the reference grid: DeviceExtrudedFiniteDifferenceGrid drops its horizontal
# grid on GPUs, so a 2D reference grid cannot restore a 3D extruded grid inside
# the kernel.
@inline arg_full_grid(_) = ()
@inline arg_full_grid(args::Union{Tuple, NamedTuple}) =
    unrolled_flatmap(arg_full_grid, values(args))

@inline has_reference_grid(_, _) = false

@inline select_reference_grid(g1, g2) =
    g1 isa AbstractExtrudedFiniteDifferenceGrid ? g1 :
    g2 isa AbstractExtrudedFiniteDifferenceGrid ? g2 :
    g1 isa PlaceholderGrid ? g2 : g1

@inline toggle_first_compact_arg(arg, _) = arg

@inline function toggle_compact_arg(arg, grid, ::Val{found}) where {found}
    if !found && has_reference_grid(arg, grid)
        return (toggle_first_compact_arg(arg, grid), Val(true))
    end
    return (toggle_placeholder_grid(arg, grid), Val(found))
end
@inline toggle_compact_arg(::Tuple{}, _, found::Val) = ((), found)
@inline function toggle_compact_arg(args::Tuple, grid, found::Val)
    (head, head_found) = toggle_compact_arg(first(args), grid, found)
    (tail, tail_found) = toggle_compact_arg(Base.tail(args), grid, head_found)
    return ((head, tail...), tail_found)
end
@inline function toggle_compact_arg(
    args::NamedTuple{names},
    grid,
    found::Val,
) where {names}
    (new_vals, next_found) = toggle_compact_arg(values(args), grid, found)
    return (NamedTuple{names}(new_vals), next_found)
end

@drop_recursion_limits toggle_compact_arg

"""
    toggle_compact_args(args...)

Modify `args` to take up as few bytes as possible, or undo that transformation.
For example, this can call [`Grids.toggle_placeholder_grid`](@ref) to eliminate
duplicate grid pointers. This is only applied to arguments of GPU kernels
launched from a host device, and it is reversed at the start of each kernel.
Specific data structures like `Field`s should extend this in their own modules.
"""
function toggle_compact_args(args...)
    grids = unrolled_flatmap(arg_full_grid, args)
    isempty(grids) && return args
    grid = unrolled_reduce(select_reference_grid, grids)
    return first(toggle_compact_arg(args, grid, Val(false)))
end
