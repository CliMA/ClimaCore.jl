using StaticArrays: MArray
using ..Geometry: LocalGeometry

"""
	columnwise!(
		::ClimaComms.AbstractDevice,
		ᶜf::ᶜF,
		ᶠf::ᶠF,
		ᶜYₜ::Fields.Field,
		ᶠYₜ::Fields.Field,
		ᶜY::Fields.Field,
		ᶠY::Fields.Field,
		p, # cache
		t, # time
		Val(localmem_lg),
		Val(localmem_state)
	)

This function can be used to assign a large set of point-wise and or
vertical neighbor-wise (e.g., interpolations, divergence, gradient, etc.)
tendencies to `ᶜYₜ` and `ᶠYₜ` in the form:

```
	@. ᶜYₜ = ᶜf(ᶜY, ᶠY, p, t)
	@. ᶠYₜ = ᶠf(ᶜY, ᶠY, p, t)
```
where

 - `ᶜf(ᶜY, ᶠY, p, t)` returns a subtype of `Base.AbstractBroadcasted` for `ᶜYₜ`
   tendencies
 - `ᶠf(ᶜY, ᶠY, p, t)` returns a subtype of `Base.AbstractBroadcasted` for `ᶠYₜ`
   tendencies

This function has a few key design aspects. If on the gpu:

 - a single kernel is launched on the gpu
 - shared memory is used for `ᶜY` and `ᶠY`
 - (optionally) shared memory for the local geometry (specified via `localmem_lg`)
 - (optionally) shared memory for the state (specified via `localmem_state`)

on the cpu:

 - multi-threading is applied across columns
 - local memory is used for `ᶜY` and `ᶠY`
 - (optionally) local memory for the local geometry (specified via `localmem_lg`)
 - (optionally) local memory for the state (specified via `localmem_state`)
"""
function columnwise! end

# todo:
# We can inspect the broadcast expressions and determine which components of the
# LocalGeometry actually need to be read into local memory.

# todo:
# use KernelAbstractions.jl instead
# issue holding us back:
# https://github.com/JuliaGPU/KernelAbstractions.jl/issues/598

# TODO: can we improve the CPU performance?
function columnwise!(
    device::ClimaComms.AbstractCPUDevice,
    ᶜf::ᶜF,
    ᶠf::ᶠF,
    ᶜYₜ::Fields.Field,
    ᶠYₜ::Fields.Field,
    ᶜY::Fields.Field,
    ᶠY::Fields.Field,
    p,
    t,
    ::Val{localmem_lg} = Val(false),
    ::Val{localmem_state} = Val(false),
) where {ᶜF, ᶠF, localmem_lg, localmem_state}
    ᶜspace = axes(ᶜY)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠNv = Spaces.nlevels(ᶠspace)
    ᶜcf = Fields.coordinate_field(ᶜspace)
    (_, Ni, Nj, Nh) = size(Fields.field_values(ᶜcf))

    mask = Spaces.get_mask(axes(ᶜYₜ))
    @inbounds begin
        for h in 1:Nh
            for j in 1:Nj, i in 1:Ni
                DataLayouts.should_compute(
                    mask,
                    CartesianIndex(1, i, j, h),
                ) || continue
                for v in 1:ᶠNv
                    UI = CartesianIndex((v, i, j, h))
                    columnwise_kernel!(
                        device,
                        ᶜf,
                        ᶠf,
                        ᶜYₜ,
                        ᶠYₜ,
                        ᶜY,
                        ᶠY,
                        p,
                        t,
                        UI,
                        Val(localmem_lg),
                        Val(localmem_state),
                    )
                end
            end
        end
    end
    return nothing
end

# Canonical parent-array dimensions for a single-column layout whose values
# span Nf base types: (Nv, 1, 1, 1) with Nf inserted at the F axis (or dropped
# when there is no F axis). Layout constructors require canonically shaped
# parent arrays, so local memory must be allocated accordingly.
@inline local_mem_dims(data, Nf) = DataLayouts.add_f_dim(
    (DataLayouts.nlevels(data), 1, 1, 1),
    Nf,
    Val(DataLayouts.f_dim(data)),
)

function columnwise_kernel!(
    device,
    ᶜf,
    ᶠf,
    ᶜYₜ,
    ᶠYₜ,
    _ᶜY,
    _ᶠY,
    p,
    t,
    UI,
    ::Val{localmem_lg},
    ::Val{localmem_state},
) where {localmem_lg, localmem_state}
    ᶜY_fv = Fields.field_values(_ᶜY)
    ᶠY_fv = Fields.field_values(_ᶠY)
    FT = Spaces.undertype(axes(_ᶜY))
    ᶜTS = DataLayouts.num_basetypes(FT, eltype(ᶜY_fv))
    ᶠTS = DataLayouts.num_basetypes(FT, eltype(ᶠY_fv))
    ᶜlg = Spaces.local_geometry_data(axes(_ᶜY))
    ᶠlg = Spaces.local_geometry_data(axes(_ᶠY))
    SLG = eltype(ᶜlg)
    ᶜTS_lg = DataLayouts.num_basetypes(FT, SLG)

    ᶜui = universal_index_columnwise(device, UI, ᶜY_fv)
    ᶠui = universal_index_columnwise(device, UI, ᶠY_fv)
    colidx = Grids.ColumnIndex((ᶠui.I[2], ᶠui.I[3]), ᶠui.I[4])

    if localmem_state
        ᶜY_arr = local_mem(device, FT, Val(local_mem_dims(ᶜY_fv, ᶜTS)))
        ᶠY_arr = local_mem(device, FT, Val(local_mem_dims(ᶠY_fv, ᶠTS)))
        ᶜdata_col = rebuild_column(ᶜY_fv, ᶜY_arr)
        ᶠdata_col = rebuild_column(ᶠY_fv, ᶠY_arr)
    else
        ᶜdata_col = DataLayouts.column(ᶜY_fv, colidx.ij..., colidx.h)
        ᶠdata_col = DataLayouts.column(ᶠY_fv, colidx.ij..., colidx.h)
    end

    if localmem_lg
        ᶜlg_arr = local_mem(device, FT, Val(local_mem_dims(ᶜlg, ᶜTS_lg)))
        ᶠlg_arr = local_mem(device, FT, Val(local_mem_dims(ᶠlg, ᶜTS_lg)))
        (ᶜspace_col, ᶠspace_col) =
            column_spaces(_ᶜY, _ᶠY, ᶠui, ᶜlg_arr, ᶠlg_arr, SLG)
    else
        ᶜspace_col = Spaces.column(axes(_ᶜY), colidx)
        ᶠspace_col = Spaces.column(axes(_ᶠY), colidx)
    end

    ᶜvi = CartesianIndex(ᶜui.I[1], 1, 1, 1)
    ᶠvi = CartesianIndex(ᶠui.I[1], 1, 1, 1)

    if localmem_state
        is_valid_index_cw(ᶜY_fv, ᶜui) && (ᶜdata_col[ᶜvi] = ᶜY_fv[ᶜui])
        is_valid_index_cw(ᶠY_fv, ᶠui) && (ᶠdata_col[ᶠvi] = ᶠY_fv[ᶠui])
    end

    if localmem_lg
        ᶜlg_col = Spaces.local_geometry_data(ᶜspace_col)
        ᶠlg_col = Spaces.local_geometry_data(ᶠspace_col)
        is_valid_index_cw(ᶜY_fv, ᶜui) && (ᶜlg_col[ᶜvi] = ᶜlg[ᶜui])
        is_valid_index_cw(ᶠY_fv, ᶠui) && (ᶠlg_col[ᶠvi] = ᶠlg[ᶠui])
    end

    device_sync_threads(device)

    if is_valid_index_cw(ᶜY_fv, ᶜui)
        ᶜY = Fields.Field(ᶜdata_col, ᶜspace_col)
        ᶠY = Fields.Field(ᶠdata_col, ᶠspace_col)
        ᶜbc = ᶜf(ᶜY, ᶠY, p, t)
        (ᶜidx, ᶜhidx) = operator_inds(axes(ᶜY), ᶜui)
        ᶜval = Operators.getidx(axes(ᶜY), ᶜbc, ᶜidx, ᶜhidx)
        Fields.field_values(ᶜYₜ)[ᶜui] = ᶜval
    end
    if is_valid_index_cw(ᶠY_fv, ᶠui)
        ᶜY = Fields.Field(ᶜdata_col, ᶜspace_col)
        ᶠY = Fields.Field(ᶠdata_col, ᶠspace_col)
        ᶠbc = ᶠf(ᶜY, ᶠY, p, t)
        (ᶠidx, ᶠhidx) = operator_inds(axes(ᶠY), ᶠui)
        ᶠval = Operators.getidx(axes(ᶠY), ᶠbc, ᶠidx, ᶠhidx)
        Fields.field_values(ᶠYₜ)[ᶠui] = ᶠval
    end
    return nothing
end


__size(args::Tuple) = Tuple{args...}
__size(i::Int) = Tuple{i}

local_mem(
    device::ClimaComms.AbstractCPUDevice,
    ::Type{T},
    ::Val{dims},
) where {T, dims} = MArray{__size(dims), T}(undef)

device_sync_threads(device::ClimaComms.AbstractCPUDevice) = nothing

@inline function operator_inds(space, I)
    li = Operators.left_idx(space)
    (v, i, j, h) = I.I
    hidx = (i, j, h)
    idx = v - 1 + li
    return (idx, hidx)
end

"""
	rebuild_column(data, array)

Returns a new column datalayout, using `array` as its backing data
"""
rebuild_column(data, array::AbstractArray) =
    new_rebuild_column(eltype(data), data, array)

"""
	new_rebuild_column(::Type{S}, data, lg_arr) where {S}

Returns a new column datalayout, using `array` as its backing data
using a new type S.
"""
function new_rebuild_column(::Type{S}, data, array::AbstractArray) where {S}
    params = (; DataLayouts.shape_params(data)..., Ni = 1, Nj = 1, Nh = 1)
    return DataLayouts.layout_type(data){S, params...}(array)
end

"""
	column_lg_local_mem(space, ui, lg_arr, ::Type{SLG}) where {SLG}

Returns a new LocalGeometry datalayout, using `lg_arr` as its backing data
"""
function column_lg_local_mem(space, ui, lg_arr, ::Type{SLG}) where {SLG}
    (_, i, j, h) = ui.I
    lg = Spaces.local_geometry_data(space)
    lg_col = DataLayouts.column(lg, i, j, h)
    return new_rebuild_column(SLG, lg_col, lg_arr)
end

# TODO: this needs to be generalized for other spaces
function column_spaces(ᶜY, ᶠY, ui, ᶜlg_arr, ᶠlg_arr, ::Type{SLG}) where {SLG}
    (_, i, j, h) = ui.I
    colidx = Grids.ColumnIndex((i, j), h)
    ᶜlg_col = column_lg_local_mem(axes(ᶜY), ui, ᶜlg_arr, SLG)
    ᶠlg_col = column_lg_local_mem(axes(ᶠY), ui, ᶠlg_arr, SLG)
    col_space = Spaces.column(axes(ᶜY), colidx)
    col_grid = Spaces.grid(col_space)
    if col_grid isa Grids.ColumnGrid &&
       col_grid.full_grid isa Grids.DeviceExtrudedFiniteDifferenceGrid
        (; full_grid) = col_grid
        (; vertical_topology, global_geometry) = full_grid
        col_grid_shmem = Grids.DeviceFiniteDifferenceGrid(
            vertical_topology,
            global_geometry,
            ᶜlg_col,
            ᶠlg_col,
        )
        ᶜspace_col = Spaces.space(col_grid_shmem, Grids.CellCenter())
        ᶠspace_col = Spaces.space(col_grid_shmem, Grids.CellFace())
    elseif col_grid isa Grids.ColumnGrid &&
           col_grid.full_grid isa Grids.ExtrudedFiniteDifferenceGrid
        (; full_grid) = col_grid
        (; vertical_grid, global_geometry) = full_grid
        col_grid_shmem = Grids.FiniteDifferenceGrid(
            vertical_grid.topology,
            global_geometry,
            ᶜlg_col,
            ᶠlg_col,
        )
        ᶜspace_col = Spaces.space(col_grid_shmem, Grids.CellCenter())
        ᶠspace_col = Spaces.space(col_grid_shmem, Grids.CellFace())
    elseif col_grid isa Grids.DeviceFiniteDifferenceGrid
        col_grid_shmem = Grids.DeviceFiniteDifferenceGrid(
            col_grid.topology,
            col_grid.global_geometry,
            ᶜlg_col,
            ᶠlg_col,
        )
        ᶜspace_col = Spaces.space(col_grid_shmem, Grids.CellCenter())
        ᶠspace_col = Spaces.space(col_grid_shmem, Grids.CellFace())
    else
        error("Uncaught case")
    end
    return (ᶜspace_col, ᶠspace_col)
end

@inline is_valid_index_cw(data, ui) = 1 ≤ ui.I[1] ≤ size(data, 1)

@inline universal_index_columnwise(
    device::ClimaComms.AbstractCPUDevice,
    UI,
    data,
) = UI
