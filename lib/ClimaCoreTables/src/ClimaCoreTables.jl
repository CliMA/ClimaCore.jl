module ClimaCoreTables

import Tables

import ClimaCore

export ClimaCoreTable

# Could be used to create a simple general (paritioned)
# file format given a ClimaCore schema description.

# ClimaCore Table interface
Tables.columnaccess(
    ::Type{<:T},
) where {T <: ClimaCore.DataLayouts.AbstractData} = true
Tables.columnaccess(::Type{<:T}) where {T <: ClimaCore.Fields.Field} = true
Tables.columnaccess(::Type{<:T}) where {T <: ClimaCore.Spaces.AbstractSpace} =
    true

const COL_DELIM = "."

function addcolumn!(cols::Vector, prefix::String, fields::NamedTuple)
    for ((key, val), _) in enumerate(pairs(fields))
        name = string(key)
        if prefix != ""
            name = prefix * COL_DELIM * name
        end
        addcolumn!(cols, name, val)
    end
    return symbols
end

function addcolumn!(cols::Vector, prefix::String, fields::Tuple)
    for (val, idx) in enumerate(fields)
        name = string(idx)
        if prefix != ""
            name = prefix * COL_DELIM * name
        end
        addcolumn!(cols, name, val)
    end
    return symbols
end

addcolumns!(
    cols::Vector,
    prefix::String,
    data::ClimaCore.DataLayouts.AbstractData,
) = addcolumns!(cols, prefix, data, eltype(data))

addcolumns!(cols::Vector, prefix::String, data::ClimaCore.Fields.Field) =
    addcolumns!(cols, prefix, data, eltype(data))

function addcolumns!(cols::Vector, prefix::String, data, ::Type{T}) where {T}
    for i in 1:fieldcount(T)
        name = string(fieldname(T, i))
        if prefix != ""
            name = prefix * COL_DELIM * name
        end
        addcolumns!(cols, name, getproperty(data, i))
    end
    return cols
end

# this is the base type implementation that returns the reshaped column view
# other terminal types could be defined (ex. AxisTensors) but then you would have to
# define a custom serialized representation for ex. Arrow and other Table implmentations.

function addcolumns!(
    cols::Vector,
    prefix::String,
    data,
    ::Type{T},
) where {T <: Real}
    name = isempty(prefix) ? "data" : prefix
    push!(cols, (Symbol(name) => vec(parent(data))))
end

struct ClimaCoreTable{T, CT} <: Tables.AbstractColumns
    data::T
    column_names::Vector{Symbol}
    column_types::Vector{DataType}
    column_rows::Int
    columns::Vector{CT}
end

function _table_from_columns(data, columns)
    column_names = [col_name for (col_name, _) in columns]
    column_types = [eltype(col) for (_, col) in columns]
    column_data = [col for (_, col) in columns]
    column_rows = length(column_data[1])
    return ClimaCoreTable{typeof(data), eltype(column_data)}(
        data,
        column_names,
        column_types,
        column_rows,
        column_data,
    )
end

function ClimaCoreTable(
    data::ClimaCore.DataLayouts.AbstractData;
    prefix::String = "",
)
    columns = []
    addcolumns!(columns, prefix, data)
    return _table_from_columns(data, columns)
end

function ClimaCoreTable(
    space::ClimaCore.Spaces.AbstractSpace;
    prefix::String = "",
)
    return ClimaCoreTable(getfield(space, :local_geometry); prefix)
end

function ClimaCoreTable(
    field::ClimaCore.Fields.Field;
    coords::Bool = true,
    local_geometry::Bool = false,
    prefix::String = "",
)
    columns = []
    if local_geometry
        lg_field = ClimaCore.Fields.local_geometry_field(axes(field))
        addcolumns!(columns, "local_geometry", lg_field)
    elseif coords
        coords_field = ClimaCore.Fields.coordinate_field(axes(field))
        addcolumns!(columns, "coordinates", coords_field)
    end
    addcolumns!(columns, prefix, field)
    return _table_from_columns(field, columns)
end

_table_data(table::ClimaCoreTable) = getfield(table, :data)
_table_column_names(table::ClimaCoreTable) = getfield(table, :column_names)
_table_column_types(table::ClimaCoreTable) = getfield(table, :column_types)
_table_column_rows(table::ClimaCoreTable) = getfield(table, :column_rows)
_table_columns(table::ClimaCoreTable) = getfield(table, :columns)

Tables.istable(::Type{<:ClimaCoreTable}) = true
Tables.table(d; kw...) = ClimaCoreTable(d; kw...)


function Tables.schema(table::ClimaCoreTable)
    return Tables.Schema(_table_column_names(table), _table_column_types(table))
end

# Column interface
Tables.columnaccess(::Type{<:ClimaCoreTable}) = true

Tables.columns(table::ClimaCoreTable) = table
Tables.columns(d::ClimaCore.DataLayouts.AbstractData) = ClimaCoreTable(d)
Tables.columns(d::ClimaCore.Spaces.AbstractSpace) = ClimaCoreTable(d)
Tables.columns(d::ClimaCore.Fields.Field) = ClimaCoreTable(d)

Tables.columnnames(table::ClimaCoreTable) = _table_column_names(table)

function Tables.getcolumn(table::ClimaCoreTable, i::Int)
    _table_columns(table)[i]
end

function Tables.getcolumn(table::ClimaCoreTable, nm::Symbol)
    i = findfirst(n -> n === nm, _table_column_names(table))
    Tables.getcolumn(table, i)
end

function Tables.getcolumn(
    table::ClimaCoreTable,
    ::Type{T},
    i::Int,
    nm::Symbol,
) where {T}
    Tables.getcolumn(table, i)
end

# Row interface
struct ClimaCoreTableRow{T, CT} <: Tables.AbstractRow
    idx::Int
    source::ClimaCoreTable{T, CT}
end

Tables.rowaccess(::Type{<:ClimaCoreTable}) = true
Tables.rows(table::ClimaCoreTable) = table

Base.eltype(::ClimaCoreTable{T, CT}) where {T, CT} = ClimaCoreTableRow{T, CT}
Base.length(table::ClimaCoreTable) = _table_column_rows(table)
Base.iterate(table::ClimaCoreTable, state = 1) =
    state > length(table) ? nothing :
    (ClimaCoreTableRow(state, table), state + 1)

Tables.getcolumn(row::ClimaCoreTableRow, i::Int) =
    getindex(Tables.getcolumn(getfield(row, :source), i), getfield(row, :idx))

Tables.getcolumn(row::ClimaCoreTableRow, nm::Symbol) =
    getindex(Tables.getcolumn(getfield(row, :source), nm), getfield(row, :idx))

Tables.getcolumn(row::ClimaCoreTableRow, ::Type, col::Int, nm::Symbol) =
    getindex(Tables.getcolumn(getfield(row, :source), col), getfield(row, :idx))

Tables.columnnames(row::ClimaCoreTableRow) =
    _table_column_names(getfield(row, :source))

# This isn't necessarily needed, just a default example
# include("arrow.jl")

end
