#####
##### Column integrals (definite)
#####

"""
    column_integral_definite(field::Field)

The definite vertical column integral of field `field`.
"""
column_integral_definite(field::Fields.CenterExtrudedFiniteDifferenceField) =
    column_integral_definite(field, 1)
column_integral_definite(field::Fields.FaceExtrudedFiniteDifferenceField) =
    column_integral_definite(field, PlusHalf(1))

column_integral_definite(field::Fields.Field, one_index::Union{Int, PlusHalf}) =
    column_integral_definite(similar(level(field, one_index)), field)

"""
    column_integral_definite!(col∫field::Field, field::Field)

The definite vertical column integral, `col∫field`, of field `field`.
"""
function column_integral_definite!(col∫field::Fields.Field, field::Fields.Field)
    Fields.bycolumn(axes(field)) do colidx
        column_integral_definite!(col∫field[colidx], field[colidx])
        nothing
    end
    return nothing
end

function column_integral_definite!(
    col∫field::Fields.PointField,
    field::Fields.ColumnField,
)
    @inbounds col∫field[] = column_integral_definite(field)
    return nothing
end

function column_integral_definite(field::Fields.ColumnField)
    field_data = Fields.field_values(field)
    Δz_data = Spaces.dz_data(axes(field))
    Nv = Spaces.nlevels(axes(field))
    ∫field = zero(eltype(field))
    @inbounds for j in 1:Nv
        ∫field += field_data[j] * Δz_data[j]
    end
    return ∫field
end

# TODO: add support for indefinite integrals
