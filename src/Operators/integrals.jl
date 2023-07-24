#####
##### Column integrals (definite)
#####

"""
    column_integral_definite!(col∫field::Field, field::Field)

The definite vertical column integral, `col∫field`, of field `field`.
"""
column_integral_definite!(col∫field::Fields.Field, field::Fields.Field) =
    column_integral_definite!(ClimaComms.device(axes(field)), col∫field, field)

function column_integral_definite!(
    ::ClimaComms.CUDADevice,
    col∫field::Fields.Field,
    field::Fields.Field,
)
    Ni, Nj, _, _, Nh = size(Fields.field_values(col∫field))
    nthreads, nblocks = Spaces._configure_threadblock(Ni * Nj * Nh)
    @cuda threads = nthreads blocks = nblocks column_integral_definite_kernel!(
        col∫field,
        field,
    )
end

function column_integral_definite_kernel!(
    col∫field::Fields.SpectralElementField,
    field::Fields.ExtrudedFiniteDifferenceField,
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(field))
    if idx <= Ni * Nj * Nh
        i, j, h = Spaces._get_idx((Ni, Nj, Nh), idx)
        colfield = Spaces.column(field, i, j, h)
        _column_integral_definite!(Spaces.column(col∫field, i, j, h), colfield)
    end
    return nothing
end

column_integral_definite_kernel!(
    col∫field::Fields.PointField,
    field::Fields.FiniteDifferenceField,
) = _column_integral_definite!(col∫field, field)

function column_integral_definite!(
    ::ClimaComms.AbstractCPUDevice,
    col∫field::Fields.SpectralElementField,
    field::Fields.ExtrudedFiniteDifferenceField,
)
    Fields.bycolumn(axes(field)) do colidx
        _column_integral_definite!(col∫field[colidx], field[colidx])
        nothing
    end
    return nothing
end

column_integral_definite!(
    ::ClimaComms.AbstractCPUDevice,
    col∫field::Fields.PointField,
    field::Fields.FiniteDifferenceField,
) = _column_integral_definite!(col∫field, field)

function _column_integral_definite!(
    col∫field::Fields.PointField,
    field::Fields.ColumnField,
)
    space = axes(field)
    Δz_field = Fields.Δz_field(space)
    Nv = Spaces.nlevels(space)

    col∫field[] = 0
    @inbounds for idx in 1:Nv
        col∫field[] +=
            reduction_getindex(field, idx) * reduction_getindex(Δz_field, idx)
    end
    return nothing
end

reduction_getindex(column_field, index) = @inbounds getidx(
    axes(column_field),
    column_field,
    Interior(),
    index - 1 + left_idx(axes(column_field)),
)

# TODO: add support for indefinite integrals
