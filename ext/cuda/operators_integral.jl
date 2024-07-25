import ClimaCore: Spaces, Fields, Spaces, Topologies
import ClimaCore.Operators: strip_space
import ClimaCore.Operators:
    column_integral_definite!,
    column_integral_definite_kernel!,
    column_integral_indefinite_kernel!,
    column_integral_indefinite!,
    column_mapreduce_device!,
    _column_integral_definite!,
    _column_integral_indefinite!

import ClimaComms
using CUDA: @cuda

function column_integral_definite!(
    ::ClimaComms.CUDADevice,
    ∫field::Fields.Field,
    ᶜfield::Fields.Field,
)
    space = axes(∫field)
    Ni, Nj, _, _, Nh = size(Fields.field_values(∫field))
    nthreads, nblocks = _configure_threadblock(Ni * Nj * Nh)
    args = (strip_space(∫field, space), strip_space(ᶜfield, space))
    auto_launch!(
        column_integral_definite_kernel!,
        args,
        size(Fields.field_values(∫field));
        threads_s = nthreads,
        blocks_s = nblocks,
    )
end

function column_integral_definite_kernel!(
    ∫field,
    ᶜfield::Fields.CenterExtrudedFiniteDifferenceField,
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(ᶜfield))
    if idx <= Ni * Nj * Nh
        i, j, h = Topologies._get_idx((Ni, Nj, Nh), idx)
        ∫field_column = Spaces.column(∫field, i, j, h)
        ᶜfield_column = Spaces.column(ᶜfield, i, j, h)
        _column_integral_definite!(∫field_column, ᶜfield_column)
    end
    return nothing
end

function column_integral_indefinite_kernel!(
    ᶠ∫field::Fields.FaceExtrudedFiniteDifferenceField,
    ᶜfield::Fields.CenterExtrudedFiniteDifferenceField,
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(ᶜfield))
    if idx <= Ni * Nj * Nh
        i, j, h = Topologies._get_idx((Ni, Nj, Nh), idx)
        ᶠ∫field_column = Spaces.column(ᶠ∫field, i, j, h)
        ᶜfield_column = Spaces.column(ᶜfield, i, j, h)
        _column_integral_indefinite!(ᶠ∫field_column, ᶜfield_column)
    end
    return nothing
end

function column_integral_indefinite!(
    ::ClimaComms.CUDADevice,
    ᶠ∫field::Fields.Field,
    ᶜfield::Fields.Field,
)
    Ni, Nj, _, _, Nh = size(Fields.field_values(ᶠ∫field))
    nthreads, nblocks = _configure_threadblock(Ni * Nj * Nh)
    args = (ᶠ∫field, ᶜfield)
    auto_launch!(
        column_integral_indefinite_kernel!,
        args,
        size(Fields.field_values(ᶠ∫field));
        threads_s = nthreads,
        blocks_s = nblocks,
    )
end

function column_mapreduce_device!(
    ::ClimaComms.CUDADevice,
    fn::F,
    op::O,
    reduced_field::Fields.Field,
    fields::Fields.Field...,
) where {F, O}
    Ni, Nj, _, _, Nh = size(Fields.field_values(reduced_field))
    nthreads, nblocks = _configure_threadblock(Ni * Nj * Nh)
    kernel! = if first(fields) isa Fields.ExtrudedFiniteDifferenceField
        column_mapreduce_kernel_extruded!
    else
        column_mapreduce_kernel!
    end
    args = (
        fn,
        op,
        # reduced_field,
        strip_space(reduced_field, axes(reduced_field)),
        # fields...,
        map(field -> strip_space(field, axes(field)), fields)...,
    )
    auto_launch!(
        kernel!,
        args,
        size(Fields.field_values(reduced_field));
        threads_s = nthreads,
        blocks_s = nblocks,
    )
end

function column_mapreduce_kernel_extruded!(
    fn::F,
    op::O,
    reduced_field,
    fields...,
) where {F, O}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(reduced_field))
    if idx <= Ni * Nj * Nh
        i, j, h = Topologies._get_idx((Ni, Nj, Nh), idx)
        reduced_field_column = Spaces.column(reduced_field, i, j, h)
        field_columns = map(field -> Spaces.column(field, i, j, h), fields)
        _column_mapreduce!(fn, op, reduced_field_column, field_columns...)
    end
    return nothing
end
