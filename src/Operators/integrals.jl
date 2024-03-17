import ..CUDAUtils: auto_launch!
import ..RecursiveApply: rzero, ⊠, ⊞, radd, rmul
import ..Utilities
import ..DataLayouts
import RootSolvers
import ClimaComms

"""
    column_integral_definite!(∫field::Field, ᶜfield::Field)

Sets `∫field```{}= \\int_0^{z_{max}}\\,```ᶜfield```(z)\\,dz``, where ``z_{max}``
is the value of `z` at the top of the domain. The input `ᶜfield` must lie on a
cell-center space, and the output `∫field` must lie on the corresponding
horizontal space.
"""
column_integral_definite!(∫field::Fields.Field, ᶜfield::Fields.Field) =
    column_integral_definite!(ClimaComms.device(axes(ᶜfield)), ∫field, ᶜfield)

function column_integral_definite!(
    ::ClimaComms.CUDADevice,
    ∫field::Fields.Field,
    ᶜfield::Fields.Field,
)
    space = axes(∫field)
    Ni, Nj, _, _, Nh = size(Fields.field_values(∫field))
    nthreads, nblocks = Topologies._configure_threadblock(Ni * Nj * Nh)
    args = (strip_space(∫field, space), strip_space(ᶜfield, space))
    auto_launch!(
        column_integral_definite_kernel!,
        args,
        length(parent(Fields.field_values(∫field))),
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

function column_integral_definite_kernel!(
    ∫field,
    ᶜfield::Fields.CenterFiniteDifferenceField,
)
    _column_integral_definite!(∫field, ᶜfield)
    return nothing
end

column_integral_definite!(
    ::ClimaComms.AbstractCPUDevice,
    ∫field::Fields.SpectralElementField,
    ᶜfield::Fields.ExtrudedFiniteDifferenceField,
) =
    Fields.bycolumn(axes(ᶜfield)) do colidx
        _column_integral_definite!(∫field[colidx], ᶜfield[colidx])
    end

column_integral_definite!(
    ::ClimaComms.AbstractCPUDevice,
    ∫field::Fields.PointField,
    ᶜfield::Fields.FiniteDifferenceField,
) = _column_integral_definite!(∫field, ᶜfield)

function _column_integral_definite!(∫field, ᶜfield::Fields.ColumnField)
    Δz = Fields.Δz_field(ᶜfield)
    first_level = Operators.left_idx(axes(ᶜfield))
    last_level = Operators.right_idx(axes(ᶜfield))
    ∫field_data = Fields.field_values(∫field)
    Base.setindex!(∫field_data, rzero(eltype(∫field)))
    @inbounds for level in first_level:last_level
        val =
            ∫field_data[] ⊞
            Fields.level(ᶜfield, level)[] ⊠ Fields.level(Δz, level)[]
        Base.setindex!(∫field_data, val)
    end
    return nothing
end

"""
    column_integral_indefinite!(ᶠ∫field::Field, ᶜfield::Field)

Sets `ᶠ∫field```(z) = \\int_0^z\\,```ᶜfield```(z')\\,dz'``. The input `ᶜfield`
must lie on a cell-center space, and the output `ᶠ∫field` must lie on the
corresponding cell-face space.

    column_integral_indefinite!(
        f::Function,
        ᶠ∫field::Fields.ColumnField,
        ϕ₀ = 0,
        average = (ϕ⁻, ϕ⁺) -> (ϕ⁻ + ϕ⁺) / 2,
    )

The indefinite integral `ᶠ∫field = ϕ(z) = ∫ f(ϕ,z) dz` given:

- `f` the integral integrand function (which may be a function)
- `ᶠ∫field` the resulting (scalar) field `ϕ(z)`
- `ϕ₀` (optional) the boundary condition
- `average` (optional) a function to compute the cell center
   average between two cell faces (`ϕ⁻, ϕ⁺`).
"""
column_integral_indefinite!(ᶠ∫field::Fields.Field, ᶜfield::Fields.Field) =
    column_integral_indefinite!(ClimaComms.device(ᶠ∫field), ᶠ∫field, ᶜfield)

function column_integral_indefinite!(
    ::ClimaComms.CUDADevice,
    ᶠ∫field::Fields.Field,
    ᶜfield::Fields.Field,
)
    args = (ᶠ∫field, ᶜfield)
    auto_launch!(
        column_integral_indefinite_kernel!,
        args,
        length(parent(Fields.field_values(ᶠ∫field))),
    )
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

column_integral_indefinite_kernel!(
    ᶠ∫field::Fields.FaceFiniteDifferenceField,
    ᶜfield::Fields.CenterFiniteDifferenceField,
) = _column_integral_indefinite!(ᶠ∫field, ᶜfield)

column_integral_indefinite!(
    ::ClimaComms.AbstractCPUDevice,
    ᶠ∫field::Fields.FaceExtrudedFiniteDifferenceField,
    ᶜfield::Fields.CenterExtrudedFiniteDifferenceField,
) =
    Fields.bycolumn(axes(ᶜfield)) do colidx
        _column_integral_indefinite!(ᶠ∫field[colidx], ᶜfield[colidx])
    end

column_integral_indefinite!(
    ::ClimaComms.AbstractCPUDevice,
    ᶠ∫field::Fields.FaceFiniteDifferenceField,
    ᶜfield::Fields.CenterFiniteDifferenceField,
) = _column_integral_indefinite!(ᶠ∫field, ᶜfield)

function _column_integral_indefinite!(
    ᶠ∫field::Fields.ColumnField,
    ᶜfield::Fields.ColumnField,
)
    face_space = axes(ᶠ∫field)
    ᶜΔz = Fields.Δz_field(ᶜfield)
    first_level = Operators.left_idx(face_space)
    last_level = Operators.right_idx(face_space)
    @inbounds Fields.level(ᶠ∫field, first_level)[] = rzero(eltype(ᶜfield))
    @inbounds for level in (first_level + 1):last_level
        Fields.level(ᶠ∫field, level)[] =
            Fields.level(ᶠ∫field, level - 1)[] ⊞
            Fields.level(ᶜfield, level - half)[] ⊠
            Fields.level(ᶜΔz, level - half)[]
    end
end

dual_space(space::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    Spaces.CenterExtrudedFiniteDifferenceSpace(space)
dual_space(space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(space)

dual_space(space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
dual_space(space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)

# First, dispatch on device:
column_integral_indefinite!(
    f::Function,
    ᶠ∫field::Fields.Field,
    ϕ₀ = zero(eltype(ᶠ∫field)),
    average = (ϕ⁻, ϕ⁺) -> (ϕ⁻ + ϕ⁺) / 2,
) = column_integral_indefinite!(
    f,
    ClimaComms.device(ᶠ∫field),
    ᶠ∫field,
    ϕ₀,
    average,
)

#####
##### CPU
#####

column_integral_indefinite!(
    f::Function,
    ::ClimaComms.AbstractCPUDevice,
    ᶠ∫field,
    args...,
) = column_integral_indefinite_cpu!(f, ᶠ∫field, args...)

column_integral_indefinite!(
    f::Function,
    ::ClimaComms.AbstractCPUDevice,
    ᶠ∫field::Fields.FaceExtrudedFiniteDifferenceField,
    args...,
) =
    Fields.bycolumn(axes(ᶠ∫field)) do colidx
        column_integral_indefinite_cpu!(f, ᶠ∫field[colidx], args...)
    end

#=
Function-based signature, solve for ϕ:
```
∂ϕ/∂z = f(ϕ,z)
(ᶠϕ^{k+1}-ᶠϕ^{k})/ᶜΔz = ᶜf(ᶜϕ̄,ᶜz)
ᶜϕ̄ = (ϕ^{k+1}+ϕ^{k})/2
(ᶠϕ^{k+1}-ᶠϕ^{k})/ᶜΔz = ᶜf((ᶠϕ^{k+1}+ᶠϕ^{k})/2,ᶜz)
root equation: (_ϕ-ϕ^{k})/Δz = f((_ϕ+ϕ^{k})/2,ᶜz)
```
=#
function column_integral_indefinite_cpu!(
    f::Function,
    ᶠ∫field::Fields.ColumnField,
    ϕ₀ = zero(eltype(ᶠ∫field)),
    average = (ϕ⁻, ϕ⁺) -> (ϕ⁻ + ϕ⁺) / 2,
)
    cspace = dual_space(axes(ᶠ∫field))
    ᶜzfield = Fields.coordinate_field(cspace)
    face_space = axes(ᶠ∫field)
    first_level = Operators.left_idx(face_space)
    last_level = Operators.right_idx(face_space)
    ᶜΔzfield = Fields.Δz_field(ᶜzfield)
    @inbounds Fields.level(ᶠ∫field, first_level)[] = ϕ₀
    ϕ₁ = ϕ₀
    @inbounds for level in (first_level + 1):last_level
        ᶜz = Fields.level(ᶜzfield.z, level - half)[]
        ᶜΔz = Fields.level(ᶜΔzfield, level - half)[]
        ϕ₀ = ϕ₁
        root_eq(_x) = (_x - ϕ₀) / ᶜΔz - f(average(_x, ϕ₀), ᶜz)
        sol = RootSolvers.find_zero(
            root_eq,
            RootSolvers.NewtonsMethodAD(ϕ₀),
            RootSolvers.CompactSolution(),
        )
        ϕ₁ = sol.root
        f₁ = f(average(ϕ₀, ϕ₁), ᶜz)
        ᶜintegrand_lev = f₁
        @inbounds Fields.level(ᶠ∫field, level)[] =
            radd(Fields.level(ᶠ∫field, level - 1)[], rmul(ᶜintegrand_lev, ᶜΔz))
    end
    return nothing
end

"""
    column_mapreduce!(fn, op, reduced_field::Field, fields::Field...)

Applies mapreduce along the vertical direction. The input `fields` must all lie
on the same space, and the output `reduced_field` must lie on the corresponding
horizontal space. The function `fn` is mapped over every input, and the function
`op` is used to reduce the outputs of `fn`.
"""
column_mapreduce!(
    fn::F,
    op::O,
    reduced_field::Fields.Field,
    fields::Fields.Field...,
) where {F, O} = column_mapreduce_device!(
    ClimaComms.device(reduced_field),
    fn,
    op,
    reduced_field,
    fields...,
)

function column_mapreduce_device!(
    ::ClimaComms.CUDADevice,
    fn::F,
    op::O,
    reduced_field::Fields.Field,
    fields::Fields.Field...,
) where {F, O}
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
        length(parent(Fields.field_values(first(fields)))),
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

column_mapreduce_kernel!(fn::F, op::O, reduced_field, fields...) where {F, O} =
    _column_mapreduce!(fn, op, reduced_field, fields...)

column_mapreduce_device!(
    ::ClimaComms.AbstractCPUDevice,
    fn::F,
    op::O,
    reduced_field::Fields.SpectralElementField,
    fields::Fields.ExtrudedFiniteDifferenceField...,
) where {F, O} =
    Fields.bycolumn(axes(reduced_field)) do colidx
        column_fields = map(field -> field[colidx], fields)
        _column_mapreduce!(fn, op, reduced_field[colidx], column_fields...)
    end

column_mapreduce_device!(
    ::ClimaComms.AbstractCPUDevice,
    fn::F,
    op::O,
    reduced_field::Fields.PointField,
    fields::Fields.FiniteDifferenceField...,
) where {F, O} = _column_mapreduce!(fn, op, reduced_field, fields...)

function _column_mapreduce!(fn::F, op::O, reduced_field, fields...) where {F, O}
    space = axes(first(fields))
    for field in Base.tail(fields) # Base.rest breaks on the gpu
        axes(field) === space ||
            error("All inputs to column_mapreduce must lie on the same space")
    end
    (_, _, _, Nv, _) = size(Fields.field_values(first(fields)))
    first_level = left_boundary_idx(Nv, space)
    last_level = right_boundary_idx(Nv, space)
    # TODO: This code is allocating memory. In particular, even if we comment
    # out the rest of this function, the first line alone allocates memory.
    # This problem is not fixed by replacing map with ntuple or unrolled_map.
    fields_data = map(field -> Fields.field_values(field), fields)
    first_level_values = map(
        field_data ->
            (@inbounds data_level(field_data, space, first_level)[]),
        fields_data,
    )
    reduced_field_data = Fields.field_values(reduced_field)
    Base.setindex!(reduced_field_data, fn(first_level_values...))
    for level in (first_level + 1):last_level
        values = map(
            field_data ->
                (@inbounds data_level(field_data, space, level)[]),
            fields_data,
        )
        Base.setindex!(
            reduced_field_data,
            op(reduced_field_data[], fn(values...)),
        )
    end
    return nothing
end

import ..Utilities
Base.@propagate_inbounds data_level(
    data,
    ::Operators.CenterPlaceholderSpace,
    v::Int,
) = DataLayouts.level(data, v)
Base.@propagate_inbounds data_level(
    data,
    ::Spaces.CenterFiniteDifferenceSpace,
    v::Int,
) = DataLayouts.level(data, v)

Base.@propagate_inbounds data_level(
    data,
    ::Operators.FacePlaceholderSpace,
    v::Utilities.PlusHalf,
) = DataLayouts.level(data, v.i + 1)
Base.@propagate_inbounds data_level(
    data,
    ::Spaces.FaceFiniteDifferenceSpace,
    v::Utilities.PlusHalf,
) = DataLayouts.level(data, v.i + 1)

left_boundary_idx(n, ::Operators.CenterPlaceholderSpace) = 1
right_boundary_idx(n, ::Operators.CenterPlaceholderSpace) = n
left_boundary_idx(n, ::Operators.FacePlaceholderSpace) = Utilities.half
right_boundary_idx(n, ::Operators.FacePlaceholderSpace) = n - Utilities.half

left_boundary_idx(n, ::Spaces.CenterFiniteDifferenceSpace) = 1
right_boundary_idx(n, ::Spaces.CenterFiniteDifferenceSpace) = n
left_boundary_idx(n, ::Spaces.FaceFiniteDifferenceSpace) = Utilities.half
right_boundary_idx(n, ::Spaces.FaceFiniteDifferenceSpace) = n - Utilities.half
