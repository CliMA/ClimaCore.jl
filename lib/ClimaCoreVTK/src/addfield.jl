
"""
addfield!(vtkfile, prefix::Union{String,Nothing}, f, dataspace)

Add a field or fields `f`, optionally prefixing the name with `prefix` to the
VTK file `vtkfile`, interpolating to `dataspace`.

`f` can be any of the following:
- a scalar or vector field (if no `prefix` is provided, then the field will be named `"data"`)
- a composite field, which will be named accordingly
- a `NamedTuple` of fields
"""
function addfield!(vtkfile, prefix, fields::NamedTuple, dataspace)
    for (key, val) in pairs(fields)
        name = string(key)
        if !isnothing(prefix)
            name = prefix * "." * name
        end
        addfield!(vtkfile, name, val, dataspace)
    end
end
addfield!(vtkfile, prefix, fields::Fields.FieldVector, dataspace) =
    addfield!(vtkfile, prefix, Fields._values(fields), dataspace)

addfield!(vtkfile, prefix, field::Fields.Field, dataspace) =
    addfield!(vtkfile, prefix, field, dataspace, eltype(field))

# composite field
function addfield!(vtkfile, prefix, field, dataspace, ::Type{T}) where {T}
    for i in 1:fieldcount(T)
        name = string(fieldname(T, i))
        if !isnothing(prefix)
            name = prefix * "." * name
        end
        addfield!(vtkfile, name, getproperty(field, i), dataspace)
    end
end

# scalars
function addfield!(vtkfile, name, field, dataspace, ::Type{T}) where {T <: Real}
    interp = Operators.Interpolate(dataspace)
    ifield = interp.(field)
    if isnothing(name)
        name = "data"
    end
    vtkfile[name] = vec(parent(ifield))
end

# vectors
function addfield!(
    vtkfile,
    name,
    field,
    dataspace,
    ::Type{T},
) where {T <: Geometry.AxisVector}
    interp = Operators.Interpolate(dataspace)
    # should we convert then interpolate, or vice versa?
    ifield = interp.(Geometry.Cartesian123Vector.(field))
    if isnothing(name)
        name = "data"
    end
    vtkfile[name] = (
        vec(parent(ifield.components.data.:1)),
        vec(parent(ifield.components.data.:2)),
        vec(parent(ifield.components.data.:3)),
    )
end
