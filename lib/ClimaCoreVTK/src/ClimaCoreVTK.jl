# Adapted from ClimateMachine.jl
# Original code by jkozdon

module ClimaCoreVTK

export writevtk

using WriteVTK
import ClimaCore: Fields, Geometry, Spaces, Topologies, Operators

include("space.jl")
include("addfield.jl")

"""
    vtk_grid(basename, space::ClimaCore.Spaces.AbstractSpace;
        basis=:lagrange, vtkargs...)

"""
function WriteVTK.vtk_grid(
    basename::String,
    ispace::Spaces.AbstractSpace;
    basis = :lagrange,
    vtkargs...,
)

    if basis == :lagrange
        cells = vtk_cells_lagrange(ispace)
    elseif basis == :linear
        cells = vtk_cells_linear(ispace)
    else
        error("invalid basis $basis")
    end
    cart_coords =
        Geometry.Cartesian123Point.(
            Fields.coordinate_field(ispace),
            Ref(ispace.global_geometry),
        )
    return vtk_grid(
        basename,
        vec(parent(cart_coords.x1)),
        vec(parent(cart_coords.x2)),
        vec(parent(cart_coords.x3)),
        cells;
        vtkargs...,
    )
end

function vtk_file(
    basename::String,
    fields,
    ispace::Spaces.AbstractSpace = vtk_space(fields);
    vtkargs...,
)
    vtkfile = vtk_grid(basename, ispace; vtkargs...)
    addfield!(vtkfile, nothing, fields, ispace)
    return vtkfile
end

"""
    writevtk(
        basename::String,
        fields,
        [ispace=vtk_space(fields)];
        basis=:lagrange,
        vtkargs...
    )

Write `fields` to as an unstructured mesh VTK file named `$(basename).vtu`.

`fields` can be either:
- a ClimaCore `Field` object,
- a `FieldVector` object,
- a `NamedTuple` of `Field`s.

The optional `ispace` argument determines the coordinates which will be written
to a file: by default, this places nodes equispaced within the reference
element.

The `basis` keyword option determines the type of cells used to write.:
- `:lagrange` (default): Use VTK Lagrange cells to accurately represent
  high-order elements.
- `:linear`: Divide each element into linear elements. This is a less faithful
  representation, but is compatible with more software.

Any additional keyword arguments are passed to
[`WriteVTK.vtk_grid`](https://jipolanco.github.io/WriteVTK.jl/stable/grids/syntax/#Supported-options).
"""
function writevtk(
    basename::String,
    fields,
    ispace::Spaces.AbstractSpace = vtk_space(fields);
    vtkargs...,
)
    vtkfile = vtk_file(basename, fields, ispace; vtkargs...)
    vtk_save(vtkfile)
end

"""
    writevtk(
        basename::String,
        times,
        fields,
        [ispace=vtk_space(fields)];
        vtkargs...
    )

Write a sequence of fields `fields` at times `times` as a Paraview collection.
"""
function writevtk(
    basename::String,
    times,
    fields,
    ispace = vtk_space(first(fields));
    vtkargs...,
)
    npad = ndigits(length(times); pad = 3)

    paraview_collection(basename) do pvd
        for (n, time) in enumerate(times)
            pvd[time] = vtk_file(
                basename * "_" * string(n, pad = npad),
                fields[n],
                ispace;
                vtkargs...,
            )
        end
    end
end


end # module
