# Adapted from ClimateMachine.jl
# Original code by jkozdon

module ClimaCoreVTK

export writevtk, writepvd

using WriteVTK
import ClimaCore: Fields, Geometry, Spaces, Topologies, Operators

include("space.jl")
include("addfield.jl")

"""
    vtk_grid(basename, gridspace::ClimaCore.Spaces.AbstractSpace;
        basis=:cell, vtkargs...)

Construct a VTK grid from a `ClimaCore.Spaces.AbstractSpace`. If
`basis=:lagrange`, it will construct a mesh made of Lagrange cells (valid only
for spectral element spaces), otherwise it will it subdivide the space into
quads, with vertices at nodal points.
"""
function WriteVTK.vtk_grid(
    basename::String,
    gridspace::Spaces.AbstractSpace;
    basis,
    latlong = false,
    vtkargs...,
)

    if basis == :lagrange
        cells = vtk_cells_lagrange(gridspace)
    else
        cells = vtk_cells_linear(gridspace)
    end
    if latlong
        coords = Fields.coordinate_field(gridspace)
        if eltype(coords) <: Geometry.LatLongPoint
            coord_vecs = (vec(parent(coords.long)), vec(parent(coords.lat)))
        elseif eltype(coords) <: Geometry.LatLongZPoint
            coord_vecs = (
                vec(parent(coords.long)),
                vec(parent(coords.lat)),
                vec(parent(coords.z)),
            )
        else
            error(
                "latlong=true only works for fields with LatLongPoint or LatLongZPoint coordinates",
            )
        end
    else
        coords =
            Geometry.Cartesian123Point.(
                Fields.coordinate_field(gridspace),
                Ref(Spaces.global_geometry(gridspace)),
            )
        coord_vecs = (
            vec(parent(coords.x1)),
            vec(parent(coords.x2)),
            vec(parent(coords.x3)),
        )
    end
    return vtk_grid(basename, coord_vecs..., cells; vtkargs...)
end

function vtk_file(
    basename::String,
    fields,
    ;
    gridspace::Spaces.AbstractSpace = vtk_grid_space(fields),
    basis = :cell,
    dataspace::Spaces.AbstractSpace = basis == :cell ?
                                      vtk_cell_space(gridspace) : gridspace,
    vtkargs...,
)
    vtkfile = vtk_grid(basename, gridspace; basis, vtkargs...)
    addfield!(vtkfile, nothing, fields, dataspace)
    return vtkfile
end

"""
    writevtk(
        basename::String,
        fields;
        basis=:cell,
        latlong=false,
        vtkargs...
    )

Write `fields` to as an unstructured mesh VTK file named `$(basename).vtu`.

`fields` can be either:
- a ClimaCore `Field` object,
- a `FieldVector` object,
- a `NamedTuple` of `Field`s.

The `basis` keyword option determines the type of cells used to write.:
- `:cell` (default): output values at cell centers (interpolating where
  necessary).
- `:point`: output values at cell vertices.
- `:lagrange`: output values at Lagrange nodes (valid only for spectral element
  spaces), using Use VTK Lagrange cells to accurately represent high-order
  elements.

The `latlong=true` keyword option will output a spherical or spherical shell
domain using the Mercator projection, with longitude along the x-axis, latitude
along the y-axis, and altitude along the z-axis (if applicable). Note this
currently only displays correctly if the number of elements across the cubed
sphere face is even.

Any additional keyword arguments are passed to
[`WriteVTK.vtk_grid`](https://jipolanco.github.io/WriteVTK.jl/stable/grids/syntax/#Supported-options).
"""
function writevtk(basename::String, fields; vtkargs...)
    vtkfile = vtk_file(basename, fields; vtkargs...)
    vtk_save(vtkfile)
end

"""
    writepvd(
        basename::String,
        times,
        fields;
        vtkargs...
    )

Write a sequence of fields at `times` as a Paraview collection (.pvd) file, along with VTK files.

`fields` can be either be an iterable collection of fields, or a `NamedTuple` of collections.
"""
function writepvd(basename::String, times, fields; vtkargs...)
    npad = ndigits(length(times); pad = 3)

    paraview_collection(basename) do pvd
        for (n, time) in enumerate(times)
            pvd[time] = vtk_file(
                basename * "_" * string(n, pad = npad),
                fields[n];
                vtkargs...,
            )
        end
    end
end
function writepvd(basename::String, times, fields::NamedTuple; vtkargs...)
    npad = ndigits(length(times); pad = 3)

    paraview_collection(basename) do pvd
        for (n, time) in enumerate(times)
            field = NamedTuple(key => val[n] for (key, val) in pairs(fields))
            pvd[time] = vtk_file(
                basename * "_" * string(n, pad = npad),
                field;
                vtkargs...,
            )
        end
    end
end

Base.@deprecate writevtk(basename::String, times, fields; vtkargs...) writepvd(
    basename,
    times,
    fields;
    vtkargs...,
)


end # module
