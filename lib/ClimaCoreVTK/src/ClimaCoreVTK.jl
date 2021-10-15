# Adapted from ClimateMachine.jl
# Original code by jkozdon

module ClimaCoreVTK

export writevtk

import ClimaCore.Fields
import ClimaCore.Geometry
import ClimaCore.Operators
import ClimaCore.Spaces
import ClimaCore.Topologies

using WriteVTK

function vtk_connectivity_map_highorder(Nq1, Nq2 = 1, Nq3 = 1)
    connectivity = Array{Int, 1}(undef, Nq1 * Nq2 * Nq3)
    L = LinearIndices((1:Nq1, 1:Nq2, 1:Nq3))

    corners = (
        (1, 1, 1),
        (Nq1, 1, 1),
        (Nq1, Nq2, 1),
        (1, Nq2, 1),
        (1, 1, Nq3),
        (Nq1, 1, Nq3),
        (Nq1, Nq2, Nq3),
        (1, Nq2, Nq3),
    )
    edges = (
        (2:(Nq1 - 1), 1:1, 1:1),
        (Nq1:Nq1, 2:(Nq2 - 1), 1:1),
        (2:(Nq1 - 1), Nq2:Nq2, 1:1),
        (1:1, 2:(Nq2 - 1), 1:1, 1:1),
        (2:(Nq1 - 1), 1:1, Nq3:Nq3),
        (Nq1:Nq1, 2:(Nq2 - 1), Nq3:Nq3),
        (2:(Nq1 - 1), Nq2:Nq2, Nq3:Nq3),
        (1:1, 2:(Nq2 - 1), Nq3:Nq3),
        (1:1, 1:1, 2:(Nq3 - 1)),
        (Nq1:Nq1, 1:1, 2:(Nq3 - 1)),
        (1:1, Nq2:Nq2, 2:(Nq3 - 1)),
        (Nq1:Nq1, Nq2:Nq2, 2:(Nq3 - 1)),
    )
    faces = (
        (1:1, 2:(Nq2 - 1), 2:(Nq3 - 1)),
        (Nq1:Nq1, 2:(Nq2 - 1), 2:(Nq3 - 1)),
        (2:(Nq1 - 1), 1:1, 2:(Nq3 - 1)),
        (2:(Nq1 - 1), Nq2:Nq2, 2:(Nq3 - 1)),
        (2:(Nq1 - 1), 2:(Nq2 - 1), 1:1),
        (2:(Nq1 - 1), 2:(Nq2 - 1), Nq3:Nq3),
    )
    if Nq2 == Nq3 == 1
        @assert Nq1 > 1
        corners = (corners[1:2]...,)
        edges = (edges[1],)
        faces = ()
    elseif Nq3 == 1
        @assert Nq1 > 1
        @assert Nq2 > 1
        corners = (corners[1:4]...,)
        edges = (edges[1:4]...,)
        faces = (faces[5],)
    end

    # corners
    iter = 1
    for (i, j, k) in corners
        connectivity[iter] = L[i, j, k]
        iter += 1
    end
    # edges
    for (is, js, ks) in edges
        for k in ks, j in js, i in is
            connectivity[iter] = L[i, j, k]
            iter += 1
        end
    end
    # faces
    for (is, js, ks) in faces
        for k in ks, j in js, i in is
            connectivity[iter] = L[i, j, k]
            iter += 1
        end
    end
    # interior
    for k in 2:(Nq3 - 1), j in 2:(Nq2 - 1), i in 2:(Nq1 - 1)
        connectivity[iter] = L[i, j, k]
        iter += 1
    end

    return connectivity
end

function setup_cells_highorder(Nq, realelems)
    con_map = vtk_connectivity_map_highorder(Nq, Nq)
    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, length(realelems))
    for (i, e) in enumerate(realelems)
        offset = (e - 1) * Nq * Nq
        cells[i] =
            MeshCell(VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, offset .+ con_map)
    end

    return cells
end

function setup_cells_raw(Nq, realelems)
    N = Nq - 1

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, N * N * length(realelems))
    ind = LinearIndices((1:Nq, 1:Nq))
    for e in realelems
        offset = (e - 1) * Nq * Nq
        for j in 1:N, i in 1:N
            cells[i + (j - 1) * N + (e - 1) * N * N] = MeshCell(
                VTKCellTypes.VTK_PIXEL,
                offset .+ ind[i:(i + 1), j:(j + 1)][:],
            )
        end
    end

    return cells
end

# TODO: support writing multiple fields to a single VTK
"""
    writevtk(
        basename::String,
        fieldname::String,
        field::Fields.SpectralElementField2D;
        number_sample_points = 0,
    )

Write `field` to `$(basename).vtu` using geometry and connectivity information
from the underlying `Space`.

If `number_sample_points > 0` then the fields are sampled on an equally spaced,
tensor-product grid of points with 'number_sample_points' in each direction and
the output VTK element type is set to by a VTK lagrange type.

When `number_sample_points == 0` the raw nodal values are saved, and linear VTK
elements are used connecting the degree of freedom boxes.
"""
function writevtk(
    basename::String,
    fieldname::String,
    field::Fields.SpectralElementField2D;
    number_sample_points = 0,
)
    # TODO: support Fields with more than one component
    @assert isempty(propertynames(field))

    space = axes(field)
    if number_sample_points > 0
        Nq = number_sample_points
        iquad = Spaces.Quadratures.ClosedUniform{Nq}()
        space = Spaces.SpectralElementSpace2D(space.topology, iquad)
        iop = Operators.Interpolate(space)
        field = iop.(field)
    else
        quadrature_style = Spaces.quadrature_style(space)
        Nq = Spaces.Quadratures.degrees_of_freedom(quadrature_style)
    end

    cf = Fields.coordinate_field(space)
    ccoords = Geometry.Cartesian123Point.(cf)

    cells = if number_sample_points > 0
        setup_cells_highorder(Nq, 1:Topologies.nlocalelems(field))
    else
        setup_cells_raw(Nq, 1:Topologies.nlocalelems(field))
    end

    fv = parent(Fields.field_values(field))[:]

    vtkfile = vtk_grid(
        basename,
        parent(ccoords.x1)[:],
        parent(ccoords.x2)[:],
        parent(ccoords.x3)[:],
        cells;
        compress = false,
    )
    vtk_point_data(vtkfile, fv, fieldname)
    vtk_save(vtkfile)
end


end # module
