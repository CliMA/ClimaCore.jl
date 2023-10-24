function vtk_connectivity_map_lagrange(Nq1, Nq2 = 1, Nq3 = 1)
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

"""
  vtk_cells_lagrange(space)

Construct a vector of `MeshCell` objects representing the elements of `space` as
an unstuctured mesh of Lagrange polynomial cells, suitable for passing to
`vtk_grid`.
"""
function vtk_cells_lagrange(gspace::Spaces.SpectralElementSpace2D)
    quad = Spaces.quadrature_style(gspace)
    @assert quad isa Quadratures.ClosedUniform
    Nq = Quadratures.degrees_of_freedom(quad)    # TODO: this should depend on the backing DataLayouts (e.g. IJFH)
    con_map = vtk_connectivity_map_lagrange(Nq, Nq)
    [
        MeshCell(
            VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL,
            ((e - 1) * Nq * Nq) .+ con_map,
        ) for e in 1:Topologies.nlocalelems(gspace)
    ]
end

"""
  vtk_cells_linear(space)

Construct a vector of `MeshCell` objects representing the elements of `space` as
an unstuctured mesh of linear cells, suitable for passing to
`vtk_grid`.
"""
function vtk_cells_linear(gridspace::Spaces.SpectralElementSpace2D)
    Nq = Spaces.Quadratures.degrees_of_freedom(
        Spaces.quadrature_style(gridspace),
    )
    Nh = Topologies.nlocalelems(gridspace)
    ind = LinearIndices((1:Nq, 1:Nq, 1:Nh))
    cells = [
        MeshCell(
            VTKCellTypes.VTK_QUAD,
            (
                ind[i, j, e],
                ind[i + 1, j, e],
                ind[i + 1, j + 1, e],
                ind[i, j + 1, e],
            ),
        ) for i in 1:(Nq - 1), j in 1:(Nq - 1), e in 1:Nh
    ]
    return vec(cells)
end

function vtk_cells_linear(gridspace::Spaces.FaceExtrudedFiniteDifferenceSpace2D)
    hspace = Spaces.horizontal_space(gridspace)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(hspace))
    Nh = Topologies.nlocalelems(hspace)
    Nv = Spaces.nlevels(gridspace)
    ind = LinearIndices((1:Nv, 1:Nq, 1:Nh)) # assume VIFH
    cells = [
        MeshCell(
            VTKCellTypes.VTK_QUAD,
            (
                ind[v, i, e],
                ind[v + 1, i, e],
                ind[v + 1, i + 1, e],
                ind[v, i + 1, e],
            ),
        ) for v in 1:(Nv - 1), i in 1:(Nq - 1), e in 1:Nh
    ]
    return vec(cells)
end
function vtk_cells_linear(gridspace::Spaces.FaceExtrudedFiniteDifferenceSpace3D)
    hspace = Spaces.horizontal_space(gridspace)
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(hspace))
    Nh = Topologies.nlocalelems(hspace)
    Nv = Spaces.nlevels(gridspace)
    ind = LinearIndices((1:Nv, 1:Nq, 1:Nq, 1:Nh)) # assumes VIJFH
    cells = [
        MeshCell(
            VTKCellTypes.VTK_HEXAHEDRON,
            (
                ind[v, i, j, e],
                ind[v + 1, i, j, e],
                ind[v + 1, i + 1, j, e],
                ind[v, i + 1, j, e],
                ind[v, i, j + 1, e],
                ind[v + 1, i, j + 1, e],
                ind[v + 1, i + 1, j + 1, e],
                ind[v, i + 1, j + 1, e],
            ),
        ) for v in 1:(Nv - 1), i in 1:(Nq - 1), j in 1:(Nq - 1), e in 1:Nh
    ]
    return vec(cells)
end

"""
    vtk_grid_space(space::ClimaCore.Spaces.AbstractSpace)

The space for the grid used by VTK, for any field on `space`.

This generally does two things:
 - Modifies the horizontal space to use a `ClosedUniform` quadrature rule, which
   will use equispaced nodal points in the reference element. This is required
   for using [VTK Lagrange elements](https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/)
   (see [1](https://discourse.paraview.org/t/node-positions-of-high-order-lagrange-quadrilateral-cells/7012/3)).
 - Modifies the vertical space to be on the faces.
"""
function vtk_grid_space(space::Spaces.SpectralElementSpace1D)
    if Spaces.quadrature_style(space) isa Spaces.Quadratures.ClosedUniform
        return space
    end
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    lagrange_quad = Spaces.Quadratures.ClosedUniform{Nq}()
    return Spaces.SpectralElementSpace1D(Spaces.topology(space), lagrange_quad)
end
function vtk_grid_space(space::Spaces.SpectralElementSpace2D)
    if Spaces.quadrature_style(space) isa Spaces.Quadratures.ClosedUniform
        return space
    end
    Nq = Spaces.Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    lagrange_quad = Spaces.Quadratures.ClosedUniform{Nq}()
    return Spaces.SpectralElementSpace2D(Spaces.topology(space), lagrange_quad)
end
function vtk_grid_space(space::Spaces.FaceExtrudedFiniteDifferenceSpace)
    # this will need to be updated for warped meshes
    horizontal_space = vtk_grid_space(Spaces.horizontal_space(space))
    vertical_space = Spaces.FaceFiniteDifferenceSpace(space.vertical_topology)
    return Spaces.ExtrudedFiniteDifferenceSpace(
        horizontal_space,
        vertical_space,
    )
end
function vtk_grid_space(space::Spaces.CenterExtrudedFiniteDifferenceSpace)
    return vtk_grid_space(Spaces.FaceExtrudedFiniteDifferenceSpace(space))
end

vtk_grid_space(field::Fields.Field) = vtk_grid_space(axes(field))
vtk_grid_space(fields::NamedTuple) = vtk_grid_space(first(fields))
vtk_grid_space(fieldvec::Fields.FieldVector) =
    vtk_grid_space(Fields._values(fieldvec))


"""
    vtk_cell_space(gridspace::ClimaCore.Spaces.AbstractSpace)

Construct a space for outputting cell data, when using outputting a grid `gridspace`.
be stored.

This generally does two things:
 - Modifies the horizontal space to use a `Uniform` quadrature rule, which
   will use equispaced nodal points in the reference element (excluding the boundary).
 - Modifies the vertical space to be on the centers.
"""
function vtk_cell_space(gridspace::Spaces.SpectralElementSpace1D)
    @assert Spaces.quadrature_style(gridspace) isa
            Spaces.Quadratures.ClosedUniform
    Nq = Spaces.Quadratures.degrees_of_freedom(
        Spaces.quadrature_style(gridspace),
    )
    quad = Spaces.Quadratures.Uniform{Nq - 1}()
    return Spaces.SpectralElementSpace1D(Spaces.topology(gridspace), quad)
end
function vtk_cell_space(gridspace::Spaces.SpectralElementSpace2D)
    @assert Spaces.quadrature_style(gridspace) isa
            Spaces.Quadratures.ClosedUniform
    Nq = Spaces.Quadratures.degrees_of_freedom(
        Spaces.quadrature_style(gridspace),
    )
    quad = Spaces.Quadratures.Uniform{Nq - 1}()
    return Spaces.SpectralElementSpace2D(Spaces.topology(gridspace), quad)
end
function vtk_cell_space(gridspace::Spaces.FaceExtrudedFiniteDifferenceSpace)
    # this will need to be updated for warped meshes
    horizontal_space = vtk_cell_space(Spaces.horizontal_space(gridspace))
    vertical_space =
        Spaces.CenterFiniteDifferenceSpace(gridspace.vertical_topology)
    return Spaces.ExtrudedFiniteDifferenceSpace(
        horizontal_space,
        vertical_space,
    )
end
