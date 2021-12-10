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
function vtk_cells_lagrange(
    space::Spaces.SpectralElementSpace2D{
        T,
        Spaces.Quadratures.ClosedUniform{Nq},
    },
) where {T, Nq}
    # TODO: this should depend on the backing DataLayouts (e.g. IJFH)
    con_map = vtk_connectivity_map_lagrange(Nq, Nq)
    [
        MeshCell(
            VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL,
            ((e - 1) * Nq * Nq) .+ con_map,
        ) for e in 1:Topologies.nlocalelems(space)
    ]
end

"""
  vtk_cells_linear(space)

Construct a vector of `MeshCell` objects representing the elements of `space` as
an unstuctured mesh of linear cells, suitable for passing to
`vtk_grid`.
"""
function vtk_cells_linear(space::Spaces.SpectralElementSpace2D)
    Nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    nelems = Topologies.nlocalelems(space)
    N = Nq - 1

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, N * N * nelems)
    ind = LinearIndices((1:Nq, 1:Nq))
    for e in 1:nelems
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

"""
  vtk_space(space::ClimaCore.Spaces.AbstractSpace)

The space on which the VTK output will be stored.

[VTK Lagrange elements](https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/)
require nodes be uniformly spaced within the reference element (see
https://discourse.paraview.org/t/node-positions-of-high-order-lagrange-quadrilateral-cells/7012/3).
This corresponds to the `ClosedUniform` quadrature rule.
"""
function vtk_space(space::Spaces.SpectralElementSpace2D)
    if space.quadrature_style isa Spaces.Quadratures.ClosedUniform
        return space
    end
    Nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    lagrange_quad = Spaces.Quadratures.ClosedUniform{Nq}()
    return Spaces.SpectralElementSpace2D(space.topology, lagrange_quad)
end

vtk_space(field::Fields.Field) = vtk_space(axes(field))
vtk_space(fields::NamedTuple) = vtk_space(first(fields))
vtk_space(fieldvec::Fields.FieldVector) = vtk_space(Fields._values(fieldvec))
