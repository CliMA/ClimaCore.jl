
function Mesh2D(
    stype::AbstractSphere{FT},
    ne,
    radius,
) where {FT <: AbstractFloat}
    # map the cube to [-radius, radius],[-radius, radius],[-radius, radius]
    mesh = cube_panel_mesh(ne, FT)
    mesh.coordinates .-= FT(0.5)
    mesh.coordinates .*= (FT(2) * radius)

    for i in 1:(mesh.nverts)
        mesh.coordinates[i, :] .=
            cubed_sphere_warp(stype, mesh.coordinates[i, :]...)
    end
    return mesh
end

"""
    cube_panel_mesh(ne)

This function builds a cube panel mesh with a resolution of `ne` elements along each edge.

                v7                     v8 (1,1,1)
                 o--------e4----------o       
                /|                   /|                                   
               / |                  / |       
              /  |                 /  |       
             e7  e11              e8 e12       
            /    |               /    |       
           /     |            v6/     |       
       v5 o--------e3----------o      |       
          |    v3o------e2-----|------o v4 (1,1,0)
          |     /              |     / 
          |    /               |    / 
          e9  e5              e10  e6 
          |  /                 |  / 
          | /                  | / 
          |/                   |/ 
          o--------e1----------o   
         v1                    v2 
       (0,0,0)               (1,0,0)

       Partitioning with Space-Filling Curves on the Cubed-Sphere - John M. Dennis  (2003 iEEE)
       https://www.researchgate.net/publication/224742710_Partitioning_with_space-filling_curves_on_the_cubed-sphere

       panel 1 => 1 2 3 4
       panel 2 => 2 4 6 8
       panel 3 => 3 4 7 8
       panel 4 => 1 3 5 7
       panel 5 => 1 2 5 6
       panel 6 => 5 6 7 8

            +-------+
            |       |
            |   3   |
            |       |
    +-------+-------+-------+-------+
    |       |       |       |       |
    |   4   |   1   |   2   |   6   |
    |       |       |       |       |
    +-------+-------+-------+-------+
            |       |
            |   5   |
            |       |
            +-------+
"""
function cube_panel_mesh(
    ne::I,
    ::Type{FT},
) where {FT <: AbstractFloat, I <: Integer}

    nverts = (ne + 1)^3 - (ne - 1)^3
    nfaces = 12 * ne + 6 * (2 * ne * (ne - 1))
    nelems = 6 * ne * ne
    nbndry = 0

    nx = ne + 1

    nfaces_edg = 12 * ne

    emat = reshape(1:nelems, ne, ne, 6)
    ndmat = zeros(I, ne + 1, ne + 1)

    panel_verts = [
        1 2 3 1 1 5
        2 4 4 3 2 6
        3 6 7 5 5 7
        4 8 8 7 6 8
    ]


    edge_nodes = reshape(1:((ne - 1) * 12), ne - 1, 12) .+ 8

    panel_edges = [
        5 10 11 9 9 7
        6 12 12 11 10 8
        1 6 2 5 1 3
        2 8 4 7 3 4
    ]

    face_interior = reshape(1:((ne - 1) * (ne - 1)), ne - 1, ne - 1)

    nfc1i = (nx - 2) * (nx - 1) # panels with normals along first direction 
    nfc2i = (nx - 1) * (nx - 2) # panels with normals along second direction 
    nfci = nfc1i + nfc2i
    fci1 = reshape(1:nfc1i, nx - 2, nx - 1)
    fci2 = reshape(1:nfc2i, nx - 1, nx - 2)

    fcmat1 = zeros(I, nx, nx - 1) # face numbering
    fcmat2 = zeros(I, nx - 1, nx) # for each panel 

    edge_faces = reshape(1:(12 * ne), ne, 12)

    # node coordinates
    xc = range(FT(0), FT(1); step = FT(1 / ne))
    xci = view(xc, 2:ne)
    zc = zeros(FT, ne - 1)
    oc = ones(FT, ne - 1)
    zc2 = zeros(FT, (ne - 1) * (ne - 1))
    oc2 = ones(FT, (ne - 1) * (ne - 1))

    xci12 = repeat(xci, ne - 1)
    xci21 = repeat(xci', ne - 1, 1)[:]

    coordinates = vcat(
        hcat(
            [0, 1, 0, 1, 0, 1, 0, 1],  # x1,
            [0, 0, 1, 1, 0, 0, 1, 1],  # x2,
            [0, 0, 0, 0, 1, 1, 1, 1],
        ), # x3 vertex coordinates
        vcat(
            hcat(xci, zc, zc),    # edge  1  
            hcat(xci, oc, zc),    # edge  2
            hcat(xci, zc, oc),    # edge  3
            hcat(xci, oc, oc),
        ),   # edge  4
        vcat(
            hcat(zc, xci, zc),    # edge  5
            hcat(oc, xci, zc),    # edge  6
            hcat(zc, xci, oc),    # edge  7
            hcat(oc, xci, oc),
        ),   # edge  8
        vcat(
            hcat(zc, zc, xci),    # edge  9
            hcat(oc, zc, xci),    # edge 10
            hcat(zc, oc, xci),    # edge 11
            hcat(oc, oc, xci),
        ),   # edge 12
        hcat(xci12, xci21, zc2),   # panel 1
        hcat(oc2, xci12, xci21),   # panel 2
        hcat(xci12, oc2, xci21),   # panel 3
        hcat(zc2, xci12, xci21),   # panel 4
        hcat(xci12, zc2, xci21),   # panel 5
        hcat(xci12, xci21, oc2),   # panel 6
    )

    face_verts = zeros(I, nfaces, 2)
    face_neighbors = zeros(I, nfaces, 5)
    face_boundary = Vector{I}(zeros(nfaces)) # all interior nodes (no boundaries)
    elem_verts = zeros(I, nelems, 4)
    elem_faces = zeros(I, nelems, 4)

    for sfc in 1:6
        ndmat[1, 1],
        ndmat[ne + 1, 1],  # panel vertices
        ndmat[1, ne + 1],
        ndmat[ne + 1, ne + 1] = panel_verts[:, sfc]

        ndmat[1, 2:ne] .= edge_nodes[:, panel_edges[1, sfc]] # panel edges
        ndmat[end, 2:ne] .= edge_nodes[:, panel_edges[2, sfc]]
        ndmat[2:ne, 1] .= edge_nodes[:, panel_edges[3, sfc]]
        ndmat[2:ne, end] .= edge_nodes[:, panel_edges[4, sfc]]

        offset = 8 + 12 * (ne - 1) + (sfc - 1) * (ne - 1) * (ne - 1) # interior
        ndmat[2:ne, 2:ne] .= face_interior .+ offset

        fcmat1[1, :] = edge_faces[:, panel_edges[1, sfc]]
        fcmat1[end, :] = edge_faces[:, panel_edges[2, sfc]]
        fcmat2[:, 1] = edge_faces[:, panel_edges[3, sfc]]
        fcmat2[:, end] = edge_faces[:, panel_edges[4, sfc]]
        off = ne * 12 + (sfc - 1) * nfci
        fcmat1[2:(end - 1), :] .= fci1 .+ off
        fcmat2[:, 2:(end - 1)] .= fci2 .+ (off + nfc1i)

        face_verts[fcmat1[:], 1] .= ndmat[:, 1:ne][:] # face nodes
        face_verts[fcmat1[:], 2] .= ndmat[:, 2:(ne + 1)][:]
        face_verts[fcmat2[:], 1] .= ndmat[1:ne, :][:]
        face_verts[fcmat2[:], 2] .= ndmat[2:(ne + 1), :][:]

        if sfc == 1
            bdy1 = emat[:, 1:1, 4]'
            bdy2 = emat[:, 1:1, 2]'
            bdy3 = emat[:, 1:1, 5]
            bdy4 = emat[:, 1:1, 3]
        elseif sfc == 2
            bdy1 = emat[end:end, :, 5]
            bdy2 = emat[end:end, :, 3]
            bdy3 = emat[end:end, :, 1]'
            bdy4 = emat[end:end, :, 6]'
        elseif sfc == 3
            bdy1 = emat[end:end, :, 4]
            bdy2 = emat[end:end, :, 2]
            bdy3 = emat[:, end:end, 1]
            bdy4 = emat[:, end:end, 6]
        elseif sfc == 4
            bdy1 = emat[1:1, :, 5]
            bdy2 = emat[1:1, :, 3]
            bdy3 = emat[1:1, :, 1]'
            bdy4 = emat[1:1, :, 6]'
        elseif sfc == 5
            bdy1 = emat[1:1, :, 4]
            bdy2 = emat[1:1, :, 2]
            bdy3 = emat[:, 1:1, 1]
            bdy4 = emat[:, 1:1, 6]
        else # sfc == 6
            bdy1 = emat[:, end:end, 4]'
            bdy2 = emat[:, end:end, 2]'
            bdy3 = emat[:, end:end, 5]
            bdy4 = emat[:, end:end, 3]
        end

        face_neighbors[fcmat1[:], 1] .= vcat(bdy1, emat[:, :, sfc])[:]
        face_neighbors[fcmat1[:], 3] .= vcat(emat[:, :, sfc], bdy2)[:]
        face_neighbors[fcmat2[:], 1] .= hcat(bdy3, emat[:, :, sfc])[:]
        face_neighbors[fcmat2[:], 3] .= hcat(emat[:, :, sfc], bdy4)[:]

        elem_verts[emat[:, :, sfc][:], :] .= hcat(
            ndmat[1:ne, 1:ne][:], # node numbers
            ndmat[2:(ne + 1), 1:ne][:], # for each element
            ndmat[1:ne, 2:(ne + 1)][:],
            ndmat[2:(ne + 1), 2:(ne + 1)][:],
        )

        elem_faces[emat[:, :, sfc][:], :] .= hcat(
            fcmat1[1:(nx - 1), :][:], # face numbers for 
            fcmat1[2:nx, :][:],   # each element
            fcmat2[:, 1:(nx - 1)][:],
            fcmat2[:, 2:nx][:],
        )
    end

    ref_fc_verts = [1 2 1 3; 3 4 2 4]


    for fc in 1:nfaces
        elems = (face_neighbors[fc, 1], face_neighbors[fc, 3])
        for e in 1:2
            el = elems[e]
            if el ≠ 0
                localface = findfirst(elem_faces[el, :] .== fc)
                if isnothing(localface)
                    error(
                        "rectangular_mesh: Fatal error, face could not be located in neighboring element",
                    )
                else
                    face_neighbors[fc, 2 + (e - 1) * 2] = localface
                end
            end
        end
        # setting up relative orientation
        or1 =
            elem_verts[elems[1], ref_fc_verts[:, face_neighbors[fc, 2]]] ==
            face_verts[fc, :] ? 1 : -1
        or2 =
            elem_verts[elems[2], ref_fc_verts[:, face_neighbors[fc, 4]]] ==
            face_verts[fc, :] ? 1 : -1
        face_neighbors[fc, 5] = or1 * or2
    end

    boundary_tags = sort(unique(face_boundary))
    face_boundary_offset = ones(I, length(boundary_tags) + 1)
    face_boundary_tags = sort(face_boundary)
    face_boundary = sortperm(face_boundary)

    for i in 1:(length(boundary_tags) - 1)
        tag = boundary_tags[i + 1]
        face_boundary_offset[i + 1] = findfirst(face_boundary_tags .== tag)
    end
    face_boundary_offset[end] = length(face_boundary) + 1

    boundary_tag_names = (:interior,)

    # add unique vertex iterator information
    vtconn = map(i -> zeros(I, i), zeros(I, nverts))

    for el in 1:nelems
        for lv in 1:4
            vt = elem_verts[el, lv]
            push!(vtconn[vt], el, lv)
        end
    end
    unique_verts = I[]
    uverts_conn = I[]
    uverts_offset = I.([1])
    for vt in 1:nverts
        lconn = length(vtconn[vt])
        if lconn > 0
            push!(unique_verts, vt)
            push!(uverts_conn, vtconn[vt]...)
            push!(uverts_offset, uverts_offset[end] + lconn)
        end
    end


    return Mesh2D(
        nverts,
        nfaces,
        nelems,
        nbndry,
        coordinates,
        unique_verts,
        uverts_conn,
        uverts_offset,
        face_verts,
        face_neighbors,
        face_boundary,
        boundary_tags,
        boundary_tag_names,
        face_boundary_offset,
        elem_verts,
        elem_faces,
    )
end
