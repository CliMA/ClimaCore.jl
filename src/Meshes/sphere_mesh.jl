
function Mesh2D(domain::CubePanelDomain{FT}, ne) where {FT <: AbstractFloat}
    return cube_panel_mesh(domain, NoWarp(), ne, FT)
end

function Mesh2D(
    domain::SphereDomain{FT},
    warp_type::AbstractSphereWarp,
    ne,
) where {FT <: AbstractFloat}
    radius = domain.radiuse
    # map the cube to [-radius, radius],[-radius, radius],[-radius, radius]
    mesh = cube_panel_mesh(domain, warp_type, ne, FT)
    mesh.coordinates .-= FT(0.5)
    mesh.coordinates .*= (FT(2) * radius)

    for i in 1:(mesh.nverts)
        mesh.coordinates[i, :] .=
            cubed_sphere_warp(warp_type, mesh.coordinates[i, :]...)
    end
    return mesh
end

"""
    cube_panel_mesh(domain, warp_type, ne, ::Type{FT})

This function builds a cube panel mesh with a resolution of `ne` elements along each edge.

               v8 (xs,xe,xe)          v7 (xe,xe,xe)
                 o--------e11---------o
                /|                   /|
               / |                  / |
              /  |                 /  |
            e12  e8               e10 e7
            /    |               /    |
           /     |            v6/     |
       v5 o--------e9----------o      |
          |    v4o------e3-----|------o v3 (xe,xe,xs)
          |     /   o------->  |     /
          |    /   /           |    /
          e5  e4  /           e6   e2
          |  /   /             |  /
          | /   /              | /
          |/   o               |/
          o--------e1----------o
         v1                    v2
       (xs,xs,xs)               (xe,xs,xs)

       panel 1 => 1 4 3 2
       panel 2 => 2 3 7 6
       panel 3 => 3 4 8 7
       panel 4 => 1 5 8 4
       panel 5 => 1 2 6 5
       panel 6 => 5 6 7 8

       edge  1 => 1 2
       edge  2 => 2 3
       edge  3 => 3 4
       edge  4 => 4 1
       edge  5 => 1 5
       edge  6 => 2 6
       edge  7 => 3 7
       edge  8 => 4 8
       edge  9 => 5 6
       edge 10 => 6 7
       edge 11 => 7 8
       edge 12 => 8 5

           v8 +---e11---+ v7
              | ^       |
              | |       |
             e8 ^   3   e7
              | |       |
   v8      v4 | o--<-o  | v3      v7        v8
    +---e8----+---e3----+----e7---+---e11---+
    | ^       | o-->->  | o-->--> | ^       |
    | |       | |       | |       | |       |
  e12 ^  4   e4 ^  1   e2 ^  2   e10^  6   e12
    | |       | |       | |       | |       |
    | o-<-<-o | o       | o       | o<-<--o |
    +---e5----+---e1----+----e6---+----e9---+
   v5       v1| o-->--o | v2      v6        v5
              |       | |
             e5    5  | e6
              |       ↓ |
              |         |
              +---e9----+
             v5         v6
"""
function cube_panel_mesh(
    domain::Union{CubePanelDomain{FT}, SphereDomain{FT}},
    warp_type::AbstractWarp,
    ne::I,
    ::Type{FT},
) where {FT <: AbstractFloat, I <: Integer}

    xs, xe = FT(0), FT(1)
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
        4 3 4 5 2 6
        3 7 8 8 6 7
        2 6 7 4 5 8
    ]
    panel_edges = [
        4 2 3 5 1 9
        3 7 8 12 6 10
        2 10 11 8 9 11
        1 6 7 4 5 12
    ]
    panel_edges_rev = Bool.([
        1 0 0 0 0 0
        1 0 0 1 0 0
        1 1 1 1 1 0
        1 1 1 0 1 0
    ])
    # node coordinates
    xc = range(xs, xe; step = FT(1 / ne)) # [xs, xs+Δ, xs+2Δ, ..., xe]
    xci = view(xc, 2:ne)                   # [xs+Δ, xs+2Δ, ..., xe-Δ]
    xcir = view(xc, ne:-1:2)                # reverse(xci)
    sc = ones(FT, ne - 1) * xs              # [xs, ...., xs]
    ec = ones(FT, ne - 1) * xe              # [xe, ...., xe]
    # where Δ = (xe-xs)/ne
    sc2 = ones(FT, (ne - 1) * (ne - 1)) * xs
    ec2 = ones(FT, (ne - 1) * (ne - 1)) * xe

    xci12 = repeat(xci, outer = ne - 1)
    xci1r2 = repeat(xcir, outer = ne - 1)
    xci21 = repeat(xci, inner = ne - 1)

    edge_nodes = reshape(1:((ne - 1) * 12), ne - 1, 12) .+ 8
    # coordinates
    coordinates = vcat(
        hcat(
            [xs, xe, xe, xs, xs, xe, xe, xs], # x1,
            [xs, xs, xe, xe, xs, xs, xe, xe], # x2,
            [xs, xs, xs, xs, xe, xe, xe, xe], # x3 vertex coordinates
        ),
        vcat(
            hcat(xci, sc, sc), # edge 1
            hcat(ec, xci, sc), # edge 2
            hcat(xcir, ec, sc), # edge 3
            hcat(sc, xcir, sc), # edge 4
            hcat(sc, sc, xci), # edge 5
            hcat(ec, sc, xci), # edge 6
            hcat(ec, ec, xci), # edge 7
            hcat(sc, ec, xci), # edge 8
            hcat(xci, sc, ec), # edge 9
            hcat(ec, xci, ec), # edge 10
            hcat(xcir, ec, ec), # edge 11
            hcat(sc, xcir, ec), # edge 12
        ),
        hcat(xci21, xci12, sc2), # panel 1
        hcat(ec2, xci12, xci21), # panel 2
        hcat(xci1r2, ec2, xci21), # panel 3
        hcat(sc2, xci21, xci12), # panel 4
        hcat(xci12, sc2, xci21), # panel 5
        hcat(xci12, xci21, ec2), # panel 6
    )
    face_interior = reshape(1:((ne - 1) * (ne - 1)), ne - 1, ne - 1)

    nfc1i = (nx - 2) * (nx - 1) # panels with normals along local first direction
    nfc2i = (nx - 1) * (nx - 2) # panels with normals along local second direction
    nfci = nfc1i + nfc2i
    fci1 = reshape(1:nfc1i, nx - 2, nx - 1)
    fci2 = reshape(1:nfc2i, nx - 1, nx - 2)

    fcmat1 = zeros(I, nx, nx - 1) # face numbering
    fcmat2 = zeros(I, nx - 1, nx) # for each panel

    edge_faces = reshape(1:(12 * ne), ne, 12)

    face_verts = zeros(I, nfaces, 2)
    face_neighbors = zeros(I, nfaces, 5)
    face_boundary = Vector{I}(zeros(nfaces)) # all interior nodes (no boundaries)
    elem_verts = zeros(I, nelems, 4)
    elem_faces = zeros(I, nelems, 4)


    for sfc in 1:6
        ndmat[1, 1],
        ndmat[ne + 1, 1],  # panel vertices
        ndmat[ne + 1, ne + 1],
        ndmat[1, ne + 1] = panel_verts[:, sfc]

        ndmat[2:ne, 1] .=
            panel_edges_rev[1, sfc] ?
            reverse(edge_nodes[:, panel_edges[1, sfc]]) :
            edge_nodes[:, panel_edges[1, sfc]]  # panel edges

        ndmat[end, 2:ne] .=
            panel_edges_rev[2, sfc] ?
            reverse(edge_nodes[:, panel_edges[2, sfc]]) :
            edge_nodes[:, panel_edges[2, sfc]]

        ndmat[ne:-1:2, end] .=
            panel_edges_rev[3, sfc] ?
            reverse(edge_nodes[:, panel_edges[3, sfc]]) :
            edge_nodes[:, panel_edges[3, sfc]]

        ndmat[1, ne:-1:2, end] .=
            panel_edges_rev[4, sfc] ?
            reverse(edge_nodes[:, panel_edges[4, sfc]]) :
            edge_nodes[:, panel_edges[4, sfc]]

        offset = 8 + 12 * (ne - 1) + (sfc - 1) * (ne - 1) * (ne - 1) # interior
        ndmat[2:ne, 2:ne] .= face_interior .+ offset

        fcmat1[1, end:-1:1] .=
            panel_edges_rev[4, sfc] ?
            reverse(edge_faces[:, panel_edges[4, sfc]]) :
            edge_faces[:, panel_edges[4, sfc]]

        fcmat1[end, :] .=
            panel_edges_rev[2, sfc] ?
            reverse(edge_faces[:, panel_edges[2, sfc]]) :
            edge_faces[:, panel_edges[2, sfc]]

        fcmat2[:, 1] .=
            panel_edges_rev[1, sfc] ?
            reverse(edge_faces[:, panel_edges[1, sfc]]) :
            edge_faces[:, panel_edges[1, sfc]]

        fcmat2[end:-1:1, end] .=
            panel_edges_rev[3, sfc] ?
            reverse(edge_faces[:, panel_edges[3, sfc]]) :
            edge_faces[:, panel_edges[3, sfc]]

        off = ne * 12 + (sfc - 1) * nfci

        fcmat1[2:(end - 1), :] .= fci1 .+ off
        fcmat2[:, 2:(end - 1)] .= fci2 .+ (off + nfc1i)

        face_verts[fcmat1[:], 1] .= ndmat[:, 1:ne][:] # face nodes
        face_verts[fcmat1[:], 2] .= ndmat[:, 2:(ne + 1)][:]
        face_verts[fcmat2[:], 1] .= ndmat[1:ne, :][:]
        face_verts[fcmat2[:], 2] .= ndmat[2:(ne + 1), :][:]

        if sfc == 1
            bdy1 = emat[:, 1:1, 5]'
            bdy2 = emat[ne:-1:1, 1:1, 3]'
            bdy3 = emat[1:1, 1:ne, 4]'
            bdy4 = emat[:, 1:1, 2]
        elseif sfc == 2
            bdy1 = emat[ne:ne, :, 5]
            bdy2 = emat[1:1, :, 3]
            bdy3 = emat[:, ne:ne, 1]
            bdy4 = emat[ne:ne, :, 6]'
        elseif sfc == 3
            bdy1 = emat[ne:ne, :, 2]
            bdy2 = emat[:, ne:ne, 4]'
            bdy3 = emat[ne:ne, ne:-1:1, 1]'
            bdy4 = emat[ne:-1:1, ne:ne, 6]
        elseif sfc == 4
            bdy1 = emat[:, 1:1, 1]'
            bdy2 = emat[1:1, :, 6]
            bdy3 = emat[1:1, :, 5]'
            bdy4 = emat[ne:ne, :, 3]'
        elseif sfc == 5
            bdy1 = emat[:, 1:1, 4]'
            bdy2 = emat[1:1, :, 2]
            bdy3 = emat[1:1, :, 1]'
            bdy4 = emat[:, 1:1, 6]
        else # sfc == 6
            bdy1 = emat[ne:ne, :, 4]
            bdy2 = emat[:, ne:ne, 2]'
            bdy3 = emat[:, ne:ne, 5]
            bdy4 = emat[ne:-1:1, ne:ne, 3]
        end
        face_neighbors[fcmat1[:], 1] .= vcat(bdy1, emat[:, :, sfc])[:]
        face_neighbors[fcmat1[:], 3] .= vcat(emat[:, :, sfc], bdy2)[:]
        face_neighbors[fcmat2[:], 1] .= hcat(bdy3, emat[:, :, sfc])[:]
        face_neighbors[fcmat2[:], 3] .= hcat(emat[:, :, sfc], bdy4)[:]

        elem_verts[emat[:, :, sfc][:], :] .= hcat(
            ndmat[1:ne, 1:ne][:], # node numbers (local node 1)
            ndmat[2:(ne + 1), 1:ne][:], # for each element (local node 2)
            ndmat[2:(ne + 1), 2:(ne + 1)][:], #(local node 3)
            ndmat[1:ne, 2:(ne + 1)][:], # (local node 4)
        )

        elem_faces[emat[:, :, sfc][:], :] .= hcat(
            fcmat2[:, 1:(nx - 1)][:], # (local face 1)
            fcmat1[2:nx, :][:],   # each element (local face 2)
            fcmat2[:, 2:nx][:], # (local face 3)
            fcmat1[1:(nx - 1), :][:], # face numbers for (local face 4)
        )
    end

    ref_fc_verts = [
        1 2 3 4
        2 3 4 1
    ]

    for fc in 1:nfaces
        elems = (face_neighbors[fc, 1], face_neighbors[fc, 3])
        for e in 1:2
            el = elems[e]
            if el ≠ 0
                localface = findfirst(elem_faces[el, :] .== fc)
                if isnothing(localface)
                    error(
                        "rectangular_mesh: Fatal error, face could not be located in neighboring element;\n",
                        "el = $el;\n",
                        "elem_faces[$el, :] = $(elem_faces[el, :]);\n",
                        "fc = $fc;\n",
                        "face_neighbors[$fc,:] = $(face_neighbors[fc,:]);\n",
                        "elem_faces[$(elems[1]), :] = $(elem_faces[elems[1], :]);\n",
                        "elem_faces[$(elems[2]), :] = $(elem_faces[elems[2], :]);\n",
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
        domain,
        warp_type,
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
