"""
    rectangular_mesh(x1c, x2c, per = (false, false))

This function builds a 2D rectangular mesh with points located at x1c and x2c in x1 and x2 directions respectively.
"""
function rectangular_mesh(x1c, x2c, per = (false, false))
    FT = eltype(x1c)
    I = Int

    nx1, nx2 = length(x1c), length(x2c)
    nel1, nel2 = nx1 - 1, nx2 - 1

    nverts = nx1 * nx2
    nfaces = nx1 * (nx2 - 1) + (nx1 - 1) * nx2
    nelems = (nx1 - 1) * (nx2 - 1)
    nbndry = 4

    nfc1 = nx1 * (nx2 - 1) # faces with normals along x1 direction 

    ndmat = reshape(1:nverts, nx1, nx2)

    fcmat1 = reshape(1:nfc1, nx1, nx2 - 1)        # faces with normals along x1 direction
    fcmat2 = reshape((nfc1 + 1):nfaces, nx1 - 1, nx2) # faces with normals along x2 direction

    emat = reshape(1:nelems, nx1 - 1, nx2 - 1)

    coordinates = [repeat(x1c, 1, nx2)[:] repeat(x2c', nx1, 1)[:]] # node coordinates

    face_verts = hcat(
        vcat(ndmat[:, 1:(nx2 - 1)][:], ndmat[1:(nx1 - 1), :][:]), # face nodes
        vcat(ndmat[:, 2:nx2][:], ndmat[2:nx1, :][:]),
    )

    face_bndry = zeros(I, nfaces)
    # accounting for periodic boundaries, wherever applicable
    if per[1]
        bdy1 = emat[end:end, :]
        bdy2 = emat[1:1, :]
        nbndry -= 2
    else
        bdy1 = bdy2 = zeros(I, 1, nx2 - 1)
        face_bndry[fcmat1[1, :]] .= 1  # left boundary
        face_bndry[fcmat1[end, :]] .= 2  # right boundary
    end
    if per[2]
        bdy3 = emat[:, end:end]
        bdy4 = emat[:, 1:1]
        nbndry -= 2
    else
        bdy3 = bdy4 = zeros(I, nx1 - 1, 1)
        face_bndry[fcmat2[:, 1]] .= 3  # bottom boundary
        face_bndry[fcmat2[:, end]] .= 4  # top boundary
    end

    face_neighbors = hcat(
        vcat(vcat(bdy1, emat)[:], hcat(bdy3, emat)[:]),
        vcat(vcat(emat, bdy2)[:], hcat(emat, bdy4)[:]),
    )

    elem_verts = hcat(
        ndmat[1:(nx1 - 1), 1:(nx2 - 1)][:], # node numbers
        ndmat[2:nx1, 1:(nx2 - 1)][:], # for each element
        ndmat[1:(nx1 - 1), 2:nx2][:],
        ndmat[2:nx1, 2:nx2][:],
    )
    elem_faces = hcat(
        fcmat1[1:(nx1 - 1), :][:], # face numbers for 
        fcmat1[2:nx1, :][:],   # each element
        fcmat2[:, 1:(nx2 - 1)][:],
        fcmat2[:, 2:nx2][:],
    )

    return Mesh2D(
        nverts,
        nfaces,
        nelems,
        nbndry,
        coordinates,
        face_verts,
        face_neighbors,
        face_bndry,
        elem_verts,
        elem_faces,
    )
end
