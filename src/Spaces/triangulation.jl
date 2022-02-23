function triangles(Ni, Nj, Nh)
    L = LinearIndices((1:Ni, 1:Nj))
    I = vec([
        (t == 1 ? L[i, j] : L[i + 1, j]) + Ni * Nj * (h - 1) for t in 1:2,
        i in 1:(Ni - 1), j in 1:(Nj - 1), h in 1:Nh
    ])
    J = vec([
        (t == 1 ? L[i + 1, j] : L[i + 1, j + 1]) + Ni * Nj * (h - 1) for
        t in 1:2, i in 1:(Ni - 1), j in 1:(Nj - 1), h in 1:Nh
    ])
    K = vec([
        (t == 1 ? L[i, j + 1] : L[i, j + 1]) + Ni * Nj * (h - 1) for
        t in 1:2, i in 1:(Ni - 1), j in 1:(Nj - 1), h in 1:Nh
    ])
    return (I, J, K)
end

function triangulate(space::SpectralElementSpace2D)
    Ni, Nj, _, _, Nh = size(space.local_geometry)
    return triangles(Ni, Nj, Nh)
end

function triangulate(space::ExtrudedFiniteDifferenceSpace)
    Ni, Nj, _, Nv, Nh = size(space.local_geometry)
    @assert Nj == 1 "triangulation only defined for 1D extruded fields"
    return triangles(Ni, Nv, Nh)
end
