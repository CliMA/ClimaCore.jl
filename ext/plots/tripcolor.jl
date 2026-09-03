# This file was copied verbatim from
# https://github.com/pazner/TriplotBase.jl/blob/5dd516a346299101523e91e2ebf33aff3d05845f/src/tripcolor.jl#L1-L97
# TriplotBase.jl is licensed under the Unlicense license which is a license with
# no conditions. This was done to avoid adding TriplotBase.jl as a dependency
# and to maintain it if needed.

"""
    dgtripcolor(x, y, z, t; zmin=nothing, zmax=nothing)

Draw a pseudocolor plot of discontinuous unstructured triangular data. `x`, and `y`, are
one-dimensional input arrays of the same length (the number of points). `t` is an integer array of
size `(3,nt)`, where `nt` is the number of triangles. `z` is an array of the same size as `t`
representing function values on each triangle (discontinuous across triangle edges). The coordinates
of the `i`th vertex of the `j`th triangle are given by `(x[t[i,j]], y[t[i,j]])`. The function value
at the `i`th vertex of the `j`th triangle is given by `z[i,j]`.

The colormap is specified by `cmap`. The colormap bounds can be set by specifying values for `zmin`
and `zmax`. If these parameters are set to `nothing`, then the minimum and maximum values of the `z`
array will be used.

For plotting continuous fields, see [`tripcolor`](@ref).
"""
function dgtripcolor(
    x,
    y,
    z,
    t,
    cmap::AbstractVector{T};
    px,
    py,
    zmin = nothing,
    zmax = nothing,
    yflip = false,
    bg = nothing,
) where {T}
    nv = length(x)
    nt = size(t, 2)
    @assert (
        size(x) == (nv,) && size(y) == (nv,) && size(z) == (3, nt) && size(t) == (3, nt)
    ) "Bad input sizes"

    c = isnothing(bg) ? Matrix{T}(undef, px, py) : fill(bg, px, py)
    nc = length(cmap)

    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    w = xmax-xmin
    h = ymax-ymin
    isnothing(zmin) && (zmin = minimum(z))
    isnothing(zmax) && (zmax = maximum(z))

    for it in 1:nt
        xt = @views x[t[:, it]]
        yt = @views y[t[:, it]]
        zt = @views z[:, it]

        detT = (yt[2]-yt[3])*(xt[1]-xt[3]) + (xt[3]-xt[2])*(yt[1]-yt[3])

        xmint, xmaxt = extrema(xt)
        ymint, ymaxt = extrema(yt)

        x0 = floor(Int, (xmint-xmin)/w*px)
        x1 = ceil(Int, (xmaxt-xmin)/w*px)
        y0 = floor(Int, (ymint-ymin)/h*py)
        y1 = ceil(Int, (ymaxt-ymin)/h*py)

        # Loop over all pixels in triangle's bounding box
        for i in x0:(x1 - 1), j in y0:(y1 - 1)
            # Cartesian coordinates of pixel (i,j) (zero-indexed)
            cx = xmin + i/px*w
            cy = ymin + j/py*h
            # Convert from Cartesian to barycentric coordinates
            λ1 = ((yt[2] - yt[3])*(cx-xt[3]) + (xt[3]-xt[2])*(cy-yt[3]))/detT
            λ2 = ((yt[3] - yt[1])*(cx-xt[3]) + (xt[1]-xt[3])*(cy-yt[3]))/detT
            λ3 = 1 - λ1 - λ2
            tol = 1e-12
            # If pixel is outside of triangle, move on to next pixel
            (λ1>-tol && λ2<1+tol && λ2>-tol && λ2<1+tol && λ3>-tol && λ3<1+tol) || continue
            zval = zt[1]*λ1 + zt[2]*λ2 + zt[3]*λ3
            zval = clamp(zval, zmin, zmax)
            c[i + 1, yflip ? py-j : j+1] = cmap[begin + getcolorind(zval, zmin, zmax, nc)]
        end
    end
    c
end

"""
    tripcolor(x, y, z, t; zmin=nothing, zmax=nothing)

Draw a pseudocolor plot of unstructured triangular data. `x`, `y`, and `z` are
one-dimensional input arrays of the same length (the number of points). `t` is an integer
array of size `(3,nt)`, where `nt` is the number of triangles. The coordinates of the `i`th
vertex of the `j`th triangle are given by `(x[t[i,j]], y[t[i,j]])`. The function value at the `j`th
vertex of the `i`th triangle is given by `z[t[i,j]]`.

The colormap is specified by `cmap`. The colormap bounds can be set by specifying values for `zmin`
and `zmax`. If these parameters are set to `nothing`, then the minimum and maximum values of the `z`
array will be used.

For plotting discontinuous fields, see [`dgtripcolor`](@ref).
"""
function tripcolor(
    x,
    y,
    z,
    t,
    cmap::AbstractVector{T};
    px,
    py,
    zmin = nothing,
    zmax = nothing,
    yflip = false,
    bg = nothing,
) where {T}
    nv = length(x)
    nt = size(t, 2)
    @assert (size(x) == (nv,) && size(y) == (nv,) && size(z) == (nv,) && size(t) == (3, nt)) "Bad input sizes"

    dg_z = similar(t, eltype(z))
    for it in 1:nt
        dg_z[:, it] .= @views z[t[:, it]]
    end
    dgtripcolor(x, y, dg_z, t, cmap; px, py, zmin, zmax, yflip, bg)
end

function getcolorind(level, lmin, lmax, nc)
    cfrac = (lmin == lmax) ? 0 : (level-lmin)/(lmax-lmin)
    round(Int, cfrac*(nc-1))
end
