

"""
    cubed_sphere_warp(::EquiangularSphereMesh{FT}, a, b, c)

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equiangular gnomonic grid proposed by
[Ronchi1996]

Returns a tuple of (x1, x2, x3), where x1, x2, x3 are the corresponding coordinates on a sphere
of radius R.

The "Cubed Sphere": A New Method for the Solution of Partial Differential Equations in Spherical Geometry
C. RONCHI,*,1 R. IACONO,* AND P. S. PAOLUCCI†
JOURNAL OF COMPUTATIONAL PHYSICS 124, 93–114 (1996)

https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Topologies.jl
"""
function cubed_sphere_warp(
    ::EquiangularSphere{FT},
    a::FT,
    b::FT,
    c::FT,
) where {FT <: AbstractFloat}

    R = max(abs(a), abs(b), abs(c))
    function f(sR, ξ, η)
        X, Y = tan(π * ξ / 4), tan(π * η / 4)
        ζ1 = sR / sqrt(X^2 + Y^2 + 1)
        ζ2, ζ3 = X * ζ1, Y * ζ1
        ζ1, ζ2, ζ3
    end

    fdim = argmax(abs.((a, b, c)))
    if fdim == 1 && a < 0
        # (-R, *, *) : formulas for Face I from Ronchi, Iacono, Paolucci (1996)
        #              but for us face IV of the developed net of the cube
        x1, x2, x3 = f(-R, b / a, c / a)
    elseif fdim == 2 && b < 0
        # ( *,-R, *) : formulas for Face II from Ronchi, Iacono, Paolucci (1996)
        #              but for us face V of the developed net of the cube
        x2, x1, x3 = f(-R, a / b, c / b)
    elseif fdim == 1 && a > 0
        # ( R, *, *) : formulas for Face III from Ronchi, Iacono, Paolucci (1996)
        #              but for us face II of the developed net of the cube
        x1, x2, x3 = f(R, b / a, c / a)
    elseif fdim == 2 && b > 0
        # ( *, R, *) : formulas for Face IV from Ronchi, Iacono, Paolucci (1996)
        #              but for us face III of the developed net of the cube
        x2, x1, x3 = f(R, a / b, c / b)
    elseif fdim == 3 && c > 0
        # ( *, *, R) : formulas for Face V from Ronchi, Iacono, Paolucci (1996)
        #              but for us face VI of the developed net of the cube
        x3, x2, x1 = f(R, b / c, a / c)
    elseif fdim == 3 && c < 0
        # ( *, *,-R) : formulas for Face VI from Ronchi, Iacono, Paolucci (1996)
        #              but for us face I of the developed net of the cube
        x3, x2, x1 = f(-R, b / c, a / c)
    else
        error(
            "invalid case for cubed_sphere_warp(::EquiangularCubedSphere): $a, $b, $c",
        )
    end
    return x1, x2, x3
end

"""
    cubed_sphere_warp(::EquidistantSphereMesh{FT}, a, b, c)

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equidistant gnomonic grid outlined in
[Rancic1996] and [Nair2005]

Returns a tuple of (x1, x2, x3), where x1, x2, x3 are the corresponding coordinates on a sphere
of radius R.

The "Cubed Sphere": A New Method for the Solution of Partial Differential Equations in Spherical Geometry
C. RONCHI,*,1 R. IACONO,* AND P. S. PAOLUCCI†
JOURNAL OF COMPUTATIONAL PHYSICS 124, 93–114 (1996)

A Discontinuous Galerkin Transport Scheme on the Cubed Sphere
Ramachandran D. Nair1, Stephen J. Thomas1, and Richard D. Loft1
https://doi.org/10.1175/MWR2890.1

https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Topologies.jl
"""
function cubed_sphere_warp(
    ::EquidistantSphere{FT},
    a::FT,
    b::FT,
    c::FT,
) where {FT <: AbstractFloat}
    R = max(abs(a), abs(b), abs(c))
    r = hypot(a, b, c)
    return R * a / r, R * b / r, R * c / r
end
