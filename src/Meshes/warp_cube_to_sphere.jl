using CubedSphere, Rotations

"""
    cubed_sphere_warp(::EquiangularSphereWarp, a, b, c)

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equiangular gnomonic grid proposed by
[Ronchi1996]

Returns a tuple of (x1, x2, x3), where x1, x2, x3 are the corresponding coordinates on a sphere
of radius R.

The "Cubed Sphere": A New Method for the Solution of Partial Differential Equations in Spherical Geometry
C. RONCHI, R. IACONO, AND P. S. PAOLUCCI
JOURNAL OF COMPUTATIONAL PHYSICS 124, 93–114 (1996)

Source code:
https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Topologies.jl
"""
function cubed_sphere_warp(
    ::EquiangularSphereWarp,
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
    equiangular_sphere_warp(a, b, c)
A wrapper function for the cubed_sphere_warp function, when called with the
EquiangularSphereWarp type
"""
equiangular_sphere_warp(a, b, c) =
    cubed_sphere_warp(EquiangularSphereWarp(), a, b, c)

"""
    cubed_sphere_unwarp(x1, x2, x3)

The inverse of [`cubed_sphere_warp`](@ref). This function projects
a given point `(x_1, x_2, x_3)` from the surface of a sphere onto a cube
"""
function cubed_sphere_unwarp(::EquiangularSphereWarp, x1, x2, x3)

    function g(R, X, Y)
        ξ = atan(X) * 4 / pi
        η = atan(Y) * 4 / pi
        R, R * ξ, R * η
    end

    R = hypot(x1, x2, x3)
    fdim = argmax(abs.((x1, x2, x3)))

    if fdim == 1 && x1 < 0
        # (-R, *, *) : formulas for Face I from Ronchi, Iacono, Paolucci (1996)
        #              but for us face II of the developed net of the cube
        a, b, c = g(-R, x2 / x1, x3 / x1)
    elseif fdim == 2 && x2 < 0
        # ( *,-R, *) : formulas for Face II from Ronchi, Iacono, Paolucci (1996)
        #              but for us face III of the developed net of the cube
        b, a, c = g(-R, x1 / x2, x3 / x2)
    elseif fdim == 1 && x1 > 0
        # ( R, *, *) : formulas for Face III from Ronchi, Iacono, Paolucci (1996)
        #              but for us face IV of the developed net of the cube
        a, b, c = g(R, x2 / x1, x3 / x1)
    elseif fdim == 2 && x2 > 0
        # ( *, R, *) : formulas for Face IV from Ronchi, Iacono, Paolucci (1996)
        #              but for us face I of the developed net of the cube
        b, a, c = g(R, x1 / x2, x3 / x2)
    elseif fdim == 3 && x3 > 0
        # ( *, *, R) : formulas for Face V from Ronchi, Iacono, Paolucci (1996)
        #              and the same for us on the developed net of the cube
        c, b, a = g(R, x2 / x3, x1 / x3)
    elseif fdim == 3 && x3 < 0
        # ( *, *,-R) : formulas for Face VI from Ronchi, Iacono, Paolucci (1996)
        #              and the same for us on the developed net of the cube
        c, b, a = g(-R, x2 / x3, x1 / x3)
    else
        error(
            "invalid case for cubed_sphere_unwarp(::EquiangularSphereWarp): $a, $b, $c",
        )
    end

    return a, b, c
end

"""
    equiangular_sphere_unwarp(x1, x2, x3)

A wrapper function for the cubed_sphere_unwarp function, when called with the
EquiangularSphereWarp type
"""
equiangular_sphere_unwarp(x1, x2, x3) =
    cubed_sphere_unwarp(EquiangularSphereWarp(), x1, x2, x3)

"""
    cubed_sphere_warp(::EquidistantSphereWarp, a, b, c)

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equidistant gnomonic grid outlined in
[Rancic1996] and [Nair2005]

Returns a tuple of (x1, x2, x3), where x1, x2, x3 are the corresponding coordinates on a sphere
of radius R.

A global shallow-water model using an expanded spherical cube: Gnomonic
versus conformal coordinates
Rančić M., Purser R. J., Mesinger F.
https://doi.org/10.1002/qj.49712253209

A Discontinuous Galerkin Transport Scheme on the Cubed Sphere
Ramachandran D. Nair, Stephen J. Thomas, and Richard D. Loft
https://doi.org/10.1175/MWR2890.1

Source code:
https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Topologies.jl
"""
function cubed_sphere_warp(
    ::EquidistantSphereWarp,
    a::FT,
    b::FT,
    c::FT,
) where {FT <: AbstractFloat}
    R = max(abs(a), abs(b), abs(c))
    r = hypot(a, b, c)
    return R * a / r, R * b / r, R * c / r
end

"""
    equidistant_sphere_warp(a, b, c)
A wrapper function for the cubed_sphere_warp function, when called with the
EquidistantSphereWarp type
"""
equidistant_sphere_warp(a, b, c) =
    cubed_sphere_warp(EquidistantSphereWarp(), a, b, c)

"""
    cubed_sphere_unwarp(x1, x2, x3)

The inverse of [`cubed_sphere_warp`](@ref). This function projects
a given point `(x_1, x_2, x_3)` from the surface of a sphere onto a cube
"""
function cubed_sphere_unwarp(::EquidistantSphereWarp, x1, x2, x3)

    r = hypot(1, x2 / x1, x3 / x1)
    R = hypot(x1, x2, x3)

    a = r * x1
    b = r * x2
    c = r * x3

    m = max(abs(a), abs(b), abs(c))

    return a * R / m, b * R / m, c * R / m
end

"""
    equidistant_sphere_unwarp(x1, x2, x3)

A wrapper function for the cubed_sphere_unwarp function, when called with the
    EquidistantSphereWarp type
"""
equidistant_sphere_unwarp(x1, x2, x3) =
    cubed_sphere_unwarp(EquidistantSphereWarp(), x1, x2, x3)

"""
    cubed_sphere_warp(::ConformalSphereWarp, a, b, c)

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the condormal mapping outlined in
[Rancic1996]

Returns a tuple of (x1, x2, x3), where x1, x2, x3 are the corresponding coordinates on a sphere
of radius R.

A global shallow-water model using an expanded spherical cube: Gnomonic
versus conformal coordinates
Rančić M., Purser R. J., Mesinger F.
https://doi.org/10.1002/qj.49712253209

Source code:
https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Topologies.jl
"""
function cubed_sphere_warp(
    ::ConformalSphereWarp,
    a::FT,
    b::FT,
    c::FT,
) where {FT <: AbstractFloat}
    R = max(abs(a), abs(b), abs(c))
    fdim = argmax(abs.((a, b, c)))
    M = max(abs.((a, b, c))...)
    if fdim == 1 && a < 0
        # left face
        x1, x2, x3 = conformal_cubed_sphere_mapping(-b / M, c / M)
        x1, x2, x3 = RotX(π / 2) * RotY(-π / 2) * [x1, x2, x3]
    elseif fdim == 2 && b < 0
        # front face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, c / M)
        x1, x2, x3 = RotX(π / 2) * [x1, x2, x3]
    elseif fdim == 1 && a > 0
        # right face
        x1, x2, x3 = conformal_cubed_sphere_mapping(b / M, c / M)
        x1, x2, x3 = RotX(π / 2) * RotY(π / 2) * [x1, x2, x3]
    elseif fdim == 2 && b > 0
        # back face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, -c / M)
        x1, x2, x3 = RotX(-π / 2) * [x1, x2, x3]
    elseif fdim == 3 && c > 0
        # top face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, b / M)
    elseif fdim == 3 && c < 0
        # bottom face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, -b / M)
        x1, x2, x3 = RotX(π) * [x1, x2, x3]
    else
        error(
            "invalid case for cubed_sphere_warp(::ConformalSphereWarp): $a, $b, $c",
        )
    end
    return x1 * R, x2 * R, x3 * R
end

"""
    conformal_sphere_warp(a, b, c)
A wrapper function for the cubed_sphere_warp function, when called with the
ConformalSphereWarp type
"""
conformal_sphere_warp(a, b, c) =
    cubed_sphere_warp(ConformalSphereWarp(), a, b, c)

"""
    cubed_sphere_unwarp(::ConformalSphereWarp, x1, x2, x3)

The inverse of [`cubed_sphere_warp`](@ref). This function projects
a given point `(x_1, x_2, x_3)` from the surface of a sphere onto a cube
[Rancic1996](@cite)
"""
function cubed_sphere_unwarp(::ConformalSphereWarp, x1, x2, x3)

    # Auxuliary function that flips coordinates, if needed, to prepare input
    # arguments in the correct quadrant for the `conformal_cubed_sphere_inverse_mapping`
    # function. Then, flips the output of `conformal_cubed_sphere_inverse_mapping`
    # back to original face and scales the coordinates so that result is on the cube
    function flip_unwarp_scale(x1, x2, x3)
        R = hypot(x1, x2, x3)
        flipx1, flipx2 = false, false
        if x1 < 0 # flip the point around x2 axis
            x1 = -x1
            flipx1 = true
        end
        if x2 < 0 # flip the point around x1 axis
            x2 = -x2
            flipx2 = true
        end
        a, b = conformal_cubed_sphere_inverse_mapping(x1 / R, x2 / R, x3 / R)
        if flipx1 == true
            a = -a
        end
        if flipx2 == true
            b = -b
        end

        # Rescale to desired length
        a *= R
        b *= R
        # Since we were trating coordinates on top face of the cube, the c
        # coordinate must have the top-face z value (z = R)
        c = R

        return a, b, c
    end

    fdim = argmax(abs.((x1, x2, x3)))
    if fdim == 1 && x1 < 0 # left face
        # rotate to align with top face
        x1, x2, x3 = RotY(π / 2) * RotX(-π / 2) * [x1, x2, x3]
        # call the unwarp function, with appropriate flipping and scaling
        a, b, c = flip_unwarp_scale(x1, x2, x3)
        # rotate back
        a, b, c = RotX(π / 2) * RotY(-π / 2) * [a, b, c]
    elseif fdim == 2 && x2 < 0 # front face
        # rotate to align with top face
        x1, x2, x3 = RotX(-π / 2) * [x1, x2, x3]
        # call the unwarp function, with appropriate flipping and scaling
        a, b, c = flip_unwarp_scale(x1, x2, x3)
        # rotate back
        a, b, c = RotX(π / 2) * [a, b, c]
    elseif fdim == 1 && x1 > 0 # right face
        # rotate to align with top face
        x1, x2, x3 = RotZ(-π / 2) * RotY(-π / 2) * [x1, x2, x3]
        # call the unwarp function, with appropriate flipping and scaling
        a, b, c = flip_unwarp_scale(x1, x2, x3)
        # rotate back
        a, b, c = RotY(π / 2) * RotZ(π / 2) * [a, b, c]
    elseif fdim == 2 && x2 > 0 # back face
        # rotate to align with top face
        x1, x2, x3 = RotZ(π) * RotX(π / 2) * [x1, x2, x3]
        # call the unwarp function, with appropriate flipping and scaling
        a, b, c = flip_unwarp_scale(x1, x2, x3)
        # rotate back
        a, b, c = RotX(-π / 2) * RotZ(-π) * [a, b, c]
    elseif fdim == 3 && x3 > 0 # top face
        # already on top face, no need to rotate
        a, b, c = flip_unwarp_scale(x1, x2, x3)
    elseif fdim == 3 && x3 < 0 # bottom face
        # rotate to align with top face
        x1, x2, x3 = RotX(π) * [x1, x2, x3]
        # call the unwarp function, with appropriate flipping and scaling
        a, b, c = flip_unwarp_scale(x1, x2, x3)
        # rotate back
        a, b, c = RotX(-π) * [a, b, c]
    else
        error(
            "invalid case for cubed_sphere_unwarp(::ConformalSphereWarp): $a, $b, $c",
        )
    end

    return a, b, c
end

"""
    conformal_sphere_unwarp(x1, x2, x3)

A wrapper function for the cubed_sphere_unwarp function, when called with the
    ConformalSphereWarp type
"""
conformal_sphere_unwarp(x1, x2, x3) =
    cubed_sphere_unwarp(ConformalSphereWarp(), x1, x2, x3)
