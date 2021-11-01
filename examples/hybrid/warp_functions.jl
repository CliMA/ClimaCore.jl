# Given a set of quadrature points (x_in, y_in, z_in) 
# Compute the warped topography coordinates, and output
# the new (x,y,z) coordinates. In the examples shown below, 
# the topography warp linearly relaxes to the top of the domain,
# other choices for relaxation functions may be applied. 

function warp_schar(
    x_in,
    y_in,
    z_in;
    Lx = 500.0,
    Ly = 0.0,
    Lz = 1000.0,
    a = 250.0, ## Half-width parameter [m]
    h₀::FT = 100, ## Peak height [m]
    λ::FT = 100, ## Wavelength
)
    FT = eltype(x_in)
    r = sqrt(x_in^2 + y_in^2)
    h_star = abs(x_in) <= a ? h₀ * (cospi((x_in) / 2a))^2 : FT(0)
    h = h_star * (cospi(x_in / λ))^2
    x, y, z = x_in, y_in, z_in + h * (Lz - z_in) / Lz
    return x, y, z
end

# General function for Agnesi peak
# Modify for 2D 
function warp_agnesi_peak(
    x_in,
    y_in,
    z_in;
    Lx = 500.0,
    Ly = 0.0,
    Lz = 1000.0,
    a = 1 / 2,
)
    FT = eltype(x_in)
    h = 8 * a^3 / (x_in^2 + 4 * a^2)
    x, y, z = x_in, y_in, z_in + h * (Lz - z_in) / Lz
    return x, y, z
end
