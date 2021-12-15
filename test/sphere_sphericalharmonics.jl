# using Legendre
# # input lat and long in degrees

# function Ylm(l,m,lat,long) 
# 	return real(
# 		λlm(l,m,sind(lat)) * exp(im * m * deg2rad(long)),
# 	)
# end

using SphericalHarmonics
# input lat and long in degrees

function Ylm(l, m, lat, long)
    return real(computeYlm(deg2rad(90 - lat), deg2rad(long), lmax = l)[(l, m)])
end

function Plm(l, m, x)
    return computePlmx(x, lmax = l)[(l, m)] *
           sqrt(2 * π * factorial(l + m) / (2 * l + 1) / factorial(l - m))
end

# V and W for vector spherical harmonics
function Vlm(l, m, lat)
    return 0.5 * (
        Plm(l, m + 1, sind(lat)) -
        (l + m) * (l - m + 1) * Plm(l, m - 1, sind(lat))
    ) / sqrt(l * (l + 1))
end

function Wlm(l, m, lat)
    return 0.5 * (
        Plm(l - 1, m + 1, sind(lat)) +
        (l + m) * (l + m - 1) * Plm(l - 1, m - 1, sind(lat))
    ) / sqrt(l * (l + 1))
end

function Blm(l, m, lat, long)
    C = 1 / sqrt(2 * pi)
    E = exp((im * m) * deg2rad(long))
    W = Wlm(l, m, lat)
    V = Vlm(l, m, lat)
    return (real(C * im * W * E), real(C * V * E))
end

function Clm(l, m, lat, long)
    C = 1 / sqrt(2 * pi)
    E = exp((im * m) * deg2rad(long))
    W = Wlm(l, m, lat)
    V = Vlm(l, m, lat)
    return (real(-C * V * E), real(C * im * W * E))
end

function VSH(l, m, lat, long)
    C = 1 / sqrt(2 * pi)
    E = exp((im * m) * deg2rad(long))
    W = Wlm(l, m, lat)
    V = Vlm(l, m, lat)
    return (real(C * (im * W - V) * E), real(C * (V + im * W) * E))
end
