using AssociatedLegendrePolynomials
# Input lat and long in degrees

function Ylm(l, m, lat, long)
    return λlm(l, m, sind(lat)) * cosd(m * long)
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

function VSH(l, m, lat, long)
    C = 1 / sqrt(2 * pi)
    E = exp((im * m) * deg2rad(long))
    W = Wlm(l, m, lat)
    V = Vlm(l, m, lat)
    return (real(C * (im * W - V) * E), real(C * (V + im * W) * E))
end
