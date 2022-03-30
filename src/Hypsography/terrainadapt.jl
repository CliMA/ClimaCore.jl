"""
    TerrainAdaption

Mechanism for adapting the domain to a reference surface field.
"""
abstract type TerrainAdaption end

"""
    adapt(::TerrainAdaption, z_ref, z_s, z_t)

Construct the new vertical z coordinate from the reference `z_ref`, using surface adaptation `a`.
`z_s` is the terrian surface coordinate of the vertical domain and `z_t` is the top of the vertical domain.
"""
function adapt end

"""
    LinearAdaption()

Locate the levels by linear interpolation between the bottom and top of the domain.

Ref: Gal-Chen and Somerville (1975)
"""
struct LinearAdaption <: TerrainAdaption end

function adapt(::LinearAdaption, z_ref, z_s, z_top)
    z_ref + (1 - z_ref / z_top) * z_s
end
