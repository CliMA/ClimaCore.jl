"""
    TerrainAdaption

Mechanism for adapting the domain to a surface field.
"""
abstract type TerrainAdaption end


"""
    adapt(z_ref, z_s, z_t, a::TerrainAdaption)

Construct the new z coordinate from the reference `z_ref`, using terrain
adaptation `a`.
- `z_s` is the bottom (surface) of the domain
- `z_t` is the top of the domain
"""
function adapt end

"""
    LinearAdaption()

Locate the levels by linear interpolation between the bottom and top of the
domain.

See Gal-Chen and Somerville (1975)
"""
struct LinearAdaption <: TerrainAdaption end
adapt(z_ref, z_s, z_top, a::LinearAdaption) = z_ref + (1 - z_ref / z_top) * z_s

"""
    SLEVE(z_fold)

Schär et al. (2002) Coordinate
"""
struct SLEVE{T} <: TerrainAdaption
    z_fold::T
end
function adapt(z_ref, z_s, z_top, a::SLEVE)
    s = a.z_fold / z_top
    z_ref + sinh((1 - z_ref / z_top) / s) / sinh(1 / s) * z_s
end
