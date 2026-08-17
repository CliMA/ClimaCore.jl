# Physics shared by the two dry rising bubble cases in this directory,
# `bubble_3d_invariant_rhoe.jl` and `bubble_3d_flux_form_rhoe.jl`. They solve
# the same problem with the same prognostic variables and differ only in the
# momentum formulation, so they have to agree on the constants and on the
# geopotential for the comparison between them to mean anything.
#
# Reference: Section 5a of
# https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml
import Adapt
using DocStringExtensions

"""
    PhysicalParameters{FT}

Thermodynamic constants for the dry rising bubble.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PhysicalParameters{FT}
    "Mean sea level pressure"
    MSLP::FT = FT(1e5)
    "Gravitational acceleration"
    grav::FT = FT(9.8)
    "R dry (gas constant / mol mass dry air)"
    R_d::FT = FT(287.058)
    "Heat capacity ratio"
    γ::FT = FT(1.4)
    "Heat capacity at constant pressure"
    C_p::FT = FT(R_d * γ / (γ - 1))
    "Heat capacity at constant volume"
    C_v::FT = FT(R_d / (γ - 1))
    "Triple point temperature"
    T_0::FT = FT(273.16)
end
Adapt.@adapt_structure PhysicalParameters

geopotential(z, grav) = grav * z
