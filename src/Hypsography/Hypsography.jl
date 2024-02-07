module Hypsography

import ClimaComms, Adapt

import ..slab, ..column
import ..Geometry,
    ..DataLayouts,
    ..Domains,
    ..Topologies,
    ..Grids,
    ..Spaces,
    ..Fields,
    ..Operators
import ..Spaces: ExtrudedFiniteDifferenceSpace
import ClimaCore.Utilities: half

import ..Grids:
    _ExtrudedFiniteDifferenceGrid,
    ExtrudedFiniteDifferenceGrid,
    HypsographyAdaption,
    Flat

using StaticArrays, LinearAlgebra


"""
    ref_z_to_physical_z(adaption::HypsographyAdaption, z_ref::ZPoint, z_surface::ZPoint, z_top::ZPoint) :: ZPoint

Convert reference `z`s to physical `z`s as prescribed by the given adaption.

This function has to be the inverse of `physical_z_to_ref_z`.
"""
function ref_z_to_physical_z(
    adaption::HypsographyAdaption,
    z_ref::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
) end

"""
    physical_z_to_ref_z(adaption::HypsographyAdaption, z_ref::ZPoint, z_surface::ZPoint, z_top::ZPoint) :: ZPoint

Convert physical `z`s to reference `z`s as prescribed by the given adaption.

This function has to be the inverse of `ref_z_to_physical_z`.
"""
function physical_z_to_ref_z(
    adaption::HypsographyAdaption,
    z_phys::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
) end

# Flat, z_ref = z_physical

function ref_z_to_physical_z(
    ::Flat,
    z_ref::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
)
    return z_ref
end

function physical_z_to_ref_z(
    ::Flat,
    z_physical::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
)
    return z_physical
end

"""
    LinearAdaption(surface::Field)

Locate the levels by linear interpolation between the surface field and the top
of the domain, using the method of [GalChen1975](@cite).
"""
struct LinearAdaption{F <: Fields.Field} <: HypsographyAdaption
    surface::F
    function LinearAdaption(surface::Fields.Field)
        if eltype(surface) <: Real
            @warn "`LinearAdaptation`: `surface` argument scalar field has been deprecated. Use a field `ZPoint`s."
            surface = Geometry.ZPoint.(surface)
        end
        new{typeof(surface)}(surface)
    end
end

# This method is invoked by the ExtrudedFiniteDifferenceGrid constructor
function ref_z_to_physical_z(
    ::LinearAdaption,
    z_ref,
    z_surface,
    z_top,
)
    Geometry.ZPoint.(z_ref.z .+ (1 .- z_ref.z ./ z_top.z) .* z_surface.z)
end

# This method is used for remapping
function physical_z_to_ref_z(
    ::LinearAdaption,
    z_physical::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
)
    Geometry.ZPoint.((z_physical.z .- z_surface.z) ./ (1 .- z_surface.z ./ z_top.z))
end

"""
    SLEVEAdaption(surface::Field, ηₕ::FT, s::FT)

Locate vertical levels using an exponential function between the surface field and the top
of the domain, using the method of [Schar2002](@cite). This method is modified
such no warping is applied above some user defined parameter 0 ≤ ηₕ < 1.0, where the lower and upper
bounds represent the domain bottom and top respectively. `s` governs the decay rate.
If the decay-scale is poorly specified (i.e., `s * zₜ` is lower than the maximum
surface elevation), a warning is thrown and `s` is adjusted such that it `szₜ > maximum(z_surface)`.
"""
struct SLEVEAdaption{F <: Fields.Field, FT <: Real} <: HypsographyAdaption
    surface::F
    ηₕ::FT
    s::FT
    function SLEVEAdaption(
        surface::Fields.Field,
        ηₕ::FT,
        s::FT,
    ) where {FT <: Real}
        @assert 0 <= ηₕ <= 1
        @assert s >= 0
        if eltype(surface) <: Real
            @warn "`SLEVEAdaption`: `surface` argument scalar field has been deprecated. Use a field `ZPoint`s."
            surface = Geometry.ZPoint.(surface)
        end
        new{typeof(surface), FT}(surface, ηₕ, s)
    end
end


function ref_z_to_physical_z(
    adaption::SLEVEAdaption,
    z_ref,
    z_surface,
    z_top,
)
    (; ηₕ, s) = adaption
    if s .* z_top.z <= maximum(z_surface.z)
        error("Decay scale (s*z_top) must be higher than max surface elevation")
    end

    η = z_ref.z ./ z_top.z
    return ifelse.(η .<= ηₕ, 
            Geometry.ZPoint.(
            η .* z_top.z .+
            z_surface.z .* (sinh.((ηₕ .- η) ./ s ./ ηₕ)) ./ (sinh.(1 ./ s))),
            Geometry.ZPoint.(η .* z_top.z))

end

function physical_z_to_ref_z(
    adaption::SLEVEAdaption,
    z_physical::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
)
    error("This method is not implemented")
end

# can redefine this constructor for e.g. multi-arg SLEVE
function _ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Grids.AbstractGrid,
    vertical_grid::Grids.FiniteDifferenceGrid,
    adaption::HypsographyAdaption,
    global_geometry::Geometry.AbstractGlobalGeometry,
)
    @assert Spaces.grid(axes(adaption.surface)) == horizontal_grid
    z_surface = Fields.field_values(adaption.surface)

    face_z_ref =
        Grids.local_geometry_data(vertical_grid, Grids.CellFace()).coordinates
    
    vertical_domain = Topologies.domain(vertical_grid)
    z_top = vertical_domain.coord_max

    face_z =
        ref_z_to_physical_z(adaption, face_z_ref, z_surface, z_top)

    return _ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        adaption,
        global_geometry,
        face_z,
    )
end

# generic 5-arg hypsography constructor, uses computed face_z points
function _ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Grids.AbstractGrid,
    vertical_grid::Grids.FiniteDifferenceGrid,
    adaption::HypsographyAdaption,
    global_geometry::Geometry.AbstractGlobalGeometry,
    face_z::DataLayouts.AbstractData{Geometry.ZPoint{FT}},
) where {FT}
    # construct the "flat" grid
    # avoid cached constructor so that it gets cleaned up automatically
    flat_grid = _ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        Flat(),
        global_geometry,
    )
    center_flat_space = Spaces.space(flat_grid, Grids.CellCenter())
    face_flat_space = Spaces.space(flat_grid, Grids.CellFace())

    # compute the "z-only local geometry" based on face z coords
    ArrayType = ClimaComms.array_type(horizontal_grid.topology)
    # currently only works on Arrays
    (center_z_local_geometry, face_z_local_geometry) = Grids.fd_geometry_data(
        Adapt.adapt(Array, face_z);
        periodic = Topologies.isperiodic(vertical_grid.topology),
    )

    center_z_local_geometry = Adapt.adapt(ArrayType, center_z_local_geometry)
    face_z_local_geometry = Adapt.adapt(ArrayType, face_z_local_geometry)

    # compute ∇Z at face and centers
    grad = Operators.Gradient()

    center_∇Z_field =
        grad.(
            Fields.Field(
                center_z_local_geometry,
                center_flat_space,
            ).coordinates.z
        )
    Spaces.weighted_dss!(center_∇Z_field)

    face_∇Z_field =
        grad.(
            Fields.Field(face_z_local_geometry, face_flat_space).coordinates.z
        )
    Spaces.weighted_dss!(face_∇Z_field)

    # construct full local geometry
    center_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            center_z_local_geometry,
            Ref(global_geometry),
            Ref(Geometry.WVector(1)) .*
            adjoint.(Fields.field_values(center_∇Z_field)),
        )
    face_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            face_z_local_geometry,
            Ref(global_geometry),
            Ref(Geometry.WVector(1)) .*
            adjoint.(Fields.field_values(face_∇Z_field)),
        )

    return ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        adaption,
        global_geometry,
        center_local_geometry,
        face_local_geometry,
    )
end

"""
    diffuse_surface_elevation!(f::Field; κ::T, iter::Int, dt::T)

Option for 2nd order diffusive smoothing of generated terrain.
Mutate (smooth) a given elevation profile `f` before assigning the surface
elevation to the `HypsographyAdaption` type. A spectral second-order diffusion
operator is applied with forward-Euler updates to generate
profiles for each new iteration. Steps to generate smoothed terrain (
represented as a ClimaCore Field) are as follows:
- Compute discrete elevation profile f
- Compute diffuse_surface_elevation!(f, κ, iter). f is mutated.
- Define `Hypsography.LinearAdaption(f)`
- Define `ExtrudedFiniteDifferenceSpace` with new surface elevation.
Default diffusion parameters are appropriate for spherical arrangements.
For `zmax-zsfc` == 𝒪(10^4), κ == 𝒪(10^8), dt == 𝒪(10⁻¹).
"""
function diffuse_surface_elevation!(
    f::Fields.Field;
    κ::T = 1e8,
    maxiter::Int = 100,
    dt::T = 1e-1,
) where {T}
    if eltype(f) <: Real
        f_z = f
    elseif eltype(f) <: Geometry.ZPoint
        f_z = f.z
    end
    # Define required ops
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    # Create dss buffer
    ghost_buffer = (bf = Spaces.create_dss_buffer(f_z),)
    # Apply smoothing
    for iter in 1:maxiter
        # Euler steps
        χf = @. wdiv(grad(f_z))
        Spaces.weighted_dss!(χf, ghost_buffer.bf)
        @. f_z += κ * dt * χf
    end
    # Return mutated surface elevation profile
    return f
end

end
