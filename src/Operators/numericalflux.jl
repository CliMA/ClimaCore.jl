import .DataLayouts: slab_index
"""
    add_numerical_flux_internal!(fn, dydt, args...)

Add the numerical flux at the internal faces of the spectral space mesh.

The numerical flux is determined by evaluating

    fn(normal, argvals⁻, argvals⁺)

where:
 - `normal` is the unit normal vector, pointing from the "minus" side to the "plus" side
 - `argvals⁻` is the tuple of values of `args` on the "minus" side of the face
 - `argvals⁺` is the tuple of values of `args` on the "plus" side of the face
and should return the net flux from the "minus" side to the "plus" side.

For consistency, it should satisfy the property that

    fn(normal, argvals⁻, argvals⁺) == -fn(-normal, argvals⁺, argvals⁻)


See also:
- [`CentralNumericalFlux`](@ref)
- [`RusanovNumericalFlux`](@ref)
"""
function add_numerical_flux_internal!(fn, dydt, args...)
    space = axes(dydt)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    internal_surface_geometry = Spaces.grid(space).internal_surface_geometry
    dydt_bc = Base.broadcastable(dydt)
    args_bc =
        map(arg -> arg isa Fields.Field ? Base.broadcastable(arg) : arg, args)

    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(Topologies.interior_faces(topology))

        internal_surface_geometry_slab = slab(internal_surface_geometry, iface)

        arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), elem⁻), args_bc)
        arg_slabs⁺ = map(arg -> slab(Fields.todata(arg), elem⁺), args_bc)

        dydt_slab⁻ = slab(Fields.field_values(dydt_bc), elem⁻)
        dydt_slab⁺ = slab(Fields.field_values(dydt_bc), elem⁺)

        for q in 1:Nq
            sgeom⁻ = internal_surface_geometry_slab[slab_index(q)]

            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

            argvals⁻ = map(
                slab -> slab isa DataSlab2D ? slab[slab_index(i⁻, j⁻)] : slab,
                arg_slabs⁻,
            )
            argvals⁺ = map(
                slab -> slab isa DataSlab2D ? slab[slab_index(i⁺, j⁺)] : slab,
                arg_slabs⁺,
            )
            numflux⁻ =
                add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻, argvals⁺))

            dydt_slab⁻[slab_index(i⁻, j⁻)] =
                dydt_slab⁻[slab_index(i⁻, j⁻)] - (sgeom⁻.sWJ * numflux⁻)
            dydt_slab⁺[slab_index(i⁺, j⁺)] =
                dydt_slab⁺[slab_index(i⁺, j⁺)] + (sgeom⁻.sWJ * numflux⁻)
        end
    end
end

"""
    CentralNumericalFlux(fluxfn)

Evaluates the central numerical flux using `fluxfn`.
"""
struct CentralNumericalFlux{F}
    fluxfn::F
end

function (fn::CentralNumericalFlux)(normal, argvals⁻, argvals⁺)
    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    return ((F⁻ + F⁺) / 2)' * normal
end

"""
    RusanovNumericalFlux(fluxfn, wavespeedfn)

Evaluates the Rusanov numerical flux using `fluxfn` with wavespeed `wavespeedfn`
"""
struct RusanovNumericalFlux{F, W}
    fluxfn::F
    wavespeedfn::W
end

function (fn::RusanovNumericalFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    λ = max(fn.wavespeedfn(argvals⁻...), fn.wavespeedfn(argvals⁺...))
    return ((F⁻ + F⁺) / 2)' * normal + (λ / 2) * (y⁻ - y⁺)
end


function add_numerical_flux_boundary!(fn, dydt, args...)
    space = axes(dydt)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    boundary_surface_geometries = Spaces.grid(space).boundary_surface_geometries
    dydt_bc = Base.broadcastable(dydt)
    args_bc =
        map(arg -> arg isa Fields.Field ? Base.broadcastable(arg) : arg, args)

    for (iboundary, boundarytag) in
        enumerate(Topologies.boundary_tags(topology))
        for (iface, (elem⁻, face⁻)) in
            enumerate(Topologies.boundary_faces(topology, boundarytag))
            boundary_surface_geometry_slab =
                surface_geometry_slab =
                    slab(boundary_surface_geometries[iboundary], iface)

            arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), elem⁻), args_bc)
            dydt_slab⁻ = slab(Fields.field_values(dydt_bc), elem⁻)
            for q in 1:Nq
                sgeom⁻ = boundary_surface_geometry_slab[slab_index(q)]
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                argvals⁻ = map(
                    slab ->
                        slab isa DataSlab2D ? slab[slab_index(i⁻, j⁻)] : slab,
                    arg_slabs⁻,
                )
                numflux⁻ = add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻))
                dydt_slab⁻[slab_index(i⁻, j⁻)] =
                    dydt_slab⁻[slab_index(i⁻, j⁻)] - (sgeom⁻.sWJ * numflux⁻)
            end
        end
    end
    return dydt
end
