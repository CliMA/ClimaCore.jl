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
    Nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    topology = space.topology

    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(Topologies.interior_faces(topology))

        internal_surface_geometry_slab =
            slab(space.internal_surface_geometry, iface)

        arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), elem⁻), args)
        arg_slabs⁺ = map(arg -> slab(Fields.todata(arg), elem⁺), args)

        dydt_slab⁻ = slab(Fields.field_values(dydt), elem⁻)
        dydt_slab⁺ = slab(Fields.field_values(dydt), elem⁺)

        for q in 1:Nq
            sgeom⁻ = internal_surface_geometry_slab[q]

            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

            numflux⁻ = fn(
                sgeom⁻.normal,
                map(
                    slab -> slab isa DataSlab2D ? slab[i⁻, j⁻] : slab,
                    arg_slabs⁻,
                ),
                map(
                    slab -> slab isa DataSlab2D ? slab[i⁺, j⁺] : slab,
                    arg_slabs⁺,
                ),
            )

            dydt_slab⁻[i⁻, j⁻] = dydt_slab⁻[i⁻, j⁻] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
            dydt_slab⁺[i⁺, j⁺] = dydt_slab⁺[i⁺, j⁺] ⊞ (sgeom⁻.sWJ ⊠ numflux⁻)
        end
    end
end

function add_numerical_flux_internal_horizontal!(fn, dydt, args...)
    space = axes(dydt)
    horizontal_space = space.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(space.horizontal_space.quadrature_style)
    topology = space.horizontal_space.topology
    nlevels = Spaces.nlevels(space)

    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(Topologies.interior_faces(topology))
        for level in 1:nlevels
          internal_surface_geometry_slab =
              slab(horizontal_space.internal_surface_geometry, iface)

          arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), level, elem⁻), args)
          arg_slabs⁺ = map(arg -> slab(Fields.todata(arg), level, elem⁺), args)

          dydt_slab⁻ = slab(Fields.field_values(dydt), level, elem⁻)
          dydt_slab⁺ = slab(Fields.field_values(dydt), level, elem⁺)
          
          @assert dydt_slab⁻ isa DataSlab1D
          @assert dydt_slab⁺ isa DataSlab1D

          for q in 1:Nq
              sgeom⁻ = internal_surface_geometry_slab[q]

              i⁻, = Topologies.face_node_index(face⁻, Nq, q, false)
              i⁺, = Topologies.face_node_index(face⁺, Nq, q, reversed)

              numflux⁻ = fn(
                  sgeom⁻.normal,
                  map(
                      slab -> slab isa DataSlab1D ? slab[i⁻] : slab,
                      arg_slabs⁻,
                  ),
                  map(
                      slab -> slab isa DataSlab1D ? slab[i⁺] : slab,
                      arg_slabs⁺,
                  ),
              )

              dydt_slab⁻[i⁻] = dydt_slab⁻[i⁻] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
              dydt_slab⁺[i⁺] = dydt_slab⁺[i⁺] ⊞ (sgeom⁻.sWJ ⊠ numflux⁻)
          end
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
    Favg =
        RecursiveApply.rdiv(fn.fluxfn(argvals⁻...) ⊞ fn.fluxfn(argvals⁺...), 2)
    return RecursiveApply.rmap(f -> f' * normal, Favg)
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
    Favg =
        RecursiveApply.rdiv(fn.fluxfn(argvals⁻...) ⊞ fn.fluxfn(argvals⁺...), 2)
    λ = max(fn.wavespeedfn(argvals⁻...), fn.wavespeedfn(argvals⁺...))
    return RecursiveApply.rmap(f -> f' * normal, Favg) ⊞ (λ / 2) ⊠ (y⁻ ⊟ y⁺)
end


function add_numerical_flux_boundary!(fn, dydt, args...)
    space = axes(dydt)
    Nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)
    topology = space.topology

    for (iboundary, boundarytag) in enumerate(Topologies.boundaries(topology))
        for (iface, (elem⁻, face⁻)) in
            enumerate(Topologies.boundary_faces(topology, boundarytag))
            boundary_surface_geometry_slab =
                surface_geometry_slab =
                    slab(space.boundary_surface_geometries[iboundary], iface)

            arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), elem⁻), args)
            dydt_slab⁻ = slab(Fields.field_values(dydt), elem⁻)
            for q in 1:Nq
                sgeom⁻ = boundary_surface_geometry_slab[q]
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                numflux⁻ = fn(
                    sgeom⁻.normal,
                    map(
                        slab -> slab isa DataSlab2D ? slab[i⁻, j⁻] : slab,
                        arg_slabs⁻,
                    ),
                )
                dydt_slab⁻[i⁻, j⁻] =
                    dydt_slab⁻[i⁻, j⁻] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
            end
        end
    end
    return dydt
end
