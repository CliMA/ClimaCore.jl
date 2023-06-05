function interpolate_slab(
    field::Fields.Field,
    slabidx::Fields.SlabIndex,
    (I1,)::Tuple{<:AbstractArray},
)
    space = axes(field)
    x = zero(eltype(field))
    QS = Spaces.quadrature_style(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(QS)

    for i in 1:Nq
        ij = CartesianIndex((i,))
        x += I1[i] * Operators.get_node(space, field, ij, slabidx)
    end
    return x
end

function interpolate_slab(
    field::Fields.Field,
    slabidx::Fields.SlabIndex,
    (I1, I2)::Tuple{<:AbstractArray, <:AbstractArray},
)
    space = axes(field)
    x = zero(eltype(field))
    QS = Spaces.quadrature_style(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(QS)

    for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        x += I1[i] * I2[j] * Operators.get_node(space, field, ij, slabidx)
    end
    return x
end


function interpolate_array(
    field::Fields.ExtrudedFiniteDifferenceField,
    xpts,
    ypts,
    zpts,
)
    space = axes(field)
    @assert ClimaComms.context(space) isa ClimaComms.SingletonCommsContext

    horz_topology = Spaces.topology(space)
    horz_mesh = horz_topology.mesh

    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh


    T = eltype(field)
    array = zeros(T, length(xpts), length(ypts), length(zpts))

    FT = Spaces.undertype(space)
    for (ix, x) in enumerate(xpts)
        for (iy, y) in enumerate(ypts)

            hcoord = Geometry.product_coordinates(x, y)
            helem = Meshes.containing_element(horz_mesh, hcoord)
            ξ1, ξ2 = Meshes.reference_coordinates(horz_mesh, helem, hcoord)
            quad = Spaces.quadrature_style(space)
            quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
            WI1 = Spaces.Quadratures.interpolation_matrix(
                SVector(ξ1),
                quad_points,
            )
            WI2 = Spaces.Quadratures.interpolation_matrix(
                SVector(ξ2),
                quad_points,
            )
            gidx = horz_topology.orderindex[helem]
            h = gidx
            Nz = Spaces.nlevels(space)

            for (iz, z) in enumerate(zpts)
                velem = Meshes.containing_element(vert_mesh, z)
                ξ3, = Meshes.reference_coordinates(vert_mesh, velem, z)
                if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
                    v_lo = velem - half
                    v_hi = velem + half
                elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace
                    # TODO: handle boundary
                    if ξ3 < 0
                        if Topologies.isperiodic(
                            Spaces.vertical_topology(space),
                        )
                            v_lo = mod1(velem - 1, Nz)
                        else
                            v_lo = max(velem - 1, 1)
                        end
                        v_hi = velem
                        ξ3 = ξ3 + 1
                    else
                        v_lo = velem
                        if Topologies.isperiodic(
                            Spaces.vertical_topology(space),
                        )
                            v_hi = mod1(velem + 1, Nz)
                        else
                            v_hi = min(velem + 1, Nz)
                        end
                        ξ3 = ξ3 - 1
                    end
                end
                f_lo = interpolate_slab(
                    field,
                    Fields.SlabIndex(v_lo, h),
                    (WI1, WI2),
                )
                f_hi = interpolate_slab(
                    field,
                    Fields.SlabIndex(v_hi, h),
                    (WI1, WI2),
                )
                array[ix, iy, iz] = ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
            end
        end
    end
    return array
end
