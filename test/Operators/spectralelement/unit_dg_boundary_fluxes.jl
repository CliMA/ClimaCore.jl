using Test
using LinearAlgebra
using IntervalSets
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# `add_numerical_flux_boundary!`, the boundary-face counterpart of
# `add_numerical_flux_internal!`, on a channel domain (periodic in x, walls at
# y = ±Ly/2). The interior-face path is covered elsewhere, on boundary-free
# domains:
#
#   1. A zero boundary flux leaves the residual untouched.
#   2. A constant boundary flux c integrates to −c·(total boundary length):
#      the residual update is −sWJ·f per boundary node, and the GLL surface
#      weights sum to the boundary measure.
#   3. The residual is nonzero only at nodes on the domain boundary.
#   4. Boundary normals are outward: a flux f = ĵ·n̂ is −1 on the south wall
#      and +1 on the north wall, so the two walls contribute ±Lx.
#   5. On a box bounded on all four sides, the constant flux integrates to
#      −c·perimeter; the corner nodes accumulate one contribution from each
#      of their two boundary faces (the multi-contribution gather path on
#      the GPU).

function channel_space(
    ::Type{FT};
    Lx = FT(2π),
    Ly = FT(2),
    nelem = 4,
    Nq = 4,
    x1periodic = true,
) where {FT}
    context = ClimaComms.SingletonCommsContext()
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(zero(Lx)) .. Geometry.XPoint{FT}(Lx),
        Geometry.YPoint{FT}(-Ly / 2) .. Geometry.YPoint{FT}(Ly / 2);
        x1periodic,
        x1boundary = x1periodic ? nothing : (:west, :east),
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, nelem, nelem)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{Nq}();
        discontinuous = true,
    )
end

@testset "DG boundary numerical fluxes" begin
    TU.@test_precisions FT begin
        Lx = FT(2π)
        Ly = FT(2)
        space = channel_space(FT; Lx, Ly)
        @test !Spaces.is_continuous(space)
        coords = Fields.coordinate_field(space)
        q = ones(space)

        @testset "Zero boundary flux is a no-op [$FT]" begin
            r = zeros(space)
            zero_flux(normal, (q⁻,)) = zero(q⁻)
            Operators.add_numerical_flux_boundary!(zero_flux, r, q)
            @test maximum(abs, parent(r)) == 0
        end

        @testset "Constant flux integrates to -c * boundary length [$FT]" begin
            c = FT(3)
            r = zeros(space)
            const_flux(normal, (q⁻,)) = c
            Operators.add_numerical_flux_boundary!(const_flux, r, q)
            # Boundary = south wall + north wall (x is periodic): length 2*Lx
            @test sum(parent(r)) ≈ -c * 2 * Lx rtol = sqrt(eps(FT))
        end

        @testset "Only boundary nodes are touched [$FT]" begin
            r = zeros(space)
            const_flux(normal, (q⁻,)) = one(q⁻)
            Operators.add_numerical_flux_boundary!(const_flux, r, q)
            y = Array(parent(Fields.coordinate_field(space).y))
            interior = @. abs(abs(y) - Ly / 2) > sqrt(eps(FT))
            @test all(iszero, Array(parent(r))[interior])
            @test any(!iszero, Array(parent(r))[.!interior])
        end

        @testset "Boundary normals point outward [$FT]" begin
            r = zeros(space)
            ĵ = Geometry.UVVector(FT(0), FT(1))
            normal_flux(normal, (q⁻,)) = ĵ' * normal
            Operators.add_numerical_flux_boundary!(normal_flux, r, q)
            y = Array(parent(Fields.coordinate_field(space).y))
            south = @. abs(y + Ly / 2) <= sqrt(eps(FT))
            north = @. abs(y - Ly / 2) <= sqrt(eps(FT))
            # residual = -sWJ * f: south (n̂ = -ĵ, f = -1) gains +sWJ,
            # north (n̂ = +ĵ, f = +1) gains -sWJ; each wall has length Lx.
            @test sum(Array(parent(r))[south]) ≈ Lx rtol = sqrt(eps(FT))
            @test sum(Array(parent(r))[north]) ≈ -Lx rtol = sqrt(eps(FT))
        end

        @testset "Fully bounded box: constant flux over the perimeter [$FT]" begin
            box = channel_space(FT; Lx, Ly, x1periodic = false)
            c = FT(3)
            rb = zeros(box)
            qb = ones(box)
            const_flux(normal, (q⁻,)) = c
            Operators.add_numerical_flux_boundary!(const_flux, rb, qb)
            @test sum(parent(rb)) ≈ -c * (2 * Lx + 2 * Ly) rtol = sqrt(eps(FT))
        end

        @testset "Multi-field (NamedTuple) boundary flux [$FT]" begin
            # A flux returning a NamedTuple must scale and subtract per field,
            # like the interior loops and the GPU boundary kernel: its `a`
            # component must match the scalar-flux path, and the zero-flux `b`
            # component must stay zero.
            box = channel_space(FT; Lx, Ly, x1periodic = false)
            coords = Fields.coordinate_field(box)
            a = @. sin(coords.x) + cos(coords.y)
            b = ones(box)
            y = map((ai, bi) -> (; a = ai, b = bi), a, b)
            r = similar(y)
            fill!(parent(r), 0)
            nt_flux(normal, (y⁻,)) = (; a = y⁻.a, b = zero(y⁻.b))
            Operators.add_numerical_flux_boundary!(nt_flux, r, y)

            r_a = zeros(box)
            scalar_flux(normal, (q⁻,)) = q⁻
            Operators.add_numerical_flux_boundary!(scalar_flux, r_a, a)
            @test parent(r.a) ≈ parent(r_a) rtol = sqrt(eps(FT))
            @test all(iszero, parent(r.b))
        end

        @testset "boundary connectivity gather matches CPU path [$FT]" begin
            # Host emulation of the GPU boundary algorithm (stage
            # sWJ·fn(n̂, ·⁻) per boundary face node, gather with the
            # minus-side map), compared against the CPU boundary loop;
            # validates `Operators.dg_boundary_connectivity` — including the
            # two-contribution corner nodes of the box — without a GPU.
            # Skipped on GPU devices, where the operator testsets above
            # exercise the CUDA kernels themselves.
            if ClimaComms.device(space) isa ClimaComms.AbstractCPUDevice
                box = channel_space(FT; Lx, Ly, x1periodic = false)
                bconn = Operators.dg_boundary_connectivity(box)
                @test !isnothing(bconn)
                @test all(==(1), Array(bconn.contrib_side))
                coords_b = Fields.coordinate_field(box)
                qb = @. sin(coords_b.x) + cos(coords_b.y)
                flux(normal, (q⁻,)) =
                    q⁻ * (Geometry.UVVector(FT(1), FT(1))' * normal)
                r_ref = zeros(box)
                Operators.add_numerical_flux_boundary!(flux, r_ref, qb)

                q_data = Fields.field_values(qb)
                Nq = Quadratures.degrees_of_freedom(
                    Spaces.quadrature_style(box),
                )
                staging = Array{FT}(undef, Nq, bconn.nfaces)
                for f in 1:bconn.nfaces, qq in 1:Nq
                    elem⁻ = Int(bconn.faces[1, f])
                    face⁻ = Int(bconn.faces[2, f])
                    i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, qq, false)
                    sg = bconn.sgeom[qq, 1, f]
                    q⁻ = q_data[CartesianIndex(1, i⁻, j⁻, elem⁻)]
                    staging[qq, f] = sg.sWJ * flux(sg.normal, (q⁻,))
                end
                r_em = zeros(box)
                r_data = Fields.field_values(r_em)
                for n in 1:bconn.nbnodes
                    I = CartesianIndex(
                        1,
                        Int(bconn.node_i[n]),
                        Int(bconn.node_j[n]),
                        Int(bconn.node_elem[n]),
                    )
                    acc = r_data[I]
                    for c in
                        Int(bconn.node_offset[n]):(Int(bconn.node_offset[n + 1]) - 1)

                        acc -= staging[
                            Int(bconn.contrib_q[c]),
                            Int(bconn.contrib_face[c]),
                        ]
                    end
                    r_data[I] = acc
                end
                err = maximum(abs, parent(r_em) .- parent(r_ref))
                scale = max(maximum(abs, parent(r_ref)), one(FT))
                @test err ≤ sqrt(eps(FT)) * scale
            end
        end
    end
end
