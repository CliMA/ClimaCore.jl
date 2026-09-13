using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Operators,
    Quadratures,
    Geometry
using LinearAlgebra: norm_sqr

# Reference implementation on the host parent array, (Nv, Ni, Nj, Nf, Nh):
# θ = Bᵀ (WJ .* f) B ./ (Bᵀ WJ B) in each slab, prolonged back as B θ Bᵀ, with
# B the interpolation matrix from the coarse to the fine quadrature points and
# a 1×1 identity along a horizontal dimension that has a single point.
function reference_lumped(f, coarse_quad)
    space = axes(f)
    FT = Spaces.undertype(space)
    pf = Array(parent(f))
    pw = Array(parent(Spaces.local_geometry_data(space).WJ))
    B = Array(
        Quadratures.interpolation_matrix(
            FT,
            Spaces.quadrature_style(space),
            coarse_quad,
        ),
    )
    Bi = size(pf, 2) == 1 ? ones(FT, 1, 1) : B
    Bj = size(pf, 3) == 1 ? ones(FT, 1, 1) : B
    for h in axes(pf, 5), c in axes(pf, 4), v in axes(pf, 1)
        Fv = pf[v, :, :, c, h]
        Wv = pw[v, :, :, 1, h]
        θ = (Bi' * (Wv .* Fv) * Bj) ./ (Bi' * Wv * Bj)
        pf[v, :, :, c, h] .= Bi * θ * Bj'
    end
    out = similar(f)
    copyto!(parent(out), pf)
    return out
end

max_abs_diff(a, b) = maximum(abs, Array(parent(a)) .- Array(parent(b)))
max_abs(a) = maximum(abs, Array(parent(a)))

@testset "LumpedRestriction" begin
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    grad = Operators.Gradient()
    lumped = Operators.LumpedRestriction()
    @test lumped.quadrature_style == Quadratures.GLL{2}()

    for FT in (Float64, Float32)
        tol = FT == Float64 ? 1e-12 : 1e-5
        Nq = 4
        domain = Domains.RectangleDomain(
            Domains.IntervalDomain(
                Geometry.XPoint(FT(0)),
                Geometry.XPoint(FT(4));
                periodic = true,
            ),
            Domains.IntervalDomain(
                Geometry.YPoint(FT(0)),
                Geometry.YPoint(FT(3));
                periodic = true,
            ),
        )
        mesh = Meshes.RectilinearMesh(domain, 4, 3)
        topology = Topologies.Topology2D(context, mesh)
        space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{Nq}())
        coords = Fields.coordinate_field(space)
        f = @. sin(3 * coords.x) * cos(5 * coords.y) + FT(0.3) * coords.x
        WJ = Fields.Field(Spaces.local_geometry_data(space).WJ, space)

        @testset "plane, $FT" begin
            g = lumped.(f)
            @test axes(g) === space
            @test eltype(g) == FT
            @test max_abs_diff(g, reference_lumped(f, Quadratures.GLL{2}())) <
                  tol * max_abs(f)

            # A constant is reproduced to round-off (the rows of B sum to 1).
            @test max_abs_diff(lumped.(ones(space)), ones(space)) <= 4 * eps(FT)

            # The WJ-weighted integral over every element is conserved.
            elem_int(x) = vec(
                sum(Array(parent(x)) .* Array(parent(WJ)); dims = (1, 2, 3, 4)),
            )
            @test maximum(abs, elem_int(g) .- elem_int(f)) <
                  tol * maximum(abs, elem_int(f))

            # The result is bilinear inside every element: its interior values
            # are the bilinear interpolant of its corner values.
            pg = Array(parent(g))
            ξ, _ = Quadratures.quadrature_points(FT, Quadratures.GLL{Nq}())
            B = hcat((1 .- ξ) ./ 2, (1 .+ ξ) ./ 2)
            for h in axes(pg, 5)
                corners = pg[1, [1, Nq], [1, Nq], 1, h]
                @test maximum(abs, pg[1, :, :, 1, h] .- B * corners * B') <
                      tol * max_abs(f)
            end

            # The operator is linear and composes with pointwise operations.
            w = @. 2 * lumped(f * 3) + lumped(lumped(f))
            w_ref = 6 .* g .+ lumped.(g)
            @test max_abs_diff(w, w_ref) < tol * max_abs(w_ref)
        end

        @testset "fused with Gradient, $FT" begin
            # The register-resident result of Gradient is published through a
            # buffer before the coarse points read it.
            inv_nodal = @. norm_sqr(grad(f))
            inv_lumped = @. lumped(norm_sqr(grad(f)))
            @test max_abs_diff(
                inv_lumped,
                reference_lumped(inv_nodal, Quadratures.GLL{2}()),
            ) < tol * max_abs(inv_nodal)

            # Vector-valued arguments are filtered componentwise.
            ∇f = @. grad(f)
            ∇f_lumped = @. lumped(grad(f))
            @test eltype(∇f_lumped) == eltype(∇f)
            @test max_abs_diff(
                ∇f_lumped,
                reference_lumped(∇f, Quadratures.GLL{2}()),
            ) < tol * max_abs(∇f)
        end

        @testset "other coarse quadratures, $FT" begin
            # GL{1} gives the WJ-weighted element mean at every point.
            mean_op = Operators.LumpedRestriction(Quadratures.GL{1}())
            m = mean_op.(f)
            pm = Array(parent(m))
            pf = Array(parent(f))
            pw = Array(parent(WJ))
            for h in axes(pm, 5)
                elem_mean =
                    sum(pf[:, :, :, :, h] .* pw[:, :, :, :, h]) /
                    sum(pw[:, :, :, :, h])
                @test maximum(abs, pm[:, :, :, :, h] .- elem_mean) <
                      tol * max_abs(f)
            end
            @test max_abs_diff(m, reference_lumped(f, Quadratures.GL{1}())) <
                  tol * max_abs(f)

            g3 = Operators.LumpedRestriction(Quadratures.GLL{3}()).(f)
            @test max_abs_diff(g3, reference_lumped(f, Quadratures.GLL{3}())) <
                  tol * max_abs(f)

            # The space's own quadrature gives the identity; more points than
            # the space has is an error.
            @test max_abs_diff(
                Operators.LumpedRestriction(Quadratures.GLL{Nq}()).(f),
                f,
            ) == 0
            @test_throws ArgumentError Operators.LumpedRestriction(
                Quadratures.GLL{Nq + 1}(),
            ).(
                f,
            )
        end

        @testset "extruded space, $FT" begin
            vdomain = Domains.IntervalDomain(
                Geometry.ZPoint(FT(0)),
                Geometry.ZPoint(FT(1));
                boundary_names = (:bottom, :top),
            )
            vtopology = Topologies.IntervalTopology(
                context,
                Meshes.IntervalMesh(vdomain, nelems = 5),
            )
            vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
            hv_space = Spaces.ExtrudedFiniteDifferenceSpace(space, vspace)
            c3 = Fields.coordinate_field(hv_space)
            f3 = @. sin(3 * c3.x) * cos(5 * c3.y) * (1 + c3.z)
            inv3 = @. norm_sqr(grad(f3))
            inv3_lumped = @. lumped(norm_sqr(grad(f3)))
            @test axes(inv3_lumped) === hv_space
            @test max_abs_diff(
                inv3_lumped,
                reference_lumped(inv3, Quadratures.GLL{2}()),
            ) < tol * max_abs(inv3)

            # A column has no horizontal element to filter within: no-op.
            zc = Fields.coordinate_field(vspace).z
            fc = @. sin(zc)
            @test max_abs_diff(lumped.(fc), fc) == 0
            @test max_abs(@. lumped(norm_sqr(grad(fc)))) == 0
        end

        @testset "1D horizontal space, $FT" begin
            domain1 = Domains.IntervalDomain(
                Geometry.XPoint(FT(0)),
                Geometry.XPoint(FT(4));
                periodic = true,
            )
            space1 = Spaces.SpectralElementSpace1D(
                Topologies.IntervalTopology(
                    context,
                    Meshes.IntervalMesh(domain1, nelems = 4),
                ),
                Quadratures.GLL{Nq}(),
            )
            x1 = Fields.coordinate_field(space1).x
            f1 = @. sin(3 * x1) + x1
            @test max_abs_diff(
                lumped.(f1),
                reference_lumped(f1, Quadratures.GLL{2}()),
            ) < tol * max_abs(f1)
            inv1 = @. norm_sqr(grad(f1))
            @test max_abs_diff(
                (@. lumped(norm_sqr(grad(f1)))),
                reference_lumped(inv1, Quadratures.GLL{2}()),
            ) < tol * max_abs(inv1)
        end
    end

    @testset "element-boundary bias of |∇f|² on the sphere" begin
        # sin(kλ) cos(kφ) with two elements per wavelength on a cubed sphere with
        # 12 elements per panel edge and GLL{4}: the nodal invariant is far too
        # large at element edges and corners, and its lumped restriction is not.
        FT = Float64
        Nq = 4
        mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(FT(6.371e6)), 12)
        topology =
            Topologies.Topology2D(context, mesh, Topologies.spacefillingcurve(mesh))
        space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{Nq}())
        coords = Fields.coordinate_field(space)
        k = 24
        f = @. sin(k * deg2rad(coords.long)) * cos(k * deg2rad(coords.lat))
        inv_nodal = @. norm_sqr(grad(f))
        inv_lumped = @. lumped(norm_sqr(grad(f)))
        pw = Array(parent(Spaces.local_geometry_data(space).WJ))
        band = abs.(Array(parent(coords.lat))) .<= 60 # keep polar panel corners out
        is_boundary(i) = i == 1 || i == Nq
        n_boundary = [
            is_boundary(i) + is_boundary(j) for v in 1:1, i in 1:Nq, j in 1:Nq,
            c in 1:1, h in 1:size(pw, 5)
        ]
        function class_mean(inv, class)
            pinv = Array(parent(inv))
            mask = (n_boundary .== class) .& band
            return sum(pinv .* pw .* mask) / sum(pw .* mask)
        end
        ratios(inv) = (
            class_mean(inv, 1) / class_mean(inv, 0),
            class_mean(inv, 2) / class_mean(inv, 0),
        )
        (edge_nodal, corner_nodal) = ratios(inv_nodal)
        (edge_lumped, corner_lumped) = ratios(inv_lumped)
        @test edge_nodal > 1.5 && corner_nodal > 3
        @test 0.95 < edge_lumped < 1.05
        @test 0.9 < corner_lumped < 1.1
        # The lumped invariant conserves each element's integral of the nodal one.
        total(inv) = sum(Array(parent(inv)) .* pw)
        @test total(inv_lumped) ≈ total(inv_nodal) rtol = 1e-10
    end
end
