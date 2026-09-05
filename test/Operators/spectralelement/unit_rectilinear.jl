using Test
using StaticArrays
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Geometry,
    Fields,
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Operators,
    Quadratures
using LinearAlgebra, IntervalSets

@testset "Spectral element operators on a rectilinear mesh" begin
    FT = Float64
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
        Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
        x1periodic = true,
        x2periodic = true,
    )

    Nq = 5
    quad = Quadratures.GLL{Nq}()
    device = ClimaComms.CPUSingleThreaded()
    mesh = Meshes.RectilinearMesh(domain, 17, 16)
    topology =
        Topologies.Topology2D(ClimaComms.SingletonCommsContext(device), mesh)
    space = Spaces.SpectralElementSpace2D(topology, quad)
    coords = Fields.coordinate_field(space)

    @testset "interpolate / restrict" begin

        INq = 9
        Iquad = Quadratures.GLL{INq}()
        Ispace = Spaces.SpectralElementSpace2D(topology, Iquad)

        I = Operators.Interpolate(Ispace)
        R = Operators.Restrict(space)

        f = sin.(coords.x .+ 2 .* coords.y)

        interpolated_field = I.(f)
        Spaces.weighted_dss!(interpolated_field)

        @test Spaces.quadrature_style(axes(interpolated_field)) == Iquad
        @test Spaces.topology(axes(interpolated_field)) == topology

        restrict_field = R.(f)
        Spaces.weighted_dss!(restrict_field)

        @test Spaces.quadrature_style(axes(restrict_field)) == quad
        @test Spaces.topology(axes(restrict_field)) == topology

        interp_restrict_field = R.(I.(f))
        Spaces.weighted_dss!(interp_restrict_field)

        @test Spaces.quadrature_style(axes(interp_restrict_field)) == quad
        @test Spaces.topology(axes(interp_restrict_field)) == topology

        @test norm(interp_restrict_field .- f) ≤ 3.0e-4

        interp_restrict_nested = R.(I.(f) .+ interpolated_field)
        Spaces.weighted_dss!(interp_restrict_nested)

        @test norm(interp_restrict_nested .- 2 .* f) ≤ 3.0e-4

        # Tensor operator broadcasts as arguments of pointwise broadcasts,
        # whose size and scope queries drop the tensor operator nodes.
        interp_sum = @. I(f) + interpolated_field
        Spaces.weighted_dss!(interp_sum)

        @test norm(interp_sum .- 2 .* interpolated_field) ≤ 3.0e-4

        restrict_sum = @. R(I(f)) + f
        Spaces.weighted_dss!(restrict_sum)

        @test norm(restrict_sum .- 2 .* f) ≤ 3.0e-4

        # Spectral operators nested inside tensor operators, with the results
        # used as arguments of pointwise broadcasts.
        div = Operators.Divergence()
        grad = Operators.Gradient()

        laplacian_field = @. div(grad(f)) + f
        Spaces.weighted_dss!(laplacian_field)

        spectral_nested = @. R(I(div(grad(f)) + f) + interpolated_field) + f
        Spaces.weighted_dss!(spectral_nested)

        @test norm(spectral_nested .- (laplacian_field .+ 2 .* f)) ≤ 3.0e-4
    end

    @testset "gradient" begin

        f = sin.(coords.x .+ 2 .* coords.y)

        grad = Operators.Gradient()
        gradf = grad.(f)
        Spaces.weighted_dss!(gradf)

        @test gradf ≈
              Geometry.Covariant12Vector.(
            Geometry.UVVector.(
                cos.(coords.x .+ 2 .* coords.y),
                2 .* cos.(coords.x .+ 2 .* coords.y),
            ),
        ) rtol = 1e-2

        fv =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(coords.x .+ 2 .* coords.y),
            )
        gradfv = Geometry.transform.(Ref(Geometry.UVAxis()), grad.(fv))
        Spaces.weighted_dss!(gradfv)
        @test eltype(gradfv) <: Geometry.Tensor{2}
    end


    @testset "weak gradient" begin
        f = sin.(coords.x .+ 2 .* coords.y)

        wgrad = Operators.Gradient{Operators.WeakForm}()
        gradf = wgrad.(f)
        Spaces.weighted_dss!(gradf)

        @test Geometry.UVVector.(gradf) ≈
              Geometry.UVVector.(
            cos.(coords.x .+ 2 .* coords.y),
            2 .* cos.(coords.x .+ 2 .* coords.y),
        ) rtol = 1e-2
    end

    @testset "curl" begin
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 4 .* coords.y),
            )

        curl = Operators.Curl()
        curlv = curl.(Geometry.Covariant12Vector.(v))
        Spaces.weighted_dss!(curlv)
        # `curl_result_type` is now always `Contravariant123Vector`; the
        # 1st/2nd slots are zero for a Curl{(1,2)}(Cov12) input.
        curl_scalar =
            .-3 .* sin.(3 .* coords.x .+ 4 .* coords.y) .-
            2 .* cos.(coords.x .+ 2 .* coords.y)
        curlv_ref =
            Geometry.Contravariant123Vector.(
                zero.(curl_scalar),
                zero.(curl_scalar),
                curl_scalar,
            )

        @test curlv ≈ curlv_ref rtol = 1e-2
    end

    @testset "curl-curl" begin
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )
        curlcurlv_ref1 =
            .-6 .* cos.(3 .* coords.x .+ 2 .* coords.y) .+
            4 .* sin.(coords.x .+ 2 .* coords.y)
        curlcurlv_ref2 =
            9 .* cos.(3 .* coords.x .+ 2 .* coords.y) .-
            2 .* sin.(coords.x .+ 2 .* coords.y)

        curl = Operators.Curl()
        curlcurlv =
            curl.(
                Geometry.Covariant3Vector.(
                    curl.(Geometry.Covariant12Vector.(v)),
                ),
            )
        Spaces.weighted_dss!(curlcurlv)

        @test Geometry.UVVector.(curlcurlv) ≈
              Geometry.UVVector.(curlcurlv_ref1, curlcurlv_ref2) rtol = 4e-2
    end

    @testset "weak curl-strong curl" begin
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )
        curlcurlv_ref1 =
            .-6 .* cos.(3 .* coords.x .+ 2 .* coords.y) .+
            4 .* sin.(coords.x .+ 2 .* coords.y)
        curlcurlv_ref2 =
            9 .* cos.(3 .* coords.x .+ 2 .* coords.y) .-
            2 .* sin.(coords.x .+ 2 .* coords.y)

        curl = Operators.Curl()
        wcurl = Operators.Curl{Operators.WeakForm}()
        curlcurlv =
            Geometry.UVVector.(
                wcurl.(
                    Geometry.Covariant3Vector.(
                        curl.(Geometry.Covariant12Vector.(v)),
                    ),
                ),
            )
        Spaces.weighted_dss!(curlcurlv)

        @test curlcurlv ≈ Geometry.UVVector.(curlcurlv_ref1, curlcurlv_ref2) rtol =
            4e-2
    end

    @testset "weak curl" begin
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 4 .* coords.y),
            )

        wcurl = Operators.Curl{Operators.WeakForm}()
        curlv = wcurl.(Geometry.Covariant12Vector.(v))
        Spaces.weighted_dss!(curlv)
        curlv_ref =
            .-3 .* sin.(3 .* coords.x .+ 4 .* coords.y) .-
            2 .* cos.(coords.x .+ 2 .* coords.y)

        # `curl_result_type` is now always `Contravariant123Vector`; pad the
        # reference scalar into the 3rd slot.
        @test curlv ≈
              Geometry.Contravariant123Vector.(
            zero.(curlv_ref),
            zero.(curlv_ref),
            curlv_ref,
        ) rtol = 1e-2
    end

    @testset "div" begin
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )

        div = Operators.Divergence()
        divv = div.(v)
        Spaces.weighted_dss!(divv)
        divv_ref =
            cos.(coords.x .+ 2 .* coords.y) .-
            2 .* sin.(3 .* coords.x .+ 2 .* coords.y)

        @test divv ≈ divv_ref rtol = 1e-2
    end


    @testset "weak div" begin
        v =
            Geometry.UVVector.(
                sin.(coords.x .+ 2 .* coords.y),
                cos.(3 .* coords.x .+ 2 .* coords.y),
            )

        wdiv = Operators.Divergence{Operators.WeakForm}()
        divv = wdiv.(v)
        Spaces.weighted_dss!(divv)
        divv_ref =
            cos.(coords.x .+ 2 .* coords.y) .-
            2 .* sin.(3 .* coords.x .+ 2 .* coords.y)

        @test divv ≈ divv_ref rtol = 1e-2
    end


    @testset "annihilator property: curl-grad" begin
        f = sin.(coords.x .+ 2 .* coords.y)

        grad = Operators.Gradient()
        gradf = grad.(f)
        Spaces.weighted_dss!(gradf)

        curl = Operators.Curl()
        curlgradf = curl.(gradf)
        Spaces.weighted_dss!(curlgradf)

        @test norm(curlgradf) < 1e-12
    end

    @testset "annihilator property: div-curl" begin
        v = Geometry.Covariant3Vector.(sin.(coords.x .+ 2 .* coords.y))
        curl = Operators.Curl()
        curlv = curl.(v)
        Spaces.weighted_dss!(curlv)

        div = Operators.Divergence()
        divcurlv = div.(curlv)
        Spaces.weighted_dss!(divcurlv)

        @test norm(divcurlv) < 1e-12
    end

    @testset "scalar hyperdiffusion" begin
        k = 2
        l = 3
        y = @. sin(k * coords.x + l * coords.y)
        ∇⁴y_ref = @. (k^2 + l^2)^2 * sin(k * coords.x + l * coords.y)

        wdiv = Operators.Divergence{Operators.WeakForm}()
        grad = Operators.Gradient()
        χ = Spaces.weighted_dss!(@. wdiv(grad(y)))
        ∇⁴y = Spaces.weighted_dss!(@. wdiv(grad(χ)))

        @test ∇⁴y_ref ≈ ∇⁴y rtol = 2e-2
    end

    @testset "vector hyperdiffusion" begin
        k = 2
        l = 3
        y = @. Geometry.UVVector(sin(k * coords.x + l * coords.y), 0.0)
        ∇⁴y_ref = @. Geometry.UVVector(
            (k^2 + l^2)^2 * sin(k * coords.x + l * coords.y),
            0.0,
        )
        curl = Operators.Curl()
        wcurl = Operators.Curl{Operators.WeakForm}()

        sdiv = Operators.Divergence()
        wgrad = Operators.Gradient{Operators.WeakForm}()

        χ = Spaces.weighted_dss!(
            @. Geometry.UVVector(wgrad(sdiv(y))) - Geometry.UVVector(
                wcurl(
                    Geometry.Covariant3Vector(
                        curl(Geometry.Covariant12Vector(y)),
                    ),
                ),
            )
        )
        ∇⁴y = Spaces.weighted_dss!(
            @. Geometry.UVVector(wgrad(sdiv(χ))) - Geometry.UVVector(
                wcurl(
                    Geometry.Covariant3Vector(
                        curl(Geometry.Covariant12Vector(χ)),
                    ),
                ),
            )
        )

        @test ∇⁴y_ref ≈ ∇⁴y rtol = 2e-2
    end


    @testset "vector hyperdiffusion 3d" begin
        k = 2
        l = 3

        yₕ = @. Geometry.Covariant12Vector.(
            Geometry.UVVector(sin(k * coords.x + l * coords.y), 0.0),
        )
        yᵥ = @. Geometry.Covariant3Vector.(
            Geometry.WVector(sin(k * coords.x + l * coords.y)),
        )

        curl = Operators.Curl()
        wcurl = Operators.Curl{Operators.WeakForm}()

        @test Geometry.Contravariant123Vector.(curl.(yₕ)) .+
              Geometry.Contravariant123Vector.(curl.(yᵥ)) ≈
              curl.(
            Geometry.Covariant123Vector.(yₕ) .+
            Geometry.Covariant123Vector.(yᵥ),
        )
        @test Geometry.Contravariant123Vector.(wcurl.(yₕ)) .+
              Geometry.Contravariant123Vector.(wcurl.(yᵥ)) ≈
              wcurl.(
            Geometry.Covariant123Vector.(yₕ) .+
            Geometry.Covariant123Vector.(yᵥ),
        )

    end

    @testset "operators on masked spaces" begin
        masked_space = Spaces.SpectralElementSpace2D(
            topology,
            quad;
            enable_mask = true,
        )
        Spaces.set_mask!(c -> c.x > 0, masked_space)

        grad = Operators.Gradient()
        wdiv = Operators.Divergence{Operators.WeakForm}()
        f = sin.(coords.x .+ 2 .* coords.y)
        masked_f = zeros(masked_space)
        parent(masked_f) .= parent(f)

        # A spectral operator reads every point of each slab, so its values on
        # active columns match the unmasked computation.
        active = parent(Spaces.get_mask(masked_space).is_active)
        matches_on_active_columns(masked_result, result) = all(
            !active[v, i, j, h, 1] ||
                parent(masked_result)[v, i, j, h, c] ≈ parent(result)[v, i, j, h, c]
            for v in axes(active, 1), i in axes(active, 2),
            j in axes(active, 3), h in axes(active, 4),
            c in axes(parent(result), 5)
        )
        @test matches_on_active_columns(grad.(masked_f), grad.(f))
        @test matches_on_active_columns(
            (@. wdiv(grad(masked_f)) + masked_f),
            (@. wdiv(grad(f)) + f),
        )
    end
end
