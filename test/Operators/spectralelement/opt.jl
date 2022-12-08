using Test
using JET
using ClimaComms
using LinearAlgebra, IntervalSets

import ClimaCore:
    Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators


# We need to pull these broadcasted expressions out as
# toplevel functions due to how broadcast expressions are
# lowered so JETTest can have a single callsite to analyze.

function opt_Interpolate(I, field)
    return I.(field)
end

function opt_Restrict(R, Ifield)
    return R.(Ifield)
end

function opt_Gradient(field)
    grad = Operators.Gradient()
    return grad.(field)
end

function opt_WeakGradient(field)
    wgrad = Operators.WeakGradient()
    return wgrad.(field)
end

function opt_Curl(field)
    curl = Operators.Curl()
    return curl.(field)
end

function opt_WeakCurl(field)
    wcurl = Operators.WeakCurl()
    return wcurl.(field)
end

function opt_CurlCurl(field)
    curl = Operators.Curl()
    return curl.(Geometry.CovariantVector.(curl.(field)))
end

function opt_Divergence(field)
    div = Operators.Divergence()
    return div.(field)
end

function opt_WeakDivergence(field)
    wdiv = Operators.WeakDivergence()
    return wdiv.(field)
end

function opt_ScalarDSS(field)
    grad = opt_Gradient(field)
    Spaces.weighted_dss!(grad)
    return grad
end

function opt_VectorDss_Curl(field)
    return Spaces.weighted_dss!(opt_Curl(field))
end

function opt_VectorDss_DivGrad(field)
    sdiv = Operators.Divergence()
    wgrad = Operators.WeakGradient()
    return Spaces.weighted_dss!(@. wgrad(sdiv(field)))
end

function opt_ScalarHyperdiffusion(field)
    grad = Operators.Gradient()
    wdiv = Operators.WeakDivergence()
    χ = Spaces.weighted_dss!(@. wdiv(grad(field)))
    ∇⁴field = Spaces.weighted_dss!(@. wdiv(grad(χ)))
    return ∇⁴field
end

function opt_VectorHyperdiffusion(field)
    scurl = Operators.Curl()
    wcurl = Operators.WeakCurl()

    sdiv = Operators.Divergence()
    wgrad = Operators.WeakGradient()

    χ = Spaces.weighted_dss!(
        @. wgrad(sdiv(field)) - Geometry.Covariant12Vector(
            wcurl(Geometry.Covariant3Vector(scurl(field))),
        )
    )
    ∇⁴ = Spaces.weighted_dss!(
        @. wgrad(sdiv(χ)) - Geometry.Covariant12Vector(
            wcurl(Geometry.Covariant3Vector(scurl(χ))),
        )
    )
    return ∇⁴
end

@static if @isdefined(var"@test_opt")

    filter(@nospecialize(ft)) = ft !== typeof(Base.mapreduce_empty)

    function test_operators(field, vfield)
        @test_opt opt_Gradient(field)
        opt_WeakGradient(field)

        covfield = Geometry.CovariantVector.(vfield)
        @test_opt function_filter = filter opt_Curl(covfield)
        @test_opt function_filter = filter opt_WeakCurl(covfield)
        @test_opt opt_CurlCurl(covfield)

        @test_opt opt_Divergence(vfield)
        @test_opt opt_WeakDivergence(vfield)

        @test_opt function_filter = filter opt_ScalarDSS(field)
        @test_opt function_filter = filter opt_VectorDss_Curl(covfield)
        @test_opt function_filter = filter opt_VectorDss_DivGrad(vfield)

        @test_opt function_filter = filter opt_ScalarHyperdiffusion(field)
        @test_opt function_filter = filter opt_VectorHyperdiffusion(covfield)
    end
end

# Test that Julia ia able to optimize spectral element operations v1.7+
@static if @isdefined(var"@test_opt")

    @testset "Spectral Element 2D Field optimizations" begin
        for FT in (Float64, Float32)
            domain = Domains.RectangleDomain(
                Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
                Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
                x1periodic = true,
                x2periodic = true,
            )

            Nq = 3
            quad = Spaces.Quadratures.GLL{Nq}()
            mesh = Meshes.RectilinearMesh(domain, 3, 3)

            topology = Topologies.DistributedTopology2D(
                ClimaComms.SingletonCommsContext(),
                mesh,
            )
            space = Spaces.SpectralElementSpace2D(topology, quad)

            coords = Fields.coordinate_field(space)

            field = ones(FT, space)
            vfield =
                Geometry.UVVector.(
                    sin.(coords.x .+ 2 .* coords.y),
                    cos.(3 .* coords.x .+ 4 .* coords.y),
                )

            INq = 4
            Iquad = Spaces.Quadratures.GLL{INq}()
            Ispace =
                Spaces.SpectralElementSpace2D(Spaces.topology(space), Iquad)

            I = Operators.Interpolate(Ispace)
            @test_opt opt_Interpolate(I, field)

            Ifield = opt_Interpolate(I, field)
            R = Operators.Restrict(space)
            @test_opt opt_Restrict(R, Ifield)

            test_operators(field, vfield)
        end
    end

    @testset "Spectral Element 3D Hybrid Field optimizations" begin
        for FT in (Float64, Float32)
            xelem = 3
            yelem = 3
            velem = 5
            npoly = 3

            vertdomain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(0),
                Geometry.ZPoint{FT}(1000);
                boundary_tags = (:bottom, :top),
            )
            vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
            vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

            xdomain = Domains.IntervalDomain(
                Geometry.XPoint{FT}(-500) .. Geometry.XPoint{FT}(500),
                periodic = true,
            )
            ydomain = Domains.IntervalDomain(
                Geometry.YPoint{FT}(-100) .. Geometry.YPoint{FT}(100),
                periodic = true,
            )

            horzdomain = Domains.RectangleDomain(xdomain, ydomain)
            horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)
            horztopology = Topologies.DistributedTopology2D(
                ClimaComms.SingletonCommsContext(),
                horzmesh,
            )

            quad = Spaces.Quadratures.GLL{npoly + 1}()
            horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

            hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
                horzspace,
                vert_center_space,
            )
            hv_face_space =
                Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

            for space in (hv_center_space, hv_face_space)
                coords = Fields.coordinate_field(space)
                field = ones(FT, space)
                vfield =
                    Geometry.UVWVector.(
                        sin.(coords.x .+ 2 .* coords.y),
                        cos.(3 .* coords.x .+ 4 .* coords.y),
                        zero(FT),
                    )
                test_operators(field, vfield)
            end
        end
    end
end
