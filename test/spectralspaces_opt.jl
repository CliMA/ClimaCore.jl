using Test
using JET
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
    return curl.(curl.(field))
end

function opt_Divergence(field)
    div = Operators.Divergence()
    return div.(field)
end

function opt_WeakDivergence(field)
    wdiv = Operators.WeakDivergence()
    return wdiv.(field)
end

function opt_ScalarHyperdiffusion(field)
    grad = Operators.Gradient()
    wdiv = Operators.WeakDivergence()
    χ = Spaces.weighted_dss!(@. wdiv(grad(field)))
    ∇⁴field = Spaces.weighted_dss!(@. wdiv(grad(χ)))
    return ∇⁴field
end

function opt_VectorHyperdiffusion(field)
    curl = Operators.Curl()
    wcurl = Operators.WeakCurl()

    sdiv = Operators.Divergence()
    wgrad = Operators.WeakGradient()

    χ = Spaces.weighted_dss!(
        @. Geometry.UVVector(wgrad(sdiv(field))) -
           Geometry.UVVector(wcurl(Geometry.Covariant3Vector(curl(field))))
    )
    ∇⁴field = Spaces.weighted_dss!(
        @. Geometry.UVVector(wgrad(sdiv(χ))) -
           Geometry.UVVector(wcurl(Geometry.Covariant3Vector(curl(χ))))
    )
    return ∇⁴field
end

# Test that Julia ia able to optimize spectral element operations v1.7+
@static if @isdefined(var"@test_opt")
    @testset "Scalar Field Specctral Element optimizations" begin
        for FT in (Float64,)
            domain = Domains.RectangleDomain(
                Geometry.XPoint{FT}(-pi)..Geometry.XPoint{FT}(pi),
                Geometry.YPoint{FT}(-pi)..Geometry.YPoint{FT}(pi);
                x1periodic = true,
                x2periodic = true,
            )

            Nq = 3
            quad = Spaces.Quadratures.GLL{Nq}()
            #mesh = Meshes.RectilinearMesh(domain, 3, 3)
            #topology = Topologies.Topology2D(mesh)
            mesh = Meshes.RectilinearMesh(domain, 3, 3)
            topology = Topologies.Topology2D(mesh)
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

            @test_opt opt_Gradient(field)
            @test_opt opt_WeakGradient(field)

            @test_opt opt_Curl(vfield)
            @test_opt opt_WeakCurl(vfield)
            @test_opt opt_CurlCurl(vfield)

            @test_opt opt_Divergence(vfield)
            @test_opt opt_WeakDivergence(vfield)

            @test_opt opt_ScalarHyperdiffusion(field)
            # TODO: Work on getting vector hyperdiffusion to optimize
            # after curl operator changes
            #@test_opt opt_VectorHyperdiffusion(vfield)
        end
    end
end
