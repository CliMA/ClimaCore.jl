#=
Spectral-element operators over a one-dimensional horizontal space (operator
axes `(1,)`), checked against the CPU. Covers both a bare
`SpectralElementSpace1D` and an extruded space built on one, since the two reach
the GPU kernels through different space types.
=#
using Test
using ClimaComms, ClimaCore
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

FT = Float64
Nq = 4
quad = Quadratures.GLL{Nq}()

hdomain = Domains.IntervalDomain(
    Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
    periodic = true,
)
hmesh = Meshes.IntervalMesh(hdomain, nelems = 16)

vdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0) .. Geometry.ZPoint{FT}(10);
    boundary_names = (:bottom, :top),
)
vmesh = Meshes.IntervalMesh(vdomain, nelems = 4)

function hspace(device)
    context = ClimaComms.SingletonCommsContext(device)
    topology = Topologies.IntervalTopology(context, hmesh)
    return Spaces.SpectralElementSpace1D(topology, quad)
end

function cspace(device)
    context = ClimaComms.SingletonCommsContext(device)
    vtopology = Topologies.IntervalTopology(context, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    return Spaces.ExtrudedFiniteDifferenceSpace(hspace(device), vspace)
end

# Compare an operator expression applied on the GPU against the CPU. `fields` is
# called with a space and returns the arguments for `op_expr`.
function test_gpu_matches_cpu(name, space_fn, fields, op_expr)
    cpu_args = fields(space_fn(ClimaComms.CPUSingleThreaded()))
    gpu_args = fields(space_fn(ClimaComms.device()))
    @testset "$name" begin
        @test Array(parent(op_expr(gpu_args...))) ≈ parent(op_expr(cpu_args...))
    end
end

grad = Operators.Gradient()
wgrad = Operators.WeakGradient()
div = Operators.Divergence()
wdiv = Operators.WeakDivergence()
curl = Operators.Curl()
wcurl = Operators.WeakCurl()
split_div = Operators.SplitDivergence()

@testset "1D spectral element operators on the GPU" begin
    for (space_name, space_fn) in
        (("SpectralElementSpace1D", hspace), ("extruded 1D", cspace))
        # a scalar, a vector to take divergences of, and covariant vectors whose
        # curls exercise each nonzero Levi-Civita term of a one-axis curl
        scalar(space) = (sin.(Fields.coordinate_field(space).x),)
        function vector(space)
            x = Fields.coordinate_field(space).x
            return (Geometry.UVector.(sin.(x)),)
        end
        covariant2(space) =
            (Geometry.Covariant2Vector.(cos.(Fields.coordinate_field(space).x)),)
        covariant3(space) =
            (Geometry.Covariant3Vector.(sin.(Fields.coordinate_field(space).x)),)
        both(space) = (vector(space)..., scalar(space)...)

        @testset "$space_name" begin
            test_gpu_matches_cpu("Gradient", space_fn, scalar, f -> grad.(f))
            test_gpu_matches_cpu("WeakGradient", space_fn, scalar, f -> wgrad.(f))
            test_gpu_matches_cpu("Divergence", space_fn, vector, u -> div.(u))
            test_gpu_matches_cpu("WeakDivergence", space_fn, vector, u -> wdiv.(u))
            test_gpu_matches_cpu("Curl (Covariant2)", space_fn, covariant2, u -> curl.(u))
            test_gpu_matches_cpu("Curl (Covariant3)", space_fn, covariant3, u -> curl.(u))
            test_gpu_matches_cpu(
                "WeakCurl (Covariant2)",
                space_fn,
                covariant2,
                u -> wcurl.(u),
            )
            test_gpu_matches_cpu(
                "WeakCurl (Covariant3)",
                space_fn,
                covariant3,
                u -> wcurl.(u),
            )
            test_gpu_matches_cpu(
                "SplitDivergence",
                space_fn,
                both,
                (u, f) -> split_div.(u, f),
            )
            # composed operators share shared memory between the two operators
            test_gpu_matches_cpu(
                "Divergence ∘ Gradient",
                space_fn,
                scalar,
                f -> div.(grad.(f)),
            )
            test_gpu_matches_cpu(
                "Curl ∘ Curl",
                space_fn,
                covariant2,
                u -> curl.(Geometry.CovariantVector.(curl.(u))),
            )
        end
    end
end

@testset "1D spectral element operator values on the GPU" begin
    # d/dx sin(x) == cos(x), as a Covariant1Vector on a Cartesian 1D space
    space = hspace(ClimaComms.device())
    x = Fields.coordinate_field(space).x
    @test Geometry.UVector.(grad.(sin.(x))) ≈ Geometry.UVector.(cos.(x)) rtol = 1e-2

    # curl of a Covariant3Vector field is -∂/∂x of its component, in the
    # Contravariant2 direction
    w = Geometry.Covariant3Vector.(sin.(x))
    @test Geometry.Contravariant2Vector.(curl.(w)) ≈
          Geometry.Contravariant2Vector.(.-cos.(x)) rtol = 1e-2
end
