using Test
using LinearAlgebra: ×
import LinearAlgebra as LA
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields, Spaces, Operators, Geometry

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# Allocation gates mirroring `benchmark_ops.jl`. Bare operator applications stay
# at zero allocations and will NOT catch per-slab rebuild allocations (~180 kB
# per call in a real tendency), so these kernels deliberately compose operators
# with `UnionAll` vector constructors and `LA.norm`/`×` — do not simplify them.

div_grad!(dest, ϕ, grad, wdiv) = (@. dest = wdiv(grad(ϕ)); nothing)
grad_only!(dest, ϕ, grad) = (@. dest = grad(ϕ); nothing)
grad_norm!(dest, ϕ, ψ, u, grad) =
    (@. dest = grad((ϕ + ψ) + LA.norm(u)^2 / 2); nothing)
wgrad_div!(dest, u, wgrad, div) = (@. dest = wgrad(div(u)); nothing)
wcurl_curl!(dest, u, curl, wcurl) = (
    @. dest =
        Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(u))));
    nothing
)
u_cross_curl_u!(dest, u, f, curl) = (
    @. dest = Geometry.Contravariant12Vector(u) × (f + curl(u));
    nothing
)

@testset "Spectral element broadcasts do not allocate" begin
    TU.@test_precisions FT begin
        space = TU.SphereSpectralElementSpace(FT)
        ϕ = zeros(space)
        ψ = zeros(space)
        u = Geometry.Covariant12Vector.(zeros(space), zeros(space))
        du = similar(u)
        f = Geometry.Contravariant3Vector.(Geometry.WVector.(ϕ))

        grad = Operators.Gradient()
        wgrad = Operators.Gradient{Operators.WeakForm}()
        div = Operators.Divergence()
        wdiv = Operators.Divergence{Operators.WeakForm}()
        curl = Operators.Curl()
        wcurl = Operators.Curl{Operators.WeakForm}()

        TU.@test_zero_allocations div_grad!(ϕ, ψ, grad, wdiv)
        TU.@test_zero_allocations grad_only!(du, ϕ, grad)
        TU.@test_zero_allocations grad_norm!(du, ϕ, ψ, u, grad)
        TU.@test_zero_allocations wgrad_div!(du, u, wgrad, div)
        # These two apply a `UnionAll` vector constructor on top of an operator.
        TU.@test_zero_allocations wcurl_curl!(du, u, curl, wcurl)
        TU.@test_zero_allocations u_cross_curl_u!(du, u, f, curl)
    end
end
