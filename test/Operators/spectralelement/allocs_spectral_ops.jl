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

include("utils_tensor_divergence.jl")

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
lumped_grad_norm!(dest, ϕ, grad, lumped) =
    (@. dest = lumped(LA.norm_sqr(grad(ϕ))); nothing)
lumped_grad!(dest, ϕ, grad, lumped) = (@. dest = lumped(grad(ϕ)); nothing)
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
        lumped = Operators.LumpedRestriction()

        TU.@test_zero_allocations div_grad!(ϕ, ψ, grad, wdiv)
        TU.@test_zero_allocations grad_only!(du, ϕ, grad)
        TU.@test_zero_allocations grad_norm!(du, ϕ, ψ, u, grad)
        TU.@test_zero_allocations wgrad_div!(du, u, wgrad, div)
        TU.@test_zero_allocations lumped_grad_norm!(ϕ, ψ, grad, lumped)
        # Non-scalar arguments: a vector (rank 1) and the gradient of a vector (rank 2).
        TU.@test_zero_allocations lumped_grad!(du, ϕ, grad, lumped)
        ∇u = grad.(u)
        TU.@test_zero_allocations lumped_grad!(∇u, u, grad, lumped)
        # These two apply a `UnionAll` vector constructor on top of an operator.
        TU.@test_zero_allocations wcurl_curl!(du, u, curl, wcurl)
        TU.@test_zero_allocations u_cross_curl_u!(du, u, f, curl)
    end
end

cartesian_tensor_div!(out, Tc, T, completion) = (
    Operators.cartesian_tensor_divergence!(out, Tc, T, completion);
    nothing
)

# The buffers and completion the gate below measures against. The momentum axis
# is UVW, so `similar(T)` is a wide enough scratch for the rotated tensor.
function tensor_div_alloc_setup(::Type{FT}, space) where {FT}
    coords = Fields.coordinate_field(space)
    v = Geometry.UVVector.(cosd.(coords.lat), sind.(coords.long))
    m = local_cartesian_field(
        space,
        Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5)),
    )
    T = v .⊗ m
    out = Fields.Field(Geometry.UVWVector{FT}, space)
    completion = tensor_div_completion(
        space;
        numflux = Operators.CentralNumericalFlux(identity),
    )
    return (out, similar(T), T, completion)
end

# The tensor divergence reaches two memoized objects — the momentum rotation
# and, on a continuous space, its own DSS buffer — through the untyped object
# cache, so a lost type assertion there surfaces here as boxing.
@testset "Cartesian tensor divergence does not allocate" begin
    TU.@test_precisions FT begin
        for discretization in (Spaces.CG(), Spaces.DG())
            space = tensor_div_sphere_space(FT; helem = 3, discretization)
            out, Tc, T, completion = tensor_div_alloc_setup(FT, space)
            TU.@test_zero_allocations cartesian_tensor_div!(
                out,
                Tc,
                T,
                completion,
            )
        end
    end
    # On an extruded space the rotation comes from the horizontal space, which
    # puts `Spaces.horizontal_space` in the hot path, where it allocates 16 B
    # unless `_momentum_rotation` inlines it (it is marked `@inline` for this).
    # Float64 alone: that inlining does not turn on precision, and each extruded
    # space costs ~10 s to specialize.
    for discretization in (Spaces.CG(), Spaces.DG())
        space = tensor_div_topography_space(
            Float64;
            helem = 3,
            nz = 6,
            discretization,
        )
        out, Tc, T, completion = tensor_div_alloc_setup(Float64, space)
        TU.@test_zero_allocations cartesian_tensor_div!(out, Tc, T, completion)
    end
end

# Every lookup of the object cache never allocates, including one that meets another key with
# the same slot and short hash (which it compares with its key), and one for a level space
# (whose grid is not on the heap).
@testset "memoized objects are looked up without allocating" begin
    space = tensor_div_sphere_space(Float64; helem = 3)
    out, Tc, T, completion = tensor_div_alloc_setup(Float64, space)
    level_space =
        Spaces.level(TU.CenterExtrudedFiniteDifferenceSpace(Float64; zelem = 4), 1)
    χ = Fields.coordinate_field(level_space).lat
    χ_out = similar(χ)
    dg_space = tensor_div_sphere_space(Float64; helem = 3, discretization = Spaces.DG())
    staging_buffer(space) = (Operators._dg_staging_buffer(space, Float64, 2, 10); nothing)
    cartesian_tensor_div!(out, Tc, T, completion)
    Operators.scalar_laplacian!(χ_out, χ)
    staging_buffer(dg_space)
    (; OBJECT_CACHE, ObjectKey) = ClimaCore.Utilities.Cache
    sizehint!(OBJECT_CACHE, 2 * length(OBJECT_CACHE))
    colliding_keys = map(collect(OBJECT_CACHE)) do (key, value)
        colliding_key = ObjectKey(:Dummy, Ref(0))
        while Base.hashindex(colliding_key, length(OBJECT_CACHE.keys)) !=
              Base.hashindex(key, length(OBJECT_CACHE.keys))
            colliding_key = ObjectKey(:Dummy, Ref(0))
        end
        delete!(OBJECT_CACHE, key)
        OBJECT_CACHE[colliding_key] = nothing
        OBJECT_CACHE[key] = value
        return colliding_key
    end
    TU.@test_zero_allocations cartesian_tensor_div!(out, Tc, T, completion)
    TU.@test_zero_allocations Operators.scalar_laplacian!(χ_out, χ)
    TU.@test_zero_allocations staging_buffer(dg_space)
    foreach(key -> delete!(OBJECT_CACHE, key), colliding_keys)
end

# Utilities.Cache.clean_cache!(grid) also releases the objects cached for the grid.
@testset "memoized objects are released with their grid" begin
    space = tensor_div_sphere_space(Float64; helem = 2)
    out, Tc, T, completion = tensor_div_alloc_setup(Float64, space)
    cartesian_tensor_div!(out, Tc, T, completion)
    (; OBJECT_CACHE, clean_cache!) = ClimaCore.Utilities.Cache
    num_owned_objects(grid) = count(key -> key.owner === grid, keys(OBJECT_CACHE))
    @test num_owned_objects(Spaces.grid(space)) > 0
    clean_cache!(Spaces.grid(space))
    @test num_owned_objects(Spaces.grid(space)) == 0
end
