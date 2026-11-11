using Test
using StaticArrays
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
    Quadratures,
    DataLayouts
using LinearAlgebra, IntervalSets
import LinearAlgebra: norm_sqr
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

const FT = Float64
domain = Domains.RectangleDomain(
    Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
    Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
    x1periodic = true,
    x2periodic = true,
)

Nq = 5
quad = Quadratures.GLL{Nq}()

grid_mesh = Meshes.RectilinearMesh(domain, 17, 16)


grid_topology_cpu = Topologies.Topology2D(
    ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    grid_mesh,
)
grid_space_cpu = Spaces.SpectralElementSpace2D(grid_topology_cpu, quad)
coords_cpu = Fields.coordinate_field(grid_space_cpu)

f_cpu = sin.(coords_cpu.x .+ 2 .* coords_cpu.y)
g_cpu =
    Geometry.UVVector.(
        sin.(coords_cpu.x),
        2 .* cos.(coords_cpu.y .+ coords_cpu.x),
    )

grid_topology = Topologies.Topology2D(
    ClimaComms.SingletonCommsContext(ClimaComms.device()),
    grid_mesh,
)
grid_space = Spaces.SpectralElementSpace2D(grid_topology, quad)
coords = Fields.coordinate_field(grid_space)

f = sin.(coords.x .+ 2 .* coords.y)
g = Geometry.UVVector.(sin.(coords.x), 2 .* cos.(coords.y .+ coords.x))

grad = Operators.Gradient()
wgrad = Operators.Gradient{Operators.WeakForm}()

@test Array(parent(grad.(f))) ≈ parent(grad.(f_cpu))
@test Array(parent(wgrad.(f))) ≈ parent(wgrad.(f_cpu))

div = Operators.Divergence()
wdiv = Operators.Divergence{Operators.WeakForm}()

@test Array(parent(div.(g))) ≈ parent(div.(g_cpu))
@test Array(parent(wdiv.(g))) ≈ parent(wdiv.(g_cpu))
@test Array(parent(div.(grad.(f)))) ≈ parent(div.(grad.(f_cpu))) # composite

curl = Operators.Curl()
wcurl = Operators.Curl{Operators.WeakForm}()

@test Array(parent(curl.(Geometry.Covariant12Vector.(g)))) ≈
      parent(curl.(Geometry.Covariant12Vector.(g_cpu)))
@test Array(parent(curl.(Geometry.Covariant3Vector.(f)))) ≈
      parent(curl.(Geometry.Covariant3Vector.(f_cpu)))
@test Array(parent(wcurl.(Geometry.Covariant12Vector.(g)))) ≈
      parent(wcurl.(Geometry.Covariant12Vector.(g_cpu)))
@test Array(parent(wcurl.(Geometry.Covariant3Vector.(f)))) ≈
      parent(wcurl.(Geometry.Covariant3Vector.(f_cpu)))

# The first kernel compiled in a session infers every method it calls, while
# later kernels can reuse methods inferred for earlier ones. Compiled first,
# this curl of projections fails GPU validation if inference leaves a dynamic
# call in it (as Julia 1.10 does for nested ComposedFunction calls in element
# types), so it is also compiled in a new process, where no earlier kernel has
# inferred its methods.
C123 = Geometry.Covariant123Vector
curl_of_projections(f, g) =
    curl.(C123.(Geometry.Covariant12Vector.(g)) .+ C123.(Geometry.Covariant3Vector.(f)))
@test Array(parent(curl_of_projections(f, g))) ≈
      parent(curl_of_projections(f_cpu, g_cpu))
@testset "curl of projections in a new process" begin
    code = """
    using ClimaComms, ClimaCore, IntervalSets
    ClimaComms.@import_required_backends
    import ClimaCore:
        Domains, Fields, Geometry, Meshes, Operators, Quadratures, Spaces, Topologies
    domain = Domains.RectangleDomain(
        Geometry.XPoint{Float64}(-pi) .. Geometry.XPoint{Float64}(pi),
        Geometry.YPoint{Float64}(-pi) .. Geometry.YPoint{Float64}(pi);
        x1periodic = true,
        x2periodic = true,
    )
    context = ClimaComms.SingletonCommsContext(ClimaComms.device())
    topology = Topologies.Topology2D(context, Meshes.RectilinearMesh(domain, 4, 4))
    space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
    (; x, y) = Fields.coordinate_field(space)
    f = sin.(x .+ 2 .* y)
    g = Geometry.UVVector.(sin.(x), 2 .* cos.(y .+ x))
    C123 = Geometry.Covariant123Vector
    result = Operators.Curl().(
        C123.(Geometry.Covariant12Vector.(g)) .+ C123.(Geometry.Covariant3Vector.(f)),
    )
    all(isfinite, Array(parent(result))) || exit(1)
    """
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`
    mktemp() do path, io
        succeeded = success(pipeline(cmd; stdout = io, stderr = io))
        close(io)
        succeeded || @info "Output of the new process" log = read(path, String)
        @test succeeded
    end
end

split_div = Operators.SplitDivergence()

# Test SplitDivergence with vector field and scalar
psi_cpu = f_cpu
psi = f

@test Array(parent(split_div.(g, psi))) ≈ parent(split_div.(g_cpu, psi_cpu))

# A lazy second argument is materialized into a buffer that stays live across
# both per-dimension passes, each of which materializes an equal-size
# contravariant intermediate; see the buffer reuse invariant in apply_operator.
@test Array(parent(split_div.(g, f .* 2 .+ f))) ≈
      parent(split_div.(g_cpu, f_cpu .* 2 .+ f_cpu))

# Interpolation to a lower degree gives the buffered lazy argument and the
# multiply-add's partial result equal byte sizes; see the buffer reuse
# invariant in apply_operator.
IL_cpu = Operators.Interpolate(
    Spaces.SpectralElementSpace2D(grid_topology_cpu, Quadratures.GLL{4}()),
)
IL = Operators.Interpolate(
    Spaces.SpectralElementSpace2D(grid_topology, Quadratures.GLL{4}()),
)
@test Array(parent(IL.(f .* 2 .+ f))) ≈ parent(IL_cpu.(f_cpu .* 2 .+ f_cpu))

# LumpedRestriction publishes its WJ-weighted argument through a slab-sized
# buffer and its coarse values through a smaller one; the fused form reads the
# register-resident result of Gradient. See the buffer reuse invariant in
# apply_operator.
lumped = Operators.LumpedRestriction()
@test Array(parent(lumped.(f))) ≈ parent(lumped.(f_cpu))
@test Array(parent(lumped.(f .* 2 .+ f))) ≈ parent(lumped.(f_cpu .* 2 .+ f_cpu))
@test Array(parent(lumped.(norm_sqr.(grad.(f))))) ≈
      parent(lumped.(norm_sqr.(grad.(f_cpu))))
@test Array(parent(lumped.(grad.(f)))) ≈ parent(lumped.(grad.(f_cpu)))
@test Array(parent(div.(lumped.(grad.(f))))) ≈ parent(div.(lumped.(grad.(f_cpu)))) # composite
mean_op = Operators.LumpedRestriction(Quadratures.GL{1}())
@test Array(parent(mean_op.(norm_sqr.(grad.(f))))) ≈
      parent(mean_op.(norm_sqr.(grad.(f_cpu))))

const CUDA_EXT = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)

# A slice loop over an argument that is already one slice keeps the scope it is
# given; splitting it into smaller subscopes would change how its points are assigned to
# threads and registers (see DataLayouts.slice_exceeds_subscope).
@testset "slice loops over one slice keep their scope" begin
    slice_subscope = ClimaCore.DataLayouts.slice_subscope
    slab = ClimaCore.DataLayouts.slab
    data = Fields.field_values(f)
    subscope = slice_subscope(CUDA_EXT.ThisKernel(), slab, data)
    @test subscope == CUDA_EXT.ThisSubBlock{32}() # Nq * Nq = 25 points per slab
    index = Tuple(first(ClimaCore.DataLayouts.each_slice_index(slab, data)))
    @test slice_subscope(subscope, slab, slab(data, index...)) == subscope
end

# Operators.fits_buffers decides the shared memory of a broadcast's slice loop from the
# buffers of its operator applications (Operators.largest_application_bytes, since only one
# application's buffers are live at a time): the :medium level when they fit, else their
# bytes up to :max, else one thread per slice with private buffers (see
# DataLayouts.foreach_slice). Kernels use
# no static shared memory, and slabs that fit in a warp cache arguments in registers, which
# need no shared memory.
@testset "shared memory of operator applications" begin
    function check(bc, bc_cpu, subscope)
        dest = Base.materialize(bc)
        kernel = last(TU.recorded_kernels(() -> Base.Broadcast.materialize!(dest, bc)))
        @test Int(CUDA.memory(kernel).shared) == 0
        @test TU.slice_subscope_type(kernel) == subscope
        # Some of these operators act on fields that only vary vertically, which gives
        # values near round-off, so the comparison needs an absolute tolerance too.
        @test isapprox(
            Array(parent(dest)),
            parent(Base.materialize(bc_cpu));
            nans = true,
            atol = 1e-12,
            rtol = sqrt(eps(FT)), # passing atol would otherwise set rtol to 0
        )
    end
    bc(f, args...) = Base.Broadcast.broadcasted(f, args...)
    # The subscope of a broadcast whose operator applications need `bytes` of shared memory
    # on each slab of N threads: a sub-block at the :medium level if it holds them (see the
    # CUDA extension's slice_subscope_at_level), whose bytes per thread the device sets, or
    # else with their bytes, or one thread per slab beyond the :max level.
    function subscope_for(bytes, N = 64)
        level(name) = CUDA_EXT.shmem_level_bytes(Val(name), Val(N))
        bytes <= N * level(:medium) && return typeof(CUDA_EXT.ThisSubBlock{N}())
        bytes <= N * level(:max) || return DataLayouts.ThisThread
        return typeof(CUDA_EXT.ThisSubBlock{N, cld(bytes, N)}())
    end

    cpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    (space, space_cpu) = map((nothing, cpu_context)) do context
        kwargs = isnothing(context) ? (;) : (; context)
        TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 63, helem = 2, Nq = 8, kwargs...)
    end
    fields(space) =
        let x = sin.(Fields.coordinate_field(space).z)
            (;
                x,
                u₁₂ = map(_ -> Geometry.Covariant12Vector(FT(1), FT(2)), x),
                u₁₂₃ = map(_ -> Geometry.Covariant123Vector(FT(1), FT(2), FT(3)), x),
                uᶜ₁₂ = map(_ -> Geometry.Contravariant12Vector(FT(1), FT(2)), x),
            )
        end
    ᶜinterp = Operators.InterpolateF2C()
    ᶠinterp = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶠwinterp = Operators.WeightedInterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜdiv = Operators.DivergenceF2C()
    ᶠgrad = Operators.GradientC2F()
    lumped = Operators.LumpedRestriction(Quadratures.GLL{4}())
    expressions(fs, topology) =
        let (; x, u₁₂, u₁₂₃, uᶜ₁₂) = fs
            space₄ = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
            space₈ = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{8}())
            (x₄, x₈) = map(s -> sin.(Fields.coordinate_field(s).x), (space₄, space₈))
            # Each expression with the bytes of shared memory that its operator
            # applications need on a slab, of the 64 points of its destination's slabs, or
            # the 16 points of space₄ for Restrict. Finite difference operators need none
            # (they recompute what does not fit), and a buffer of a slab of Float64s is
            # 512 bytes.
            (
                bc(ᶜinterp, bc(ᶠinterp, x)) => subscope_for(0),
                bc(*, bc(ᶜdiv, bc(ᶠgrad, x)), bc(ᶜinterp, bc(ᶠwinterp, x, x))) =>
                    subscope_for(0),
                bc(grad, x) => subscope_for(1024), # the argument, cached per dimension
                bc(wdiv, bc(grad, x)) => subscope_for(1024),
                bc(curl, u₁₂) => subscope_for(512),
                bc(wcurl, u₁₂₃) => subscope_for(1024),
                # a cached argument and a per-dimension intermediate at once
                bc(Operators.SplitDivergence(), uᶜ₁₂, bc(*, 2, x)) => subscope_for(1024),
                bc(lumped, x) => subscope_for(640), # a fine and a coarse slab at once
                # an argument and a partial result at once
                bc(Operators.Interpolate(space₈), bc(*, 2, x₄)) => subscope_for(640),
                bc(Operators.Restrict(space₄), bc(*, 2, x₈)) => subscope_for(512, 16),
                # a register-resident argument's 64 Covariant12Vectors, republished through
                # a buffer, and the partial result of the same size
                bc(Operators.Restrict(space₄), bc(grad, x₈)) => subscope_for(2048, 16),
            )
        end
    for ((expression, subscope), (expression_cpu, _)) in zip(
        expressions(fields(space), grid_topology),
        expressions(fields(space_cpu), grid_topology_cpu),
    )
        check(expression, expression_cpu, subscope)
    end

    deep(x) = @. wdiv(grad(wdiv(grad(wdiv(grad(wdiv(grad(wdiv(grad(x)))))))))) # 10
    in_registers(field, N) =
        Operators.caches_in_registers(
            CUDA_EXT.ThisSubBlock{N}(),
            Fields.slab(field, 1),
            Val(true),
        )
    @test in_registers(f, 32) # 25 points per slab
    @test Array(parent(deep(f))) ≈ parent(deep(f_cpu))
    wide_quad = Quadratures.GLL{7}() # 49 points per slab, more than a warp has lanes
    wide_f = sin.(
        Fields.coordinate_field(Spaces.SpectralElementSpace2D(grid_topology, wide_quad)).x,
    )
    wide_f_cpu = sin.(
        Fields.coordinate_field(
            Spaces.SpectralElementSpace2D(grid_topology_cpu, wide_quad),
        ).x,
    )
    @test !in_registers(wide_f, 64)
    @test Array(parent(deep(wide_f))) ≈ parent(deep(wide_f_cpu))

    # Gradient of n values per point caches n values per point per dimension: 3 KB for 3
    # values on an 8x8 slab and 6 KB for 6 values (the :medium and :large levels on an
    # A100), however deep the expression, since one application's buffers are live at a
    # time.
    lazy_deep(x, n) = foldl((y, _) -> bc(wdiv, bc(grad, y)), 1:n; init = x)
    tuple_fields(::Val{m}) where {m} = map((grid_topology, grid_topology_cpu)) do topology
        space₈ = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{8}())
        map(x -> ntuple(n -> n * x, Val(m)), sin.(Fields.coordinate_field(space₈).x))
    end
    (t₃, t₃_cpu) = tuple_fields(Val(3))
    check(
        Base.Broadcast.instantiate(lazy_deep(t₃, 1)),
        lazy_deep(t₃_cpu, 1),
        subscope_for(3072),
    )
    (t₆, t₆_cpu) = tuple_fields(Val(6))
    for depth in (1, 3)
        check(
            Base.Broadcast.instantiate(lazy_deep(t₆, depth)),
            lazy_deep(t₆_cpu, depth),
            subscope_for(6144),
        )
    end
end

# A buffered quadrature-weighted argument can alias a same-size per-dimension
# contribution and end up weighted by W², so it is only buffered when the buffer
# is private to one thread; see Operators.materialize_quadrature_weighted.
@testset "weak-form operators applied to a fused scalar argument" begin
    # The hyperdiffusion shapes: Gradient{WeakForm} of a Divergence,
    # Curl{WeakForm} of a Curl.
    @test Array(parent(@. wgrad(div(g)))) ≈ parent(@. wgrad(div(g_cpu)))
    @test Array(
        parent(
            @. wcurl(
                Geometry.Covariant3Vector(curl(Geometry.Covariant12Vector(g))),
            )
        ),
    ) ≈ parent(
        @. wcurl(
            Geometry.Covariant3Vector(curl(Geometry.Covariant12Vector(g_cpu))),
        )
    )
end

# A block holding only part of a sub-block never visits some of a slab's
# points, which only bites once a slab needs more than one warp; see the whole
# subscopes invariant in DataLayouts.subscope_indices.
@testset "slabs that need more than one warp" begin
    wide_mesh = Meshes.RectilinearMesh(domain, 3, 3)
    wide_coords(device, Nq_wide) = Fields.coordinate_field(
        Spaces.SpectralElementSpace2D(
            Topologies.Topology2D(
                ClimaComms.SingletonCommsContext(device),
                wide_mesh,
            ),
            Quadratures.GLL{Nq_wide}(),
        ),
    )
    for Nq_wide in (6, 11) # 36 and 121 points per slab
        c_cpu = wide_coords(ClimaComms.CPUSingleThreaded(), Nq_wide)
        c = wide_coords(ClimaComms.device(), Nq_wide)
        a_cpu = sin.(c_cpu.x .+ 2 .* c_cpu.y)
        a = sin.(c.x .+ 2 .* c.y)
        @test Array(parent(grad.(a))) ≈ parent(grad.(a_cpu))
        @test Array(parent(@. wdiv(grad(a)))) ≈ parent(@. wdiv(grad(a_cpu)))
    end
end
