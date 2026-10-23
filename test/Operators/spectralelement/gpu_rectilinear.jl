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

# Kernels launched by f, and whether each slice loop kernel gave every slice one thread.
function recorded_kernels(f)
    empty!(CUDA_EXT.RECORDED_KERNELS)
    CUDA_EXT.RECORD_KERNELS[] = true
    try
        f()
    finally
        CUDA_EXT.RECORD_KERNELS[] = false
    end
    return splice!(CUDA_EXT.RECORDED_KERNELS, eachindex(CUDA_EXT.RECORDED_KERNELS))
end
one_thread_per_slice(kernel) =
    typeof(kernel).parameters[1] <: CUDA_EXT.SliceLoopKernel &&
    typeof(kernel).parameters[1].parameters[5] == ClimaCore.DataLayouts.ThisThread

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

# Operators.largest_application_bytes is the shared memory that an expression needs for each
# slice (only one operator application's buffers are live at a time). A broadcast whose
# buffers do not fit in its slices' shared memory is evaluated with one thread per slice.
# Slabs that fit in a warp cache arguments in registers, which need no shared memory.
@testset "shared memory of operator applications" begin
    function check(bc, bc_cpu)
        dest = Base.materialize(bc)
        kernel = last(recorded_kernels(() -> Base.Broadcast.materialize!(dest, bc)))
        op = Operators.slice_operator(bc.style)
        scope = DataLayouts.slice_subscope(CUDA_EXT.ThisKernel(), op, dest, bc)
        slice = op === Fields.column ? Fields.column(bc, 1, 1, 1) : Fields.slab(bc, 1, 1)
        bytes = Operators.largest_application_bytes(DataLayouts.reassign(slice, scope))
        @test Int(CUDA.memory(kernel).shared) == 0 # only dynamic shared memory
        @test one_thread_per_slice(kernel) ==
              (bytes > DataLayouts.total_buffer_limit(scope))
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
            (
                bc(ᶜinterp, bc(ᶠinterp, x)),
                bc(*, bc(ᶜdiv, bc(ᶠgrad, x)), bc(ᶜinterp, bc(ᶠwinterp, x, x))),
                bc(grad, x),
                bc(wdiv, bc(grad, x)),
                bc(curl, u₁₂),
                bc(wcurl, u₁₂₃),
                bc(Operators.SplitDivergence(), uᶜ₁₂, bc(*, 2, x)), # two buffers at once
                bc(lumped, x), # a fine and a coarse slab at once
                bc(Operators.Interpolate(space₈), bc(*, 2, x₄)), # argument and partial result
                bc(Operators.Restrict(space₄), bc(*, 2, x₈)),
            )
        end
    for (expression, expression_cpu) in zip(
        expressions(fields(space), grid_topology),
        expressions(fields(space_cpu), grid_topology_cpu),
    )
        check(expression, expression_cpu)
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

    # An expression whose buffers need more shared memory than its slices have is
    # evaluated with one thread per slab.
    lazy_deep(x) = foldl((y, _) -> bc(wdiv, bc(grad, y)), 1:3; init = x) # 6 operators
    (t₈, t₈_cpu) = map((grid_topology, grid_topology_cpu)) do topology
        space₈ = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{8}())
        map(x -> ntuple(n -> n * x, 6), sin.(Fields.coordinate_field(space₈).x))
    end
    deep_t₈ = Base.Broadcast.instantiate(lazy_deep(t₈))
    check(deep_t₈, lazy_deep(t₈_cpu))
    @test one_thread_per_slice(
        last(recorded_kernels(() -> Base.materialize(deep_t₈))),
    )
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
# subscopes invariant in DataLayouts.subscope_launch_threads.
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
