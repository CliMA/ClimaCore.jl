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
    Quadratures
using LinearAlgebra, IntervalSets

FT = Float64
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

# Check the equal sizes invariant in the CUDA extension's shmem_pointer on
# compiled kernels: allocations share a global only when they ask for the same
# number of bytes AND are emitted from separately compiled functions.
const CUDA_EXT = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
# N Float64s, so 8N bytes; tag makes call sites separately compiled functions.
@noinline function scoped_edge(::Val{N}, ::Val{tag} = Val(1)) where {N, tag}
    array = ClimaCore.DataLayouts.scoped_static_array(CUDA_EXT.ThisBlock(), Float64, (N,))
    i = Int(CUDA.threadIdx().x)
    @inbounds array[i] = i * tag
    CUDA.sync_threads()
    return @inbounds array[N + 1 - i]
end
two_shared_sizes!(dest) =
    (@inbounds dest[1] = scoped_edge(Val(256)) + scoped_edge(Val(512)); nothing)
one_shared_size!(dest) =
    (@inbounds dest[1] = scoped_edge(Val(256)) + scoped_edge(Val(256), Val(2)); nothing)
function one_function_two_arrays!(dest)
    a = ClimaCore.DataLayouts.scoped_static_array(CUDA_EXT.ThisBlock(), Float64, (256,))
    b = ClimaCore.DataLayouts.scoped_static_array(CUDA_EXT.ThisBlock(), Float64, (256,))
    i = Int(CUDA.threadIdx().x)
    @inbounds a[i] = i
    @inbounds b[i] = 2 * i
    CUDA.sync_threads()
    @inbounds dest[1] = a[257 - i] + b[257 - i]
    return nothing
end
@testset "shared memory globals are sized by their allocations" begin
    llvm(f) = sprint() do io
        arg_types = Tuple{CUDA.CuDeviceVector{Float64, CUDA.AS.Global}}
        CUDA.code_llvm(io, f, arg_types; dump_module = true, always_inline = true)
    end
    shared_sizes(f) =
        sort([
            parse(Int, m[1]) for m in eachmatch(r"addrspace\(3\) global \[(\d+) x", llvm(f))
        ])
    @test shared_sizes(two_shared_sizes!) == [2048, 4096]
    @test shared_sizes(one_shared_size!) == [2048] # equal sizes are merged
    @test shared_sizes(one_function_two_arrays!) == [2048, 2048] # never merged
end

# A slice loop over an argument that is already one slice keeps the scope it is
# given; splitting it into smaller subscopes would double the shared memory
# reserved for every buffer (see Operators.inlined_buffer_bytes).
@testset "slice loops over one slice keep their scope" begin
    slice_subscope = ClimaCore.DataLayouts.slice_subscope
    slab = ClimaCore.DataLayouts.slab
    data = Fields.field_values(f)
    subscope = slice_subscope(CUDA_EXT.ThisKernel(), slab, data)
    @test subscope == CUDA_EXT.ThisSubBlock{32}() # Nq * Nq = 25 points per slab
    index = Tuple(first(ClimaCore.DataLayouts.each_slice_index(slab, data)))
    @test slice_subscope(subscope, slab, slab(data, index...)) == subscope
end

# Check Operators.inlined_buffer_bytes, the estimate that picks between inlined
# operator applications (each with its own buffers) and separately compiled ones
# (which share buffers), and that an expression on either side matches the CPU.
@testset "inlined and separately compiled operator applications" begin
    bc(f, args...) = Base.Broadcast.broadcasted(f, args...)
    bytes = Operators.inlined_buffer_bytes
    app = 2 * sizeof(Geometry.Covariant12Vector{FT}) # one application's buffers
    @test (bytes(f), bytes(bc(grad, f)), bytes(bc(wdiv, bc(grad, f)))) ==
          (0, app, 2 * app)
    @test bytes(bc(+, bc(wdiv, bc(grad, f)), bc(div, bc(grad, f)))) == 4 * app
    deep(x) = @. wdiv(grad(wdiv(grad(wdiv(grad(wdiv(grad(wdiv(grad(x)))))))))) # 10
    @test 2 * app <= Operators.MAX_INLINED_BUFFER_BYTES < 10 * app
    @test Array(parent(deep(f))) ≈ parent(deep(f_cpu))
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
