# Tests of the multi-field DSS (`Spaces.create_dss_buffer(field1, field2, ...)`)
# and of the receive-buffer reduction, shared by the serial, distributed CPU, and
# distributed CUDA test files. Each includer passes its own communications context.
using Test
import Adapt
import Random: MersenneTwister
import ClimaComms
import ClimaCore:
    Domains, Fields, Geometry, Meshes, Quadratures, Spaces, Topologies

# Extruded cubed-sphere center and face spaces sharing their horizontal space,
# mirroring the state of an atmosphere model (`Y.c`, `Y.f`).
function multi_field_dss_spaces(context; ne = 4, nlevels = 5)
    FT = Float64
    hdomain = Domains.SphereDomain(FT(1))
    htopology = Topologies.Topology2D(context, Meshes.EquiangularCubedSphere(hdomain, ne))
    hspace = Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{4}())
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(1));
        boundary_names = (:bottom, :top),
    )
    vtopology = Topologies.IntervalTopology(
        ClimaComms.device(context),
        Meshes.IntervalMesh(vdomain; nelems = nlevels),
    )
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    cspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    return hspace, cspace, fspace
end

# Discontinuous fields with random values, the same on every call. They include
# a vector that DSS transforms to the physical basis, a vertical vector, and a
# field without levels, so the combined buffer holds rows of different widths.
function multi_field_dss_fields(hspace, cspace, fspace, seed)
    FT = Float64
    rng = MersenneTwister(seed)
    c = Fields.Field(
        NamedTuple{(:ρ, :uₕ), Tuple{FT, Geometry.Covariant12Vector{FT}}},
        cspace,
    )
    f = Fields.Field(Geometry.Covariant3Vector{FT}, fspace)
    s = Fields.Field(FT, hspace)
    for field in (c, f, s)
        copyto!(parent(field), rand(rng, FT, size(parent(field))))
    end
    return c, f, s
end

host(field) = Array(parent(field))

# Reference for the receive reduction: the CPU loop on a host copy of the buffer.
function host_dss_buffer(buffer::Topologies.DSSBuffer)
    to_host(x) = Adapt.adapt(Array, x)
    return Topologies.DSSBuffer{eltype(buffer)}(
        buffer.graph_context,
        to_host(buffer.perimeter_data),
        to_host(buffer.send_data),
        to_host(buffer.recv_data),
        to_host(buffer.send_buf_idx),
        to_host(buffer.recv_buf_idx),
        to_host(buffer.buf_offset),
        to_host(buffer.recv_target_idx),
        to_host(buffer.recv_target_offset),
        to_host(buffer.recv_target_rows),
        to_host(buffer.internal_elems),
        to_host(buffer.perimeter_elems),
    )
end

# The receive groups cover every receive row once, group the rows by their
# perimeter node, and list them in ascending order. Returns the number of nodes
# that receive more than one row.
function test_recv_targets(buffer::Topologies.DSSBuffer)
    recv_buf_idx = Array(buffer.recv_buf_idx)
    targets = Array(buffer.recv_target_idx)
    offsets = Array(buffer.recv_target_offset)
    rows = Array(buffer.recv_target_rows)
    @test sort(rows) == 1:size(recv_buf_idx, 1)
    @test length(offsets) == size(targets, 1) + 1
    @test first(offsets) == 1 && last(offsets) == length(rows) + 1
    @test allunique(eachrow(targets))
    for t in axes(targets, 1)
        group = rows[offsets[t]:(offsets[t + 1] - 1)]
        @test !isempty(group) && issorted(group)
        @test all(r -> recv_buf_idx[r, :] == targets[t, :], group)
    end
    return count(>(1), diff(offsets))
end

# The device reduction gives the result of the CPU loop bitwise, and repeated runs
# give the same result.
function test_recv_reduction(buffer::Topologies.DSSBuffer, device)
    isempty(buffer.recv_buf_idx) && return nothing
    rng = MersenneTwister(1)
    perimeter_array = parent(buffer.perimeter_data)
    initial = rand(rng, eltype(perimeter_array), size(perimeter_array))
    copyto!(buffer.recv_data, rand(rng, eltype(buffer.recv_data), length(buffer.recv_data)))
    reference = host_dss_buffer(buffer)
    copyto!(parent(reference.perimeter_data), initial)
    Topologies.load_from_recv_buffer!(ClimaComms.CPUSingleThreaded(), reference)
    for _ in 1:3
        copyto!(perimeter_array, initial)
        Topologies.load_from_recv_buffer!(device, buffer)
        @test Array(perimeter_array) == parent(reference.perimeter_data)
    end
end

function test_multi_field_dss(context)
    device = ClimaComms.device(context)
    hspace, cspace, fspace = multi_field_dss_spaces(context)
    seed = 1234 + ClimaComms.mypid(context)
    fields() = multi_field_dss_fields(hspace, cspace, fspace, seed)

    # Reference: one DSS per field, each with its own buffer.
    reference = map(fields()) do field
        Spaces.weighted_dss!(field, Spaces.create_dss_buffer(field))
        host(field)
    end

    @testset "combined buffer matches single-field DSS bitwise" begin
        c, f, s = fields()
        buffer = Spaces.create_dss_buffer(c, f, s)
        @test buffer isa Topologies.MultiDSSBuffer
        Spaces.weighted_dss!((c, f, s) => buffer)
        @test map(host, (c, f, s)) == reference
        # The buffer is reusable.
        c, f, s = fields()
        Spaces.weighted_dss!((c, f, s) => buffer)
        @test map(host, (c, f, s)) == reference
        # Separate phases, for overlapping communication with computation.
        c, f, s = fields()
        Spaces.weighted_dss_start!((c, f, s), buffer)
        Spaces.weighted_dss_internal!((c, f, s), buffer)
        Spaces.weighted_dss_ghost!((c, f, s), buffer)
        @test map(host, (c, f, s)) == reference
        # Mixed with a single-field pair.
        c, f, s = fields()
        Spaces.weighted_dss!(
            (c, f) => Spaces.create_dss_buffer(c, f),
            s => Spaces.create_dss_buffer(s),
        )
        @test map(host, (c, f, s)) == reference
        # Separate pairs.
        c, f, s = fields()
        Spaces.weighted_dss!(map(x -> x => Spaces.create_dss_buffer(x), (c, f, s))...)
        @test map(host, (c, f, s)) == reference
        # A FieldVector.
        c, f, _ = fields()
        Y = Fields.FieldVector(; c, f)
        Spaces.weighted_dss!(Y, Spaces.create_dss_buffer(Y.c, Y.f))
        @test map(host, (Y.c, Y.f)) == reference[1:2]
        @test_throws ArgumentError Spaces.weighted_dss!((c, f) => buffer)
    end

    @testset "combined buffer shares one exchange" begin
        c, f, s = fields()
        buffer = Spaces.create_dss_buffer(c, f, s)
        singles = map(Spaces.create_dss_buffer, (c, f, s))
        # Callers such as `cartesian_tensor_divergence` assert the inferred type.
        @test (@inferred Spaces.create_dss_buffer(c, f, s)) isa Topologies.MultiDSSBuffer
        @test (@inferred Spaces.create_dss_buffer(c)) isa Topologies.DSSBuffer
        @test all(b -> b.graph_context === buffer.graph_context, buffer.buffers)
        @test all(b -> b.send_data === first(buffer.buffers).send_data, buffer.buffers)
        @test length(first(buffer.buffers).send_data) ==
              sum(b -> length(b.send_data), singles)
        if context isa ClimaComms.MPICommsContext
            # One message per neighbor for all fields, instead of one per field.
            nneighbors = length(Spaces.topology(hspace).neighbor_pids)
            @test length(buffer.graph_context.send_bufs) == nneighbors
            @test length(buffer.graph_context.recv_bufs) == nneighbors
            @test sum(b -> length(b.graph_context.send_bufs), singles) ==
                  3 * nneighbors
        end
    end

    @testset "deterministic receive reduction" begin
        c, f, s = fields()
        buffer = Spaces.create_dss_buffer(c, f, s)
        for b in buffer.buffers
            test_recv_targets(b)
            test_recv_reduction(b, device)
        end
    end
end
