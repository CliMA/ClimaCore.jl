# `weighted_dss!` must replace every set of nodes sharing a physical position by
# the J-weighted average of their pre-DSS values,
#
#     v ← Σ_g J_g v_g / Σ_g J_g   over the group g of coincident nodes,
#
# and leave unshared nodes alone. The reference groups nodes by coordinate rather
# than by the topology's connectivity tables, so it also covers connectivity that
# is not trivially structured, such as the reversed edge orientations at
# cubed-sphere panel seams. Summation order differs from the one `weighted_dss!`
# uses, hence the roundoff tolerance.
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore:
    Domains, Meshes, Topologies, Spaces, Fields, Geometry, Quadratures
import Random: seed!

# Group values and J-weights by rounded physical position and return the
# J-weighted average for each node.
function grouped_dss_reference(positions, values, weights)
    sums = Dict{NTuple{3, Float64}, Tuple{Float64, Float64}}()
    for (pos, v, w) in zip(positions, values, weights)
        (Σwv, Σw) = get(sums, pos, (0.0, 0.0))
        sums[pos] = (Σwv + w * v, Σw + w)
    end
    return map(positions, values) do pos, v
        (Σwv, Σw) = sums[pos]
        Σwv / Σw
    end
end

# Positions must be grouped with a tolerance: coincident nodes agree only to
# roundoff, while distinct GLL nodes are separated by a finite distance.
round_position(components...) = round.(components, sigdigits = 8)

# The reference groups nodes in a `Dict`, so its arrays must be on the host;
# `weighted_dss!` still runs on the device under test. `to_cpu` rather than
# `Array(parent(_))`: `parent` of a local-geometry component is a strided view.
host_vec(field) = vec(parent(ClimaCore.to_cpu(field)))

@testset "weighted_dss! vs coordinate-grouped reference (cubed sphere)" begin
    FT = Float64
    context = ClimaComms.context()
    domain = Domains.SphereDomain(FT(1))
    mesh = Meshes.EquiangularCubedSphere(domain, 3)
    topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())

    coords = Fields.coordinate_field(space)
    lat = host_vec(coords.lat)
    long = host_vec(coords.long)
    # Group by position on the unit sphere rather than by (lat, long), since
    # longitude wraps at ±180° along one of the panel seams.
    positions = map(
        (lat, long) -> round_position(
            cosd(lat) * cosd(long),
            cosd(lat) * sind(long),
            sind(lat),
        ),
        lat,
        long,
    )

    seed!(1)
    field = Fields.Field(FT, space)
    copyto!(parent(field), rand(FT, size(parent(field))))
    values = host_vec(field)
    weights = host_vec(Fields.local_geometry_field(space).J)

    # Every unique position on this mesh is shared by 1, 2, or 4 nodes
    # (interior, edge, and corner nodes, plus the 3-node cube corners).
    group_sizes = [count(==(p), positions) for p in unique(positions)]
    @test maximum(group_sizes) == 4
    @test count(>(1), group_sizes) > 0

    Spaces.weighted_dss!(field)
    reference = grouped_dss_reference(positions, values, weights)
    @test isapprox(host_vec(field), reference, rtol = sqrt(eps(FT)))

    # DSS is idempotent: the averaged values are already continuous.
    dss_once = host_vec(field)
    Spaces.weighted_dss!(field)
    @test isapprox(host_vec(field), dss_once, rtol = sqrt(eps(FT)))
end

@testset "weighted_dss! vs coordinate-grouped reference (1D, non-periodic)" begin
    FT = Float64
    context = ClimaComms.context()
    domain = Domains.IntervalDomain(
        Geometry.XPoint(FT(0)),
        Geometry.XPoint(FT(1));
        boundary_names = (:left, :right),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 8)
    topology = Topologies.IntervalTopology(context, mesh)
    space = Spaces.SpectralElementSpace1D(topology, Quadratures.GLL{5}())

    positions = map(
        x -> round_position(x, 0.0, 0.0),
        host_vec(Fields.coordinate_field(space).x),
    )

    seed!(1)
    field = Fields.Field(FT, space)
    copyto!(parent(field), rand(FT, size(parent(field))))
    values = host_vec(field)
    weights = host_vec(Fields.local_geometry_field(space).J)

    # Interior element boundaries are shared by exactly two nodes; the two
    # domain endpoints are not shared.
    group_sizes = [count(==(p), positions) for p in unique(positions)]
    @test count(==(2), group_sizes) == 7
    @test count(==(1), group_sizes) == length(unique(positions)) - 7

    Spaces.weighted_dss!(field)
    reference = grouped_dss_reference(positions, values, weights)
    @test isapprox(host_vec(field), reference, rtol = sqrt(eps(FT)))
end
