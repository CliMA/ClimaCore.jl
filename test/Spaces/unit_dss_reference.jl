# Reference-independent DSS check: `weighted_dss!` must replace the values at
# every set of nodes that share a physical position with the same J-weighted
# average of their pre-DSS values,
#
#     v ← Σ_g J_g v_g / Σ_g J_g   over the group g of coincident nodes,
#
# and must leave nodes that are not shared unchanged. The reference is computed
# here by grouping nodes by their physical coordinates, with no knowledge of
# the topology's connectivity tables, so it also exercises connectivity that is
# not trivially structured (e.g. the reversed edge orientations at cubed-sphere
# panel seams). Note that the summation order over a group may differ from the
# order used by `weighted_dss!`, so the comparison tolerance must accommodate
# roundoff differences in the sums.
using Test
import ClimaComms
ClimaComms.@import_required_backends
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

@testset "weighted_dss! vs coordinate-grouped reference (cubed sphere)" begin
    FT = Float64
    context = ClimaComms.context()
    domain = Domains.SphereDomain(FT(1))
    mesh = Meshes.EquiangularCubedSphere(domain, 3)
    topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())

    coords = Fields.coordinate_field(space)
    lat = vec(parent(coords.lat))
    long = vec(parent(coords.long))
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
    parent(field) .= rand.(FT)
    values = copy(vec(parent(field)))
    weights = vec(parent(Fields.local_geometry_field(space).J))

    # Every unique position on this mesh is shared by 1, 2, or 4 nodes
    # (interior, edge, and corner nodes, plus the 3-node cube corners).
    group_sizes = [count(==(p), positions) for p in unique(positions)]
    @test maximum(group_sizes) == 4
    @test count(>(1), group_sizes) > 0

    Spaces.weighted_dss!(field)
    reference = grouped_dss_reference(positions, values, weights)
    @test isapprox(vec(parent(field)), reference, rtol = sqrt(eps(FT)))

    # DSS is idempotent: the averaged values are already continuous.
    dss_once = copy(parent(field))
    Spaces.weighted_dss!(field)
    @test isapprox(parent(field), dss_once, rtol = sqrt(eps(FT)))
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

    positions =
        map(x -> round_position(x, 0.0, 0.0), vec(parent(Fields.coordinate_field(space).x)))

    seed!(1)
    field = Fields.Field(FT, space)
    parent(field) .= rand.(FT)
    values = copy(vec(parent(field)))
    weights = vec(parent(Fields.local_geometry_field(space).J))

    # Interior element boundaries are shared by exactly two nodes; the two
    # domain endpoints are not shared.
    group_sizes = [count(==(p), positions) for p in unique(positions)]
    @test count(==(2), group_sizes) == 7
    @test count(==(1), group_sizes) == length(unique(positions)) - 7

    Spaces.weighted_dss!(field)
    reference = grouped_dss_reference(positions, values, weights)
    @test isapprox(vec(parent(field)), reference, rtol = sqrt(eps(FT)))
end
