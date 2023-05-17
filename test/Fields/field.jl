using Test
using ClimaComms
using OrderedCollections
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.Utilities: PlusHalf
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, slab, Domains, Topologies, Meshes, Operators, Spaces, Geometry

using LinearAlgebra: norm
using Statistics: mean
using ForwardDiff

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

function spectral_space_2D(; n1 = 1, n2 = 1, Nij = 4)
    domain = Domains.RectangleDomain(
        Geometry.XPoint(-3.0) .. Geometry.XPoint(5.0),
        Geometry.YPoint(-2.0) .. Geometry.YPoint(8.0),
        x1periodic = false,
        x2periodic = false,
        x1boundary = (:east, :west),
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology =
        Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)

    quad = Spaces.Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    return space
end

@testset "1×1 2D domain space" begin
    Nij = 4
    n1 = n2 = 1
    space = spectral_space_2D(n1 = n1, n2 = n2, Nij = Nij)

    field =
        Fields.Field(IJFH{ComplexF64, Nij}(ones(Nij, Nij, 2, n1 * n2)), space)

    @test sum(field) ≈ Complex(1.0, 1.0) * 8.0 * 10.0 rtol = 10eps()
    @test sum(x -> 3.0, field) ≈ 3 * 8.0 * 10.0 rtol = 10eps()
    @test mean(field) ≈ Complex(1.0, 1.0) rtol = 10eps()
    @test mean(x -> 3.0, field) ≈ 3 rtol = 10eps()
    @test norm(field) ≈ sqrt(2.0) rtol = 10eps()
    @test norm(field, 1) ≈ norm(Complex(1.0, 1.0)) rtol = 10eps()
    @test norm(field, Inf) ≈ norm(Complex(1.0, 1.0)) rtol = 10eps()
    @test norm(field; normalize = false) ≈ sqrt(2.0 * 8.0 * 10.0) rtol = 10eps()
    @test norm(field, 1; normalize = false) ≈
          norm(Complex(1.0, 1.0)) * 8.0 * 10.0 rtol = 10eps()
    @test norm(field, Inf; normalize = false) ≈ norm(Complex(1.0, 1.0)) rtol =
        10eps()

    @test extrema(real, field) == (1.0, 1.0)

    @test Operators.matrix_interpolate(field, 4) ≈
          [Complex(1.0, 1.0) for i in 1:(4 * n1), j in 1:(4 * n2)]


    field_sin = map(x -> sin((x.x) / 2), Fields.coordinate_field(space))
    M = Operators.matrix_interpolate(field_sin, 20)
    @test size(M) == (20, 20)  # 20 x 20 for a 1 element field

    real_field = field.re

    # test broadcasting
    res = field .+ 1
    @test parent(Fields.field_values(res)) == Float64[
        f == 1 ? 2 : 1 for i in 1:Nij, j in 1:Nij, f in 1:2, h in 1:(n1 * n2)
    ]

    res = field.re .+ 1
    @test parent(Fields.field_values(res)) ==
          Float64[2 for i in 1:Nij, j in 1:Nij, f in 1:1, h in 1:(n1 * n2)]

    # test field slab broadcasting
    f1 = ones(space)
    f2 = ones(space)

    for h in 1:(n1 * n2)
        f1_slab = Fields.slab(f1, h)
        f2_slab = Fields.slab(f2, h)
        q = f1_slab .+ f2_slab
        f1_slab .= q .+ f2_slab
    end
    @test all(parent(f1) .== 3)

    point_field = Fields.column(field, 1, 1, 1)
    @test axes(point_field) isa Spaces.PointSpace
end

@testset "Broadcasting interception for tuple-valued fields" begin
    n1 = n2 = 1
    Nij = 4
    space = spectral_space_2D(n1 = n1, n2 = n2, Nij = Nij)

    nt_field = Fields.Field(
        IJFH{NamedTuple{(:a, :b), Tuple{Float64, Float64}}, Nij}(
            ones(Nij, Nij, 2, n1 * n2),
        ),
        space,
    )
    nt_sum = sum(nt_field)
    @test nt_sum isa NamedTuple{(:a, :b), Tuple{Float64, Float64}}
    @test nt_sum.a ≈ 8.0 * 10.0 rtol = 10eps()
    @test nt_sum.b ≈ 8.0 * 10.0 rtol = 10eps()
    @test norm(nt_field) ≈ sqrt(2.0) rtol = 10eps()

    # test scalar asignment
    nt_field.a .= 0.0
    @test sum(nt_field.a) == 0.0
end

@testset "Special case handling for broadcased norm to pass through space local geometry" begin
    space = spectral_space_2D()
    u = Geometry.Covariant12Vector.(ones(space), ones(space))
    @test norm.(u) ≈ hypot(4 / 8 / 2, 4 / 10 / 2) .* ones(space)
end

@testset "FieldVector" begin
    space = spectral_space_2D()
    u = Geometry.Covariant12Vector.(ones(space), ones(space))
    x = Fields.coordinate_field(space)
    y = [1.0, 2.0, 3.0]
    z = 1.0
    Y = Fields.FieldVector(u = u, k = (x = x, y = y, z = z))

    @test propertynames(Y) == (:u, :k)
    @test propertynames(Y.k) == (:x, :y, :z)
    @test Y.u === u
    @test Y.k.x === x
    @test Y.k.y === y
    @test Y.k.z === z

    @test deepcopy(Y).u !== u
    @test deepcopy(Y).k.x !== x
    @test deepcopy(Y).k.y !== y

    @test getfield(deepcopy(Y).u, :space) === space

    Y1 = 2 .* Y
    @test parent(Y1.u) == 2 .* parent(u)
    @test parent(Y1.k.x) == 2 .* parent(x)
    @test Y1.k.y == 2 .* y
    @test Y1.k.z === 2 * z

    Y1 .= Y1 .+ 2 .* Y
    @test parent(Y1.u) == 4 .* parent(u)
    @test parent(Y1.k.x) == 4 .* parent(x)
    @test Y1.k.y == 4 .* y
    @test Y1.k.z === 4 * z

    Y.k.z = 3.0
    @test Y.k.z === 3.0
end

@testset "FieldVector array_type" begin
    space = TU.PointSpace(Float32)
    xcenters = Fields.coordinate_field(space).x
    y = Fields.FieldVector(x = xcenters)
    @test ClimaComms.array_type(y) == Array
end

@testset "FieldVector basetype replacement and deepcopy" begin
    domain_z = Domains.IntervalDomain(
        Geometry.ZPoint(-1.0) .. Geometry.ZPoint(1.0),
        periodic = true,
    )
    mesh_z = Meshes.IntervalMesh(domain_z; nelems = 10)
    topology_z = Topologies.IntervalTopology(mesh_z)

    domain_x = Domains.IntervalDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        periodic = true,
    )
    mesh_x = Meshes.IntervalMesh(domain_x; nelems = 10)
    topology_x = Topologies.IntervalTopology(mesh_x)

    domain_xy = Domains.RectangleDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
        x1periodic = true,
        x2periodic = true,
    )
    mesh_xy = Meshes.RectilinearMesh(domain_xy, 10, 10)
    topology_xy =
        Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh_xy)

    quad = Spaces.Quadratures.GLL{4}()

    space_vf = Spaces.CenterFiniteDifferenceSpace(topology_z)
    space_ifh = Spaces.SpectralElementSpace1D(topology_x, quad)
    space_ijfh = Spaces.SpectralElementSpace2D(topology_xy, quad)
    space_vifh = Spaces.ExtrudedFiniteDifferenceSpace(space_ifh, space_vf)
    space_vijfh = Spaces.ExtrudedFiniteDifferenceSpace(space_ijfh, space_vf)

    space2field(space) = map(
        coord -> (coord, Geometry.Covariant12Vector(1.0, 2.0)),
        Fields.coordinate_field(space),
    )

    Y = Fields.FieldVector(
        field_vf = space2field(space_vf),
        field_if = slab(space2field(space_ifh), 1),
        field_ifh = space2field(space_ifh),
        field_ijf = slab(space2field(space_ijfh), 1, 1),
        field_ijfh = space2field(space_ijfh),
        field_vifh = space2field(space_vifh),
        field_vijfh = space2field(space_vijfh),
        array = [1.0, 2.0, 3.0],
        scalar = 1.0,
    )

    Yf = ForwardDiff.Dual{Nothing}.(Y, 1.0)
    Yf .= Yf .^ 2 .+ Y
    @test all(ForwardDiff.value.(Yf) .== Y .^ 2 .+ Y)
    @test all(ForwardDiff.partials.(Yf, 1) .== 2 .* Y)

    dual_field = Yf.field_vf
    dual_field_original_basetype = similar(Y.field_vf, eltype(dual_field))
    @test eltype(dual_field_original_basetype) === eltype(dual_field)
    @test eltype(parent(dual_field_original_basetype)) === Float64
    @test eltype(parent(dual_field)) === ForwardDiff.Dual{Nothing, Float64, 1}

    object_that_contains_Yf = (; Yf)
    @test axes(deepcopy(Yf).field_vf) === space_vf
    @test axes(deepcopy(object_that_contains_Yf).Yf.field_vf) === space_vf
end

@testset "Scalar field iterator" begin
    space = spectral_space_2D()
    u = Geometry.Covariant12Vector.(ones(space), ones(space))
    x = Fields.coordinate_field(space)
    y = [1.0, 2.0, 3.0]
    z = 1.0
    Y = Fields.FieldVector(u = u, k = (x = x, y = y, z = z))

    prop_chains = Fields.property_chains(Y)
    @test length(prop_chains) == 6
    @test prop_chains[1] == (:u, :components, :data, 1)
    @test prop_chains[2] == (:u, :components, :data, 2)
    @test prop_chains[3] == (:k, :x, :x)
    @test prop_chains[4] == (:k, :x, :y)
    @test prop_chains[5] == (:k, :y)
    @test prop_chains[6] == (:k, :z)

    FT = Float64
    nt =
        (; x = FT(0), y = FT(0), tup = ntuple(i -> (; a = FT(1), b = FT(1)), 2))
    Y = TU.FieldFromNamedTuple(space, nt)

    prop_chains = Fields.property_chains(Y)
    @test prop_chains[1] == (:x,)
    @test prop_chains[2] == (:y,)
    @test prop_chains[3] == (:tup, 1, :a)
    @test prop_chains[4] == (:tup, 1, :b)
    @test prop_chains[5] == (:tup, 2, :a)
    @test prop_chains[6] == (:tup, 2, :b)

    @test Fields.single_field(Y, prop_chains[1]) === Y.x
    @test Fields.single_field(Y, prop_chains[2]) === Y.y
    @test Fields.single_field(Y, prop_chains[3]) === getproperty(Y.tup, 1).a
    @test Fields.single_field(Y, prop_chains[4]) === getproperty(Y.tup, 1).b
    @test Fields.single_field(Y, prop_chains[5]) === getproperty(Y.tup, 2).a
    @test Fields.single_field(Y, prop_chains[6]) === getproperty(Y.tup, 2).b

    for (i, (var, prop_chain)) in enumerate(Fields.field_iterator(Y))
        @test prop_chains[i] == prop_chain
        @test var === Fields.single_field(Y, prop_chain)
    end
end

# Test truncated field type printing:
ClimaCore.Fields.truncate_printing_field_types() = true
@testset "Truncated printing" begin
    nt = (; x = Float64(0), y = Float64(0))
    Y = TU.FieldFromNamedTuple(spectral_space_2D(), nt)
    @test sprint(show, typeof(Y); context = IOContext(stdout)) ==
          "Field{(:x, :y)} (trunc disp)"
end
ClimaCore.Fields.truncate_printing_field_types() = false

@testset "Standard printing" begin
    nt = (; x = Float64(0), y = Float64(0))
    Y = TU.FieldFromNamedTuple(spectral_space_2D(), nt)
    s = sprint(show, typeof(Y)) # just make sure this doesn't break
end

@testset "Set!" begin
    space = spectral_space_2D()
    FT = Float64
    nt = (; x = FT(0), y = FT(0))
    Y = TU.FieldFromNamedTuple(space, nt)
    foo(local_geom) =
        sin(local_geom.coordinates.x * local_geom.coordinates.y) + 3
    Fields.set!(foo, Y.x)
    @test all((parent(Y.x) .> 1))
end

@testset "PointField" begin
    FT = Float64
    coord = Geometry.XPoint(FT(π))
    space = Spaces.PointSpace(coord)
    @test parent(Spaces.local_geometry_data(space)) ==
          FT[Geometry.component(coord, 1), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    field = Fields.coordinate_field(space)
    @test field isa Fields.PointField
    @test Fields.field_values(field)[] == coord

    @test sum(field.x) == FT(π)

    field = ones(space) .* π
    sin_field = sin.(field)
    add_field = field .+ field
    @test isapprox(Fields.field_values(sin_field)[], FT(0.0); atol = √eps(FT))
    @test isapprox(Fields.field_values(add_field)[], FT(2π))
end

@testset "Level" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        TU.levelable(space) || continue
        Y = TU.FieldFromNamedTuple(space, (; x = FT(2)))
        lg_space = Spaces.level(space, TU.fc_index(1, space))
        lg_field_space = axes(Fields.level(Y, TU.fc_index(1, space)))
        @test all(
            lg_space.local_geometry.coordinates ===
            lg_field_space.local_geometry.coordinates,
        )
        @test all(Fields.zeros(lg_space) == Fields.zeros(lg_field_space))
    end
end

@testset "Points from Columns" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        if space isa Spaces.SpectralElementSpace1D
            Y = TU.FieldFromNamedTuple(space, (; x = FT(1)))
            point_space_from_field = axes(Fields.column(Y.x, 1, 1))
            point_space = Spaces.column(space, 1, 1)
            @test Fields.ones(point_space) ==
                  Fields.ones(point_space_from_field)
        end
        if space isa Spaces.SpectralElementSpace2D
            Y = TU.FieldFromNamedTuple(space, (; x = FT(1)))
            point_space_from_field = axes(Fields.column(Y.x, 1, 1, 1))
            point_space = Spaces.column(space, 1, 1, 1)
            @test Fields.ones(point_space) ==
                  Fields.ones(point_space_from_field)
        end

    end
end

@testset "(Domain/Column)-surface broadcasting" begin
    FT = Float64
    function domain_surface_bc!(x, ᶜz_surf, ᶜx_surf)
        @. x = x + ᶜz_surf
        # exercises broadcast_shape(PointSpace, PointSpace)
        @. x = x + (ᶜz_surf * ᶜx_surf)
        nothing
    end
    function column_surface_bc!(x, ᶜz_surf, ᶜx_surf)
        Fields.bycolumn(axes(x)) do colidx
            @. x[colidx] = x[colidx] + ᶜz_surf[colidx]
            # exercises broadcast_shape(PointSpace, PointSpace)
            @. x[colidx] = x[colidx] + (ᶜz_surf[colidx] * ᶜx_surf[colidx])
        end
        nothing
    end
    for space in TU.all_spaces(FT)
        # Filter out spaces without z coordinates:
        TU.has_z_coordinates(space) || continue
        Y = TU.FieldFromNamedTuple(space, (; x = FT(1)))
        ᶜz_surf =
            Spaces.level(Fields.coordinate_field(Y.x).z, TU.fc_index(1, space))
        ᶜx_surf = Spaces.level(Y.x, TU.fc_index(1, space))

        # Still need to define broadcast rules for surface planes with 3D domains
        if nameof(typeof(space)) == :ExtrudedFiniteDifferenceSpace
            @test_broken begin
                try
                    domain_surface_bc!(Y.x, ᶜz_surf)
                    true
                catch
                    false
                end
            end
        else
            domain_surface_bc!(Y.x, ᶜz_surf, ᶜx_surf)
        end
        # Skip spaces incompatible with Fields.bycolumn:
        TU.bycolumnable(space) || continue
        column_surface_bc!(Y.x, ᶜz_surf, ᶜx_surf)
        nothing
    end
    nothing
end

@testset "Broadcasting same spaces different instances" begin
    space1 = spectral_space_2D()
    space2 = spectral_space_2D()
    field1 = ones(space1)
    field2 = 2 .* ones(space2)
    @test Fields.is_diagonalized_spaces(typeof(space1), typeof(space2))
    @test_throws ErrorException(
        "Broacasted spaces are the same ClimaCore.Spaces type but not the same instance",
    ) field1 .= field2

    # turn warning on
    Fields.allow_mismatched_diagonalized_spaces() = true
    @test_warn "Broacasted spaces are the same ClimaCore.Spaces type but not the same instance" field1 .=
        field2
    @test parent(field1) == parent(field2)
    Fields.allow_mismatched_diagonalized_spaces() = false
end

struct InferenceFoo{FT}
    bar::FT
end
Base.broadcastable(x::InferenceFoo) = Ref(x)
@testset "Inference failure message" begin
    function ics_foo(::Type{FT}, lg, foo) where {FT}
        uv = Geometry.UVVector(FT(0), FT(0))
        z = Geometry.Covariant12Vector(uv, lg)
        y = foo.bingo
        return (; x = FT(0) + y)
    end
    function ics_foo_with_field(::Type{FT}, lg, foo, f) where {FT}
        uv = Geometry.UVVector(FT(0), FT(0))
        z = Geometry.Covariant12Vector(uv, lg)
        ζ = f.a
        y = foo.baz
        return (; x = FT(0) + y - ζ)
    end
    function FieldFromNamedTupleBroken(
        space,
        ics::Function,
        ::Type{FT},
        params...,
    ) where {FT}
        lg = Fields.local_geometry_field(space)
        return ics.(FT, lg, params...)
    end
    FT = Float64
    foo = InferenceFoo(2.0)

    for space in TU.all_spaces(FT)
        Y = TU.FieldFromNamedTuple(space, (; a = FT(0), b = FT(1)))
        @test_throws ErrorException("type InferenceFoo has no field bingo") FieldFromNamedTupleBroken(
            space,
            ics_foo,
            FT,
            foo,
        )
        @test_throws ErrorException("type InferenceFoo has no field baz") FieldFromNamedTupleBroken(
            space,
            ics_foo_with_field,
            FT,
            foo,
            Y,
        )
    end

end

@testset "Δz_field" begin
    FT = Float64
    x = FT(1)
    y = FT(2)
    z = FT(3)
    lat, long = FT(4), FT(5)
    x1 = FT(1)
    x2 = FT(2)
    x3 = FT(3)
    coords = [
        Geometry.ZPoint(z),
        Geometry.XZPoint(x, z),
        Geometry.XYZPoint(x, y, z),
        Geometry.LatLongZPoint(lat, long, z),
        Geometry.Cartesian3Point(x3),
        Geometry.Cartesian13Point(x1, x3),
        Geometry.Cartesian123Point(x1, x2, x3),
    ]
    all_components = [
        SMatrix{1, 1, FT}(range(1, 1)...),
        SMatrix{2, 2, FT}(range(1, 4)...),
        SMatrix{3, 3, FT}(range(1, 9)...),
        SMatrix{3, 3, FT}(range(1, 9)...),
        SMatrix{1, 1, FT}(range(1, 1)...),
        SMatrix{2, 2, FT}(range(1, 4)...),
        SMatrix{3, 3, FT}(range(1, 9)...),
    ]

    expected_dzs = [1.0, 4.0, 9.0, 9.0, 1.0, 4.0, 9.0]

    for (components, coord, expected_dz) in
        zip(all_components, coords, expected_dzs)
        CoordType = typeof(coord)
        AIdx = Geometry.coordinate_axis(CoordType)
        at = Geometry.AxisTensor(
            (Geometry.LocalAxis{AIdx}(), Geometry.CovariantAxis{AIdx}()),
            components,
        )
        local_geometry = Geometry.LocalGeometry(coord, FT(1.0), FT(1.0), at)
        space = Spaces.PointSpace(local_geometry)
        dz_computed = parent(Fields.Δz_field(space))
        @test length(dz_computed) == 1
        @test dz_computed[1] == expected_dz
    end
end

@testset "scalar assignment" begin
    domain_z = Domains.IntervalDomain(
        Geometry.ZPoint(-1.0) .. Geometry.ZPoint(1.0),
        periodic = true,
    )
    mesh_z = Meshes.IntervalMesh(domain_z; nelems = 10)
    topology_z = Topologies.IntervalTopology(mesh_z)

    domain_x = Domains.IntervalDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        periodic = true,
    )
    mesh_x = Meshes.IntervalMesh(domain_x; nelems = 10)
    topology_x = Topologies.IntervalTopology(mesh_x)

    domain_xy = Domains.RectangleDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
        x1periodic = true,
        x2periodic = true,
    )
    mesh_xy = Meshes.RectilinearMesh(domain_xy, 10, 10)
    topology_xy =
        Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh_xy)

    quad = Spaces.Quadratures.GLL{4}()

    space_vf = Spaces.CenterFiniteDifferenceSpace(topology_z)
    space_ifh = Spaces.SpectralElementSpace1D(topology_x, quad)
    space_ijfh = Spaces.SpectralElementSpace2D(topology_xy, quad)
    space_vifh = Spaces.ExtrudedFiniteDifferenceSpace(space_ifh, space_vf)
    space_vijfh = Spaces.ExtrudedFiniteDifferenceSpace(space_ijfh, space_vf)

    C = map(x -> Geometry.Covariant12Vector(1.0, 1.0), zeros(space_vifh))
    @test all(==(1.0), parent(C))
    C .= Ref(zero(eltype(C)))
    @test all(==(0.0), parent(C))
end
function integrate_bycolumn!(∫y, Y)
    Fields.bycolumn(axes(Y.y)) do colidx
        Operators.column_integral_definite!(∫y[colidx], Y.y[colidx])
        nothing
    end
end

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "Definite column integrals bycolumn" begin
    FT = Float64
    results = OrderedCollections.OrderedDict()
    ∫y_analytic = 1 - cos(1) - (0 - cos(0))
    function col_field_copy(y)
        col_copy = similar(y[Fields.ColumnIndex((1, 1), 1)])
        return Fields.Field(Fields.field_values(col_copy), axes(col_copy))
    end
    for zelem in (2^2, 2^3, 2^4, 2^5)
        for space in TU.all_spaces(FT; zelem)
            # Filter out spaces without z coordinates:
            TU.has_z_coordinates(space) || continue
            # Skip spaces incompatible with Fields.bycolumn:
            TU.bycolumnable(space) || continue

            Y = TU.FieldFromNamedTuple(space, (; y = FT(1)))
            zcf = Fields.coordinate_field(Y.y).z
            Δz = Fields.Δz_field(axes(zcf))
            Δz_col = Δz[Fields.ColumnIndex((1, 1), 1)]
            Δz_1 = parent(Δz_col)[1]
            key = (space, zelem)
            if !haskey(results, key)
                results[key] =
                    Dict(:maxerr => 0, :Δz_1 => FT(0), :∫y => [], :y => [])
            end
            ∫y = Spaces.level(similar(Y.y), TU.fc_index(1, space))
            ∫y .= 0
            y = Y.y
            @. y .= 1 + sin(zcf)
            # Compute max error against analytic solution
            maxerr = 0
            Fields.bycolumn(axes(Y.y)) do colidx
                Operators.column_integral_definite!(∫y[colidx], y[colidx])
                maxerr = max(
                    maxerr,
                    maximum(abs.(parent(∫y[colidx]) .- ∫y_analytic)),
                )
                nothing
            end
            results[key][:Δz_1] = Δz_1
            results[key][:maxerr] = maxerr
            push!(results[key][:∫y], ∫y)
            push!(results[key][:y], y)
            nothing
        end
    end
    maxerrs = map(key -> results[key][:maxerr], collect(keys(results)))
    Δzs_1 = map(key -> results[key][:Δz_1], collect(keys(results)))
    cr = convergence_rate(maxerrs, Δzs_1)
    @test 2 < sum(abs.(cr)) / length(cr) < 2.01
end

ClimaCore.enable_threading() = false # launching threads allocates
@testset "Allocation tests for integrals" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        # Filter out spaces without z coordinates:
        TU.has_z_coordinates(space) || continue
        Y = TU.FieldFromNamedTuple(space, (; y = FT(1)))
        zcf = Fields.coordinate_field(Y.y).z
        ∫y = Spaces.level(similar(Y.y), TU.fc_index(1, space))
        ∫y .= 0
        y = Y.y
        @. y .= 1 + sin(zcf)
        # Implicit bycolumn
        Operators.column_integral_definite!(∫y, y) # compile first
        p = @allocated Operators.column_integral_definite!(∫y, y)
        @test p == 0
        # Skip spaces incompatible with Fields.bycolumn:
        TU.bycolumnable(space) || continue
        # Explicit bycolumn
        integrate_bycolumn!(∫y, Y) # compile first
        p = @allocated integrate_bycolumn!(∫y, Y)
        @test p == 0
        nothing
    end
    nothing
end
ClimaCore.enable_threading() = false

@testset "ncolumns" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        TU.bycolumnable(space) || continue
        hspace = Spaces.horizontal_space(space)
        Nh = Topologies.nlocalelems(hspace)
        Nq = Spaces.Quadratures.degrees_of_freedom(
            Spaces.quadrature_style(hspace),
        )
        if nameof(typeof(space)) == :SpectralElementSpace1D
            @test Fields.ncolumns(space) == Nh * Nq
        else
            @test Fields.ncolumns(space) == Nh * Nq * Nq
        end
        nothing
    end
    nothing
end
