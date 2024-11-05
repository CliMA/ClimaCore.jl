#=
julia --check-bounds=yes --project
julia --project
using Revise; include(joinpath("test", "Fields", "unit_field.jl"))
=#
using Test
using JET

using ClimaComms
ClimaComms.@import_required_backends
using OrderedCollections
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.InputOutput
import ClimaCore.Utilities: PlusHalf
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields,
    slab,
    Domains,
    Topologies,
    Meshes,
    Operators,
    Spaces,
    Geometry,
    Quadratures

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
    device = ClimaComms.CPUSingleThreaded()
    grid_topology =
        Topologies.Topology2D(ClimaComms.SingletonCommsContext(device), mesh)

    quad = Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    return space
end

@testset "1×1 2D domain space" begin
    Nij = 4
    n1 = n2 = 1
    Nh = n1 * n2
    space = spectral_space_2D(n1 = n1, n2 = n2, Nij = Nij)
    device = ClimaComms.device(space)
    ArrayType = ClimaComms.array_type(device)

    data = IJFH{ComplexF64}(ArrayType{Float64}, ones; Nij, Nh = n1 * n2)
    field = Fields.Field(data, space)

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

# https://github.com/CliMA/ClimaCore.jl/issues/1650
@testset "mapreduce inside broadcast expression" begin
    dev = ClimaComms.device()
    context = ClimaComms.context(dev)
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(Float32; context)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    c = fill(
        (
            ∑ab = Float32(0),
            a = ntuple(i -> Float32(1), 3),
            b = ntuple(i -> Float32(2), 3),
        ),
        cspace,
    )

    @test begin
        @. c.∑ab = mapreduce(*, +, c.a, c.b)
        true
    end broken = dev isa ClimaComms.CUDADevice
end

# https://github.com/CliMA/ClimaCore.jl/issues/1126
function pow_n(f)
    @. f.x = f.x^2
    return nothing
end
@testset "Broadcasting with ^n" begin
    FT = Float32
    device = ClimaComms.CPUSingleThreaded() # fill is broken on gpu
    context = ClimaComms.SingletonCommsContext(device)
    for space in TU.all_spaces(FT; context)
        f = fill((; x = FT(1)), space)
        pow_n(f) # Compile first
        p_allocated = @allocated pow_n(f)
        if space isa Spaces.SpectralElementSpace1D
            @test p_allocated == 0
        else
            @test p_allocated == 0 broken = (device isa ClimaComms.CUDADevice)
        end
    end
end

function ifelse_broadcast_allocating(a, b, c)
    FT = eltype(a)
    @. a = ifelse(true || c < b * FT(1), FT(0), c)
    return nothing
end

function ifelse_broadcast_or(a, b, c)
    FT = eltype(a)
    val = FT(1)
    @. a = ifelse(true || c < b * val, FT(0), c)
    return nothing
end

function ifelse_broadcast_simple(a, b, c)
    FT = eltype(a)
    @. a = ifelse(c < b * FT(1), FT(0), c)
    return nothing
end

@testset "Broadcasting ifelse" begin
    FT = Float32
    device = ClimaComms.CPUSingleThreaded() # broken on gpu
    context = ClimaComms.SingletonCommsContext(device)
    for space in (
        TU.CenterExtrudedFiniteDifferenceSpace(FT; context),
        TU.ColumnCenterFiniteDifferenceSpace(FT; context),
    )
        a = Fields.level(fill(FT(0), space), 1)
        b = Fields.level(fill(FT(2), space), 1)
        c = Fields.level(fill(FT(3), space), 1)

        ifelse_broadcast_allocating(a, b, c)
        p_allocated = @allocated ifelse_broadcast_allocating(a, b, c)
        if VERSION ≥ v"1.11.0-beta"
            @test p_allocated == 0
        else
            @test_broken p_allocated == 0
        end

        ifelse_broadcast_or(a, b, c)
        p_allocated = @allocated ifelse_broadcast_or(a, b, c)
        @test p_allocated == 0

        ifelse_broadcast_simple(a, b, c)
        p_allocated = @allocated ifelse_broadcast_simple(a, b, c)
        @test p_allocated == 0
    end
end

# Requires `--check-bounds=yes`
@testset "Constructing & broadcasting over empty fields" begin
    FT = Float32
    for space in TU.all_spaces(FT)
        f = fill((;), space)
        @. f += f
    end

    function test_broken_throws(f)
        try
            @. f += 1
            # we want to throw exception, test is broken
            @test_broken false
        catch
            # we want to throw exception, unexpected pass
            @test_broken true
        end
    end
    empty_field(space) = fill((;), space)

    # Broadcasting over the wrong size should error
    test_broken_throws(empty_field(TU.PointSpace(FT)))
    test_broken_throws(empty_field(TU.SpectralElementSpace1D(FT)))
    test_broken_throws(empty_field(TU.SpectralElementSpace2D(FT)))
    test_broken_throws(empty_field(TU.ColumnCenterFiniteDifferenceSpace(FT)))
    test_broken_throws(empty_field(TU.ColumnFaceFiniteDifferenceSpace(FT)))
    test_broken_throws(empty_field(TU.SphereSpectralElementSpace(FT)))
    test_broken_throws(empty_field(TU.CenterExtrudedFiniteDifferenceSpace(FT)))
    test_broken_throws(empty_field(TU.FaceExtrudedFiniteDifferenceSpace(FT)))

    # TODO: performance optimization: shouldn't we do
    #       nothing when broadcasting over empty fields?
    #       This is otherwise a performance penalty if
    #       users regularly rely on empty fields. In particular:
    #        - does iterating over empty fields load data?
    #        - what is the overhead in iterating over empty fields?
    #        - what is the use case of anything useful that can be
    #          done by iterating over empty fields?
end

@testset "Broadcasting interception for tuple-valued fields" begin
    n1 = n2 = 1
    Nij = 4
    Nh = n1 * n2
    space = spectral_space_2D(n1 = n1, n2 = n2, Nij = Nij)

    S = NamedTuple{(:a, :b), Tuple{Float64, Float64}}
    context = ClimaComms.context(space)
    device = ClimaComms.device(context)
    ArrayType = ClimaComms.array_type(device)
    FT = Spaces.undertype(space)
    data = IJFH{S}(ArrayType{FT}, ones; Nij, Nh)

    nt_field = Fields.Field(data, space)

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

if VERSION < v"1.10"
    # Test truncated field type printing:
    ClimaCore.Fields.truncate_printing_field_types() = true
    @testset "Truncated printing" begin
        nt = (; x = Float64(0), y = Float64(0))
        Y = fill(nt, spectral_space_2D())
        @test sprint(show, typeof(Y); context = IOContext(stdout)) ==
              "Field{(:x, :y)} (trunc disp)"
    end
    ClimaCore.Fields.truncate_printing_field_types() = false

    @testset "Standard printing" begin
        nt = (; x = Float64(0), y = Float64(0))
        Y = fill(nt, spectral_space_2D())
        s = sprint(show, typeof(Y)) # just make sure this doesn't break
    end
end

@testset "Set!" begin
    space = spectral_space_2D()
    FT = Float64
    nt = (; x = FT(0), y = FT(0))
    Y = fill(nt, space)
    foo(local_geom) =
        sin(local_geom.coordinates.x * local_geom.coordinates.y) + 3
    Fields.set!(foo, Y.x)
    @test all((parent(Y.x) .> 1))
end

@testset "PointField" begin
    device = ClimaComms.CPUSingleThreaded() # a bunch of cuda pieces are broken
    context = ClimaComms.SingletonCommsContext(device)
    FT = Float64
    coord = Geometry.XPoint(FT(π))
    space = Spaces.PointSpace(context, coord)
    @test parent(Spaces.local_geometry_data(space)) ==
          FT[Geometry.component(coord, 1), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    field = Fields.coordinate_field(space)
    @test field isa Fields.PointField
    @test Fields.field_values(field)[] == coord

    if ClimaComms.device(context) isa ClimaComms.AbstractCPUDevice
        @test sum(field.x) == FT(π)
    elseif ClimaComms.device(context) isa ClimaComms.CUDADevice
        # Not yet supported
        # @test sum(field.x) == FT(π)
    end

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
        Y = fill((; x = FT(2)), space)
        lg_space = Spaces.level(space, TU.fc_index(1, space))
        lg_field_space = axes(Fields.level(Y, TU.fc_index(1, space)))
        @test all(
            Spaces.local_geometry_data(lg_space).coordinates ===
            Spaces.local_geometry_data(lg_field_space).coordinates,
        )
        @test all(Fields.zeros(lg_space) == Fields.zeros(lg_field_space))
    end
end

@testset "Points from Columns" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        if space isa Spaces.SpectralElementSpace1D
            Y = fill((; x = FT(1)), space)
            point_space_from_field = axes(Fields.column(Y.x, 1, 1))
            point_space = Spaces.column(space, 1, 1)
            @test Fields.ones(point_space) ==
                  Fields.ones(point_space_from_field)
        end
        if space isa Spaces.SpectralElementSpace2D
            Y = fill((; x = FT(1)), space)
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
        Y = fill((; x = FT(1)), space)
        ᶜz_surf =
            Spaces.level(Fields.coordinate_field(Y).z, TU.fc_index(1, space))
        ᶜx_surf = copy(Spaces.level(Y.x, TU.fc_index(1, space)))

        # Still need to define broadcast rules for surface planes with 3D domains
        domain_surface_bc!(Y.x, ᶜz_surf, ᶜx_surf)

        # Skip spaces incompatible with Fields.bycolumn:
        TU.bycolumnable(space) || continue
        Yc = fill((; x = FT(1)), space)
        column_surface_bc!(Yc.x, ᶜz_surf, ᶜx_surf)
        @test Y.x == Yc.x
        nothing
    end
    nothing
end

@testset "Memoization of spaces" begin
    space1 = spectral_space_2D()
    space2 = spectral_space_2D()
    @test space1 === space2
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
    device = ClimaComms.CPUSingleThreaded() # cuda fill is broken
    context = ClimaComms.SingletonCommsContext(device)
    for space in TU.all_spaces(FT; context)
        Y = fill((; a = FT(0), b = FT(1)), space)
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
    context = ClimaComms.SingletonCommsContext()
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
        SMatrix{1, 1}(FT[1]),
        SMatrix{2, 2}(FT[1 2; 3 4]),
        SMatrix{3, 3}(FT[1 2 10; 4 5 6; 7 8 9]),
        SMatrix{3, 3}(FT[1 2 10; 4 5 6; 7 8 9]),
        SMatrix{2, 2}(FT[1 2; 3 4]),
        SMatrix{3, 3}(FT[1 2 10; 4 5 6; 7 8 9]),
    ]

    expected_dzs = [1.0, 4.0, 9.0, 9.0, 1.0, 2.0, 9.0]

    for (components, coord, expected_dz) in
        zip(all_components, coords, expected_dzs)
        CoordType = typeof(coord)
        AIdx = Geometry.coordinate_axis(CoordType)
        at = Geometry.AxisTensor(
            (Geometry.LocalAxis{AIdx}(), Geometry.CovariantAxis{AIdx}()),
            components,
        )
        local_geometry = Geometry.LocalGeometry(coord, FT(1.0), FT(1.0), at)
        space = Spaces.PointSpace(context, local_geometry)
        dz_computed = Array(parent(Fields.Δz_field(space)))
        @test length(dz_computed) == 1
        @test dz_computed[1] == expected_dz
    end
end

@testset "scalar assignment" begin
    device = ClimaComms.CPUSingleThreaded() # constructing space_vijfh is broken
    context = ClimaComms.SingletonCommsContext(device)
    domain_z = Domains.IntervalDomain(
        Geometry.ZPoint(-1.0) .. Geometry.ZPoint(1.0),
        periodic = true,
    )
    mesh_z = Meshes.IntervalMesh(domain_z; nelems = 10)
    topology_z = Topologies.IntervalTopology(context, mesh_z)

    domain_x = Domains.IntervalDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        periodic = true,
    )
    mesh_x = Meshes.IntervalMesh(domain_x; nelems = 10)
    topology_x = Topologies.IntervalTopology(context, mesh_x)

    domain_xy = Domains.RectangleDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
        x1periodic = true,
        x2periodic = true,
    )
    mesh_xy = Meshes.RectilinearMesh(domain_xy, 10, 10)
    topology_xy = Topologies.Topology2D(context, mesh_xy)

    quad = Quadratures.GLL{4}()

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

@testset "Allocation tests for integrals" begin
    FT = Float64
    device = ClimaComms.CPUSingleThreaded()
    context = ClimaComms.SingletonCommsContext(device)
    for space in TU.all_spaces(FT; context)
        # Filter out spaces without z coordinates:
        TU.has_z_coordinates(space) || continue
        Y = fill((; y = FT(1)), space)
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

@testset "ncolumns" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        TU.bycolumnable(space) || continue
        hspace = Spaces.horizontal_space(space)
        Nh = Topologies.nlocalelems(hspace)
        Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(hspace))
        if nameof(typeof(space)) == :SpectralElementSpace1D
            @test Fields.ncolumns(space) == Nh * Nq
        else
            @test Fields.ncolumns(space) == Nh * Nq * Nq
        end
        nothing
    end
    nothing
end

@testset "HF fields" begin
    FT = Float32
    device = ClimaComms.device()
    context = ClimaComms.context(device)
    radius = FT(128)
    zelem = 63
    zlim = (0, 1)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vtopology = Topologies.IntervalTopology(context, vertmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

    hdomain = Domains.SphereDomain(radius)
    helem = 4
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    Nq = 4
    quad = Quadratures.GLL{Nq}()
    hspace = Spaces.SpectralElementSpace2D(
        htopology,
        quad;
        horizontal_layout_type = DataLayouts.IJHF,
    )
    cspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)

    f = fill(FT(0), cspace)
    @. f += 1
end

nothing
