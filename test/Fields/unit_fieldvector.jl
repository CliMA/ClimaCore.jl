#=
julia --check-bounds=yes --project
julia --project
using Revise; include(joinpath("test", "Fields", "unit_fieldvector.jl"))
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

    @test Y == Y
    Ydc = deepcopy(Y)
    Ydc.k.z += 1
    @test !(Ydc == Y)
    # Fields.@rprint_diff(Ydc, Y)
    s = sprint(
        Fields._rprint_diff,
        Ydc,
        Y,
        "Ydc",
        "Y";
        context = IOContext(stdout),
    )
    @test occursin("==================== Difference found:", s)
end

@testset "Nested FieldVector broadcasting with permuted order" begin
    FT = Float32
    context = ClimaComms.context()
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(-3.5),
        Geometry.ZPoint{FT}(0);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain; nelems = 10)
    device = ClimaComms.device()
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)
    horzdomain = Domains.SphereDomain(FT(100))
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 1)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Spaces.Quadratures.GLL{2}()
    space = Spaces.SpectralElementSpace2D(horztopology, quad)

    vars1 = (; # order is different!
        bucket = (; # nesting is needed!
            T = Fields.Field(FT, space),
            W = Fields.Field(FT, space),
        )
    )
    vars2 = (; # order is different!
        bucket = (; # nesting is needed!
            W = Fields.Field(FT, space),
            T = Fields.Field(FT, space),
        )
    )
    Y1 = Fields.FieldVector(; vars1...)
    Y1.bucket.T .= 280.0
    Y1.bucket.W .= 0.05

    Y2 = Fields.FieldVector(; vars2...)
    Y2.bucket.T .= 280.0
    Y2.bucket.W .= 0.05

    Y1 .= Y2 # FieldVector broadcasting
    @test Fields.rcompare(Y1, Y2; strict = false)
end

# https://github.com/CliMA/ClimaCore.jl/issues/1465
@testset "Diagonal FieldVector broadcast expressions" begin
    FT = Float64
    device = ClimaComms.device()
    comms_ctx = ClimaComms.context(device)
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; context = comms_ctx)
    fspace = TU.FaceExtrudedFiniteDifferenceSpace(FT; context = comms_ctx)
    cx = Fields.fill((; a = FT(1), b = FT(2)), cspace)
    cy = Fields.fill((; a = FT(1), b = FT(2)), cspace)
    fx = Fields.fill((; a = FT(1), b = FT(2)), fspace)
    fy = Fields.fill((; a = FT(1), b = FT(2)), fspace)
    Y1 = Fields.FieldVector(; x = cx, y = cy)
    Y2 = Fields.FieldVector(; x = cx, y = cy)
    Y3 = Fields.FieldVector(; x = cx, y = cy)
    Y4 = Fields.FieldVector(; x = cx, y = cy)
    Z = Fields.FieldVector(; x = fx, y = fy)
    function test_fv_allocations!(X1, X2, X3, X4)
        @. X1 += X2 * X3 + X4
        return nothing
    end
    test_fv_allocations!(Y1, Y2, Y3, Y4)
    p_allocated = @allocated test_fv_allocations!(Y1, Y2, Y3, Y4)
    if device isa ClimaComms.AbstractCPUDevice
        @test p_allocated == 0
    elseif device isa ClimaComms.CUDADevice
        @test_broken p_allocated == 0
    end

    bc1 = Base.broadcasted(
        :-,
        Base.broadcasted(:+, Y1, Base.broadcasted(:*, 2, Y2)),
        Base.broadcasted(:*, 3, Y3),
    )
    bc2 = Base.broadcasted(
        :-,
        Base.broadcasted(:+, Y1, Base.broadcasted(:*, 2, Y1)),
        Base.broadcasted(:*, 3, Z),
    )
    @test Fields.is_diagonal_bc(bc1)
    @test !Fields.is_diagonal_bc(bc2)
end

function call_getcolumn(fv, colidx, device)
    ClimaComms.allowscalar(device) do
        fvcol = fv[colidx]
    end
    nothing
end
function call_getproperty(fv)
    fva = fv.c.a
    nothing
end
@testset "FieldVector getindex" begin
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(Float32)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    c = fill((a = Float32(1), b = Float32(2)), cspace)
    f = fill((x = Float32(1), y = Float32(2)), fspace)
    fv = Fields.FieldVector(; c, f)
    colidx = Fields.ColumnIndex((1, 1), 1) # arbitrary index
    device = ClimaComms.device()

    ClimaComms.allowscalar(device) do
        @test all(parent(fv.c.a[colidx]) .== Float32(1))
        @test all(parent(fv.f.y[colidx]) .== Float32(2))
        @test propertynames(fv) == propertynames(fv[colidx])
    end

    # JET tests
    # prerequisite
    call_getproperty(fv) # compile first
    @test_opt call_getproperty(fv)

    call_getcolumn(fv, colidx, device) # compile first
    @test_opt call_getcolumn(fv, colidx, device)
    p = @allocated call_getcolumn(fv, colidx, device)
    if ClimaComms.SingletonCommsContext(device) isa ClimaComms.AbstractCPUDevice
        @test p ≤ 32
    end
end

@testset "FieldVector array_type" begin
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    space = TU.SpectralElementSpace1D(Float32; context)
    xcenters = Fields.coordinate_field(space).x
    y = Fields.FieldVector(x = xcenters)
    @test ClimaComms.array_type(y) == ClimaComms.array_type(device)
    y = Fields.FieldVector(x = xcenters, y = xcenters)
    @test ClimaComms.array_type(y) == ClimaComms.array_type(device)
end

@testset "FieldVector basetype replacement and deepcopy" begin
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
    Y = fill(nt, space)

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

@testset "dss of FieldVectors" begin
    function field_vec(center_space, face_space)
        Y = Fields.FieldVector(
            c = map(Fields.coordinate_field(center_space)) do coord
                FT = Spaces.undertype(center_space)
                (;
                    ρ = FT(coord.lat + coord.long),
                    uₕ = Geometry.Covariant12Vector(
                        FT(coord.lat),
                        FT(coord.long),
                    ),
                )
            end,
            f = map(Fields.coordinate_field(face_space)) do coord
                FT = Spaces.undertype(face_space)
                (; w = Geometry.Covariant3Vector(FT(coord.lat + coord.long)))
            end,
        )
        return Y
    end

    fv = field_vec(toy_sphere(Float64)...)

    c_copy = copy(getproperty(fv, :c))
    f_copy = copy(getproperty(fv, :f))

    # Test that dss_buffer is created and has the correct keys
    dss_buffer = Spaces.create_dss_buffer(fv)
    @test haskey(dss_buffer, :c)
    @test haskey(dss_buffer, :f)

    # Test weighted_dss! with and without preallocated buffer
    Spaces.weighted_dss!(fv, dss_buffer)
    @test getproperty(fv, :c) ≈ Spaces.weighted_dss!(c_copy)
    @test getproperty(fv, :f) ≈ Spaces.weighted_dss!(f_copy)

    fv = field_vec(toy_sphere(Float64)...)
    c_copy = copy(getproperty(fv, :c))
    f_copy = copy(getproperty(fv, :f))

    Spaces.weighted_dss!(fv)
    @test getproperty(fv, :c) ≈ Spaces.weighted_dss!(c_copy)
    @test getproperty(fv, :f) ≈ Spaces.weighted_dss!(f_copy)
end

nothing
