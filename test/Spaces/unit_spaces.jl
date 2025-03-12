#=
julia --project
using Revise; include(joinpath("test", "Spaces", "unit_spaces.jl"))
=#
using Test
using ClimaComms
using StaticArrays, IntervalSets, LinearAlgebra
import Adapt
ClimaComms.@import_required_backends

import ClimaCore:
    slab,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    DataLayouts,
    Geometry,
    Operators,
    DeviceSideContext,
    DeviceSideDevice

using ClimaCore.CommonSpaces
using ClimaCore.Utilities.Cache
import ClimaCore.DataLayouts: IJFH, VF, slab_index

on_gpu = ClimaComms.device() isa ClimaComms.CUDADevice

@testset "2D spaces with mask" begin
    # We need to test a fresh instance of the spaces, since
    # masked spaces include data set by users.
    Cache.clean_cache!()
    FT = Float64
    context = ClimaComms.context()
    x_max = FT(1)
    y_max = FT(1)
    x_elem = 2
    y_elem = 2
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(y_max)),
        Geometry.YPoint(y_max);
        periodic = true,
    )
    domain = Domains.RectangleDomain(x_domain, y_domain)
    hmesh = Meshes.RectilinearMesh(domain, x_elem, y_elem)

    quad = Quadratures.GL{1}()
    htopology = Topologies.Topology2D(context, hmesh)
    # Test for no-mask case
    hspace = Spaces.SpectralElementSpace2D(htopology, quad; enable_mask = false)
    @test Spaces.get_mask(hspace) isa DataLayouts.NoMask

    # Tests with mask
    hspace = Spaces.SpectralElementSpace2D(htopology, quad; enable_mask = true)
    mask = Spaces.get_mask(hspace)
    @test mask isa DataLayouts.IJHMask
    @test all(x -> x == true, parent(mask.is_active)) # test that default is true
    Spaces.set_mask!(hspace) do coords
        coords.x > 0.5
    end
    @test count(parent(mask.is_active)) == 2
    @test length(parent(mask.is_active)) == 4

    f = Fields.Field(FT, hspace)
    fill!(parent(f), 0)
    @. f = 1 # tests fill!
    @test count(iszero, parent(f)) == 2
    ᶜx = Fields.coordinate_field(hspace).x
    @. f = 1 + ᶜx * 0 # tests copyto!
    @test count(iszero, parent(f)) == 2

    fbc = @. 1 + ᶜx * 0 # tests copy
    @test Spaces.get_mask(axes(fbc)) isa DataLayouts.IJHMask

    FT = Float64
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
        enable_mask = true,
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠcoords = Fields.coordinate_field(ᶠspace)
    hᶠcoords = Fields.coordinate_field(Spaces.horizontal_space(ᶠspace))
    mask = Spaces.get_mask(ᶜspace)
    @test mask isa DataLayouts.IJHMask

    # Test that mask-field assignment works:
    # TODO: we should make this easier
    is_active = similar(mask.is_active)
    parent(is_active) .= parent(hᶠcoords.lat) .> 0.5
    Spaces.set_mask!(ᶜspace, is_active)

    Spaces.set_mask!(ᶜspace) do coords
        coords.lat > 0.5
    end
    @test count(parent(mask.is_active)) == 4640
    @test length(parent(mask.is_active)) == 9600
    ᶜf = zeros(ᶜspace)
    @. ᶜf = 1 # tests fill!
    @test count(x -> x == 1, parent(ᶜf)) == 4640 * Spaces.nlevels(axes(ᶜf))
    @test length(parent(ᶜf)) == 9600 * Spaces.nlevels(axes(ᶜf))
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶜf = zeros(ᶜspace)
    @. ᶜf = 1 + 0 * ᶜz # tests copyto!
    @test count(x -> x == 1, parent(ᶜf)) == 4640 * Spaces.nlevels(axes(ᶜf))
    @test length(parent(ᶜf)) == 9600 * Spaces.nlevels(axes(ᶜf))

    ᶠf = zeros(ᶠspace)
    c = zeros(ᶜspace)
    div = Operators.DivergenceF2C()
    foo(f, cf) = cf.lat > 0.5 ? zero(f) : sqrt(-1) # results in NaN in masked regions
    @. c = div(Geometry.WVector(foo(ᶠf, ᶠcoords)))
    @test count(isnan, parent(c)) == 0

    ᶜspace_no_mask = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ᶠspace_no_mask = Spaces.face_space(ᶜspace_no_mask)
    ᶠcoords_no_mask = Fields.coordinate_field(ᶠspace_no_mask)
    c_no_mask = Fields.Field(FT, ᶜspace_no_mask)
    @test_throws ErrorException("Broacasted spaces are not the same.") @. c_no_mask +
                                                                          ᶜf
    ᶠf_no_mask = Fields.Field(FT, ᶠspace_no_mask)
    if ClimaComms.device(ᶜspace_no_mask) isa ClimaComms.CUDADevice
        @. c_no_mask = div(Geometry.WVector(foo(ᶠf_no_mask, ᶠcoords_no_mask)))
        @test count(isnan, parent(c_no_mask)) == 49600
    else
        @test_throws DomainError begin
            @. c_no_mask =
                div(Geometry.WVector(foo(ᶠf_no_mask, ᶠcoords_no_mask)))
        end
    end

end

@testset "1d domain space" begin
    FT = Float64
    domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        periodic = true,
    )
    device = ClimaComms.device()
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(device, mesh)

    quad = Quadratures.GLL{4}()
    points, weights = Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace1D(topology, quad)


    expected_repr = """
    SpectralElementSpace1D:
      mask_enabled: false
      context: SingletonCommsContext using $(nameof(typeof(device)))
      mesh: 1-element IntervalMesh of IntervalDomain: x ∈ [-3.0,5.0] (periodic)
      quadrature: 4-point Gauss-Legendre-Lobatto quadrature"""

    @test repr(space) === expected_repr

    coord_data = Spaces.coordinates_data(space)
    @test eltype(coord_data) == Geometry.XPoint{Float64}

    @test DataLayouts.farray_size(Spaces.coordinates_data(space)) == (4, 1, 1)
    coord_slab = Adapt.adapt(Array, slab(Spaces.coordinates_data(space), 1))
    @test coord_slab[slab_index(1)] == Geometry.XPoint{FT}(-3)
    @test typeof(coord_slab[slab_index(4)]) == Geometry.XPoint{FT}
    @test coord_slab[slab_index(4)].x ≈ FT(5)

    local_geometry_slab =
        Adapt.adapt(Array, slab(Spaces.local_geometry_data(space), 1))
    dss_weights_slab = Adapt.adapt(Array, slab(space.grid.dss_weights, 1))

    for i in 1:4
        @test Geometry.components(local_geometry_slab[slab_index(i)].∂x∂ξ) ≈
              @SMatrix [8 / 2]
        @test Geometry.components(local_geometry_slab[slab_index(i)].∂ξ∂x) ≈
              @SMatrix [2 / 8]
        @test local_geometry_slab[slab_index(i)].J ≈ (8 / 2)
        @test local_geometry_slab[slab_index(i)].WJ ≈ (8 / 2) * weights[i]
        if i in (1, 4)
            @test dss_weights_slab[slab_index(i)] ≈ 1 / 2
        else
            @test dss_weights_slab[slab_index(i)] ≈ 1
        end
    end

    @test Spaces.local_geometry_type(typeof(space)) <: Geometry.LocalGeometry

    point_space = Spaces.column(space, 1, 1)
    @test point_space isa Spaces.PointSpace
    @test parent(Spaces.coordinates_data(point_space)) ==
          parent(Spaces.column(coord_data, 1, 1))
    @test Spaces.local_geometry_type(typeof(point_space)) <:
          Geometry.LocalGeometry
end

on_gpu || @testset "extruded (2d 1×3) finite difference space" begin

    FT = Float32

    device = ClimaComms.device()
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(10);
        boundary_names = (:bottom, :top),
    )
    vertmesh =
        Meshes.IntervalMesh(vertdomain, Meshes.Uniform(), nelems = 10)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(device, vertmesh)
    # Generate Horizontal Space
    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(0),
        Geometry.XPoint{FT}(10);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = 5)
    horztopology = Topologies.IntervalTopology(device, horzmesh)
    quad = Quadratures.GLL{4}()

    hspace = Spaces.SpectralElementSpace1D(horztopology, quad)
    # Extrusion
    f_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vert_face_space)
    c_space = Spaces.CenterExtrudedFiniteDifferenceSpace(f_space)

    @test f_space == Spaces.face_space(f_space)
    @test c_space == Spaces.center_space(f_space)
    @test f_space == Spaces.face_space(c_space)
    @test c_space == Spaces.center_space(c_space)

    s = DataLayouts.farray_size(Spaces.coordinates_data(c_space))
    z = Fields.coordinate_field(c_space).z
    @test s == (10, 4, 2, 5) # 10V, 4I, 2F(x,z), 5H
    @test Spaces.local_geometry_type(typeof(f_space)) <: Geometry.LocalGeometry
    @test Spaces.local_geometry_type(typeof(c_space)) <: Geometry.LocalGeometry

    @test Spaces.z_min(f_space) == 0
    @test Spaces.z_max(f_space) == 10

    @test Spaces.z_min(c_space) == 0
    @test Spaces.z_max(c_space) == 10

    # Define test col index
    colidx = Fields.ColumnIndex{1}((4,), 5)
    z_values = Fields.field_values(z[colidx])
    # Here valid `colidx` are `Fields.ColumnIndex{1}((1:4,), 1:5)`
    @test DataLayouts.farray_size(z_values) == (10, 1)
    @test z_values isa DataLayouts.VF
    @test Spaces.column(z, 1, 1, 1) isa Fields.Field
    @test_throws BoundsError Spaces.column(z, 1, 2, 1)
    @test Spaces.column(z, 1, 2) isa Fields.Field
end

@testset "finite difference space" begin
    FT = Float64
    context = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0) .. Geometry.ZPoint{FT}(5),
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(context, mesh)

    c_space = Spaces.CenterFiniteDifferenceSpace(topology)
    f_space = Spaces.FaceFiniteDifferenceSpace(topology)
    @test repr(c_space) == """
    CenterFiniteDifferenceSpace:
      context: SingletonCommsContext using CPUSingleThreaded
      mesh: 1-element IntervalMesh of IntervalDomain: z ∈ [0.0,5.0] (:bottom, :top)"""

    @test f_space == Spaces.face_space(f_space)
    @test c_space == Spaces.center_space(f_space)
    @test f_space == Spaces.face_space(c_space)
    @test c_space == Spaces.center_space(c_space)

    coord_data = Spaces.coordinates_data(c_space)
    point_space = Spaces.level(c_space, 1)
    @test point_space isa Spaces.PointSpace
    @test Spaces.coordinates_data(point_space)[] ==
          Spaces.level(coord_data, 1)[]

    @test Spaces.local_geometry_type(typeof(c_space)) <: Geometry.LocalGeometry

    x_max = FT(1)
    y_max = FT(1)
    x_elem = 2
    y_elem = 2
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(y_max)),
        Geometry.YPoint(y_max);
        periodic = true,
    )
    domain = Domains.RectangleDomain(x_domain, y_domain)
    hmesh = Meshes.RectilinearMesh(domain, x_elem, y_elem)

    quad = Quadratures.GL{1}()
    htopology = Topologies.Topology2D(context, hmesh)
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)

    @test collect(Spaces.unique_nodes(hspace)) ==
          [((1, 1), 1);;; ((1, 1), 2);;; ((1, 1), 3);;; ((1, 1), 4)]
    @test length(Spaces.unique_nodes(hspace)) == 4
    @test length(Spaces.all_nodes(hspace)) == 4

    @static if on_gpu
        adapted_space = Adapt.adapt(CUDA.KernelAdaptor(), c_space)
        @test ClimaComms.context(adapted_space) == DeviceSideContext()
        @test ClimaComms.device(adapted_space) == DeviceSideDevice()

        adapted_hspace = Adapt.adapt(CUDA.KernelAdaptor(), hspace)
        @test ClimaComms.context(adapted_hspace) == DeviceSideContext()
        @test ClimaComms.device(adapted_hspace) == DeviceSideDevice()
    end

end

@testset "1×1 domain space" begin
    FT = Float32
    context = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        Geometry.YPoint{FT}(-2) .. Geometry.YPoint{FT}(8),
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    grid_topology = Topologies.Topology2D(context, mesh)

    quad = Quadratures.GLL{4}()
    points, weights = Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    @test repr(space) == """
    SpectralElementSpace2D:
      mask_enabled: false
      context: SingletonCommsContext using CPUSingleThreaded
      mesh: 1×1-element RectilinearMesh of RectangleDomain: x ∈ [-3.0,5.0] (periodic) × y ∈ [-2.0,8.0] (:south, :north)
      quadrature: 4-point Gauss-Legendre-Lobatto quadrature"""

    coord_data = Spaces.coordinates_data(space)
    @test DataLayouts.farray_size(coord_data) == (4, 4, 2, 1)
    coord_slab = slab(coord_data, 1)
    @test coord_slab[slab_index(1, 1)] ≈ Geometry.XYPoint{FT}(-3.0, -2.0)
    @test coord_slab[slab_index(4, 1)] ≈ Geometry.XYPoint{FT}(5.0, -2.0)
    @test coord_slab[slab_index(1, 4)] ≈ Geometry.XYPoint{FT}(-3.0, 8.0)
    @test coord_slab[slab_index(4, 4)] ≈ Geometry.XYPoint{FT}(5.0, 8.0)

    @test Spaces.local_geometry_type(typeof(space)) <: Geometry.LocalGeometry
    local_geometry_slab = slab(Spaces.local_geometry_data(space), 1)
    dss_weights_slab = slab(Spaces.dss_weights(space), 1)

    @static if on_gpu
        adapted_space = Adapt.adapt(CUDA.KernelAdaptor(), space)
        @test ClimaComms.context(adapted_space) == DeviceSideContext()
        @test ClimaComms.device(adapted_space) == DeviceSideDevice()
    end

    for i in 1:4, j in 1:4
        @test Geometry.components(local_geometry_slab[slab_index(i, j)].∂x∂ξ) ≈
              @SMatrix [8/2 0; 0 10/2]
        @test Geometry.components(local_geometry_slab[slab_index(i, j)].∂ξ∂x) ≈
              @SMatrix [2/8 0; 0 2/10]
        @test local_geometry_slab[slab_index(i, j)].J ≈ (10 / 2) * (8 / 2)
        @test local_geometry_slab[slab_index(i, j)].WJ ≈
              (10 / 2) * (8 / 2) * weights[i] * weights[j]
        if i in (1, 4)
            @test dss_weights_slab[slab_index(i, j)] ≈ 1 / 2
        else
            @test dss_weights_slab[slab_index(i, j)] ≈ 1
        end
    end

    boundary_surface_geometries = Spaces.grid(space).boundary_surface_geometries
    @test length(boundary_surface_geometries) == 2
    @test keys(boundary_surface_geometries) == (:south, :north)
    @test sum(parent(boundary_surface_geometries.north.sWJ)) ≈ 8
    @test parent(boundary_surface_geometries.north.normal)[1, :, 1] ≈ [0.0, 1.0]

    point_space = Spaces.column(space, 1, 1, 1)
    @test point_space isa Spaces.PointSpace
    @test Spaces.coordinates_data(point_space)[] ==
          Spaces.column(coord_data, 1, 1, 1)[]
end

@testset "2D perimeter iterator on 2×2 rectangular mesh" begin
    context = ClimaComms.SingletonCommsContext()
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(-2π),
            Geometry.XPoint(2π),
            periodic = true,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(-2π),
            Geometry.YPoint(2π),
            periodic = true,
        ),
    )
    n1, n2 = 2, 2
    Nq = 5
    quad = Quadratures.GLL{Nq}()
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    perimeter = Spaces.perimeter(space)
    @test Spaces.local_geometry_type(typeof(space)) <: Geometry.LocalGeometry


    reference = [
        (1, 1),  # vertex 1
        (Nq, 1), # vertex 2
        (Nq, Nq),# vertex 3
        (1, Nq), # vertex 4
        (2, 1),  # face 1
        (3, 1),
        (4, 1),
        (Nq, 2), # face 2
        (Nq, 3),
        (Nq, 4),
        (4, Nq), # face 3
        (3, Nq),
        (2, Nq),
        (1, 4),  # face 4
        (1, 3),
        (1, 2),
    ]
    for (p, (ip, jp)) in enumerate(perimeter)
        @test (ip, jp) == reference[p] # face_node_index also counts the bordering vertex dof
    end
end


#=
@testset "dss on 2×2 rectangular mesh (unstructured)" begin
    FT = Float64
    n1, n2 = 2, 2
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(0) .. Geometry.XPoint{FT}(4),
        Geometry.YPoint{FT}(0) .. Geometry.YPoint{FT}(4),
        x1periodic = false,
        x2periodic = false,
        x1boundary = (:west, :east),
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)

    quad = Quadratures.GLL{4}()
    points, weights = Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (4, 4, 2, 4)

    Nij = length(points)
    field = Fields.Field(IJFH{FT, Nij, n1 * n2}(ones(Nij, Nij, 1, n1 * n2)), space)
    field_values = Fields.field_values(field)
    Spaces.horizontal_dss!(field)

    @testset "dss should not modify interior degrees of freedom of any element" begin
        result = true
        for el in 1:(n1 * n2)
            slb = slab(field_values, 1, el)
            for i in 2:(Nij - 1), j in 2:(Nij - 1)
                if slb[i, j] ≠ 1
                    result = false
                end
            end
        end
        @test result
    end
    s1 = slab(field_values, 1, 1)
    s2 = slab(field_values, 1, 2)
    s3 = slab(field_values, 1, 3)
    s4 = slab(field_values, 1, 4)

    @testset "vertex common to all (4) elements" begin
        @test (s1[Nij, Nij] == s2[1, Nij] == s3[Nij, 1] == s4[1, 1])
    end

    @testset "vertices common to (2) elements" begin
        @test s1[Nij, 1] == s2[1, 1]
        @test s1[1, Nij] == s3[1, 1]
        @test s2[Nij, Nij] == s4[Nij, 1]
        @test s3[Nij, Nij] == s4[1, Nij]
    end

    @testset "boundary faces" begin
        for fc in 2:(Nij - 1)
            @test s1[1, fc] == 1 # element 1 face 1
            @test s1[fc, 1] == 1 # element 1 face 3
            @test s2[Nij, fc] == 1 # element 2 face 2
            @test s2[fc, 1] == 1 # element 2 face 3
            @test s3[1, fc] == 1 # element 3 face 1
            @test s3[fc, Nij] == 1 # element 3 face 4
            @test s4[Nij, fc] == 1 # element 4 face 2
            @test s4[fc, Nij] == 1 # element 4 face 4
        end
    end

    @testset "interior faces" begin
        for fc in 2:(Nij - 1)
            @test (s1[Nij, fc] == s2[1, fc] == 2) # (e1, f2) == (e2, f1) == 2
            @test (s1[fc, Nij] == s3[fc, 1] == 2) # (e1, f4) == (e3, f3) == 2
            @test (s2[fc, Nij] == s4[fc, 1] == 2) # (e2, f4) == (e4, f3) == 2
            @test (s3[Nij, fc] == s4[1, fc] == 2) # (e3, f2) == (e4, f1) == 2
        end
    end
end


@testset "dss on 2×2 rectangular mesh" begin
    FT = Float64
    n1, n2 = 2, 2
    Nij = 4
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(0) .. Geometry.XPoint{FT}(4),
        Geometry.YPoint{FT}(0) .. Geometry.YPoint{FT}(4),
        x1periodic = false,
        x2periodic = false,
        x1boundary = (:west, :east),
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)

    quad = Quadratures.GLL{Nij}()
    points, weights = Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (Nij, Nij, 2, n1 * n2)

    data = zeros(Nij, Nij, 3, n1 * n2)
    data[:, :, 1, :] .= 1:Nij
    data[:, :, 2, :] .= (1:Nij)'
    data[:, :, 3, :] .= reshape(1:(n1 * n2), 1, 1, :)
    field = Fields.Field(IJFH{Tuple{FT, FT, FT}, Nij, n1 * n2}(data), space)
    field_dss = Spaces.horizontal_dss!(copy(field))
    data_dss = parent(field_dss)

    @testset "slab 1" begin
        @test data_dss[1:(Nij - 1), 1:(Nij - 1), :, 1] ==
              data[1:(Nij - 1), 1:(Nij - 1), :, 1]
        @test data_dss[Nij, 1:(Nij - 1), :, 1] ==
              data[Nij, 1:(Nij - 1), :, 1] .+ data[1, 1:(Nij - 1), :, 2]
        @test data_dss[1:(Nij - 1), Nij, :, 1] ==
              data[1:(Nij - 1), Nij, :, 1] .+ data[1:(Nij - 1), 1, :, 3]
        @test data_dss[Nij, Nij, :, 1] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end

    @testset "slab 2" begin
        @test data_dss[2:Nij, 1:(Nij - 1), :, 2] ==
              data[2:Nij, 1:(Nij - 1), :, 2]
        @test data_dss[1, 1:(Nij - 1), :, 2] ==
              data[Nij, 1:(Nij - 1), :, 1] .+ data[1, 1:(Nij - 1), :, 2]
        @test data_dss[2:Nij, Nij, :, 2] ==
              data[2:Nij, Nij, :, 2] .+ data[2:Nij, 1, :, 4]
        @test data_dss[1, Nij, :, 2] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end

    @testset "slab 3" begin
        @test data_dss[1:(Nij - 1), 2:Nij, :, 3] ==
              data[1:(Nij - 1), 2:Nij, :, 3]
        @test data_dss[Nij, 2:Nij, :, 3] ==
              data[Nij, 2:Nij, :, 3] .+ data[1, 2:Nij, :, 4]
        @test data_dss[1:(Nij - 1), 1, :, 3] ==
              data[1:(Nij - 1), Nij, :, 1] .+ data[1:(Nij - 1), 1, :, 3]
        @test data_dss[Nij, 1, :, 3] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end

    @testset "slab 3" begin
        @test data_dss[2:Nij, 2:Nij, :, 4] == data[2:Nij, 2:Nij, :, 4]
        @test data_dss[1, 2:Nij, :, 4] ==
              data[Nij, 2:Nij, :, 3] .+ data[1, 2:Nij, :, 4]
        @test data_dss[2:Nij, 1, :, 4] ==
              data[2:Nij, Nij, :, 2] .+ data[2:Nij, 1, :, 4]
        @test data_dss[1, 1, :, 4] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end
end
=#
