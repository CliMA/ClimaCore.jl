#=
julia --project
using Revise; include(joinpath("test", "Operators", "finitedifference", "unit_column.jl"))
=#
using Test
using StaticArrays, IntervalSets, LinearAlgebra

using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: slab, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

device = ClimaComms.device()

@testset "Scalar Field FiniteDifferenceSpaces" begin
    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_names = (:left, :right),
        )
        @test eltype(domain) === Geometry.ZPoint{FT}

        mesh = Meshes.IntervalMesh(domain; nelems = 16)
        topology = Topologies.IntervalTopology(
            ClimaComms.SingletonCommsContext(device),
            mesh,
        )
        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        @test sum(ones(FT, center_space)) ≈ pi
        @test sum(ones(FT, face_space)) ≈ pi

        centers = getproperty(Fields.coordinate_field(center_space), :z)
        @test sum(sin.(centers)) ≈ FT(2.0) atol = 1e-2

        faces = getproperty(Fields.coordinate_field(face_space), :z)
        @test sum(sin.(faces)) ≈ FT(2.0) atol = 1e-2

        ∇ᶜ = Operators.GradientF2C()
        ∂sin = Geometry.WVector.(∇ᶜ.(sin.(faces)))
        @test ∂sin ≈ Geometry.WVector.(cos.(centers)) atol = 1e-2

        divᶜ = Operators.DivergenceF2C()
        ∂sin = divᶜ.(Geometry.WVector.(sin.(faces)))
        @test ∂sin ≈ cos.(centers) atol = 1e-2

        # Center -> Face operator
        # first order convergence at boundaries
        ∇ᶠ = Operators.GradientC2F(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(pi)),
        )
        ∂z = Geometry.WVector.(∇ᶠ.(centers))
        @test ∂z ≈ Geometry.WVector.(ones(FT, face_space)) rtol = 10 * eps(FT)

        ∇ᶠ = Operators.GradientC2F(
            left = Operators.SetValue(FT(1)),
            right = Operators.SetValue(FT(-1)),
        )
        ∂cos = Geometry.WVector.(∇ᶠ.(cos.(centers)))
        @test ∂cos ≈ Geometry.WVector.(.-sin.(faces)) atol = 1e-1

        ∇ᶠ = Operators.GradientC2F(
            left = Operators.SetGradient(Geometry.WVector(FT(0))),
            right = Operators.SetGradient(Geometry.WVector(FT(0))),
        )
        ∂cos = Geometry.WVector.(∇ᶠ.(cos.(centers)))
        @test ∂cos ≈ Geometry.WVector.(.-sin.(faces)) atol = 1e-2

        # test that broadcasting into incorrect field space throws an error
        empty_centers = zeros(FT, center_space)
        @test_throws Exception empty_centers .= ∇ᶠ.(cos.(centers))
    end
end


@testset "Scalar Field FiniteDifferenceSpaces - periodic" begin
    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(2pi);
            periodic = true,
        )
        @test eltype(domain) === Geometry.ZPoint{FT}

        mesh = Meshes.IntervalMesh(domain; nelems = 16)
        topology = Topologies.IntervalTopology(
            ClimaComms.ClimaComms.SingletonCommsContext(),
            mesh,
        )

        center_space = Spaces.CenterFiniteDifferenceSpace(topology)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        @test sum(ones(FT, center_space)) ≈ 2pi
        @test sum(ones(FT, face_space)) ≈ 2pi

        sinz_c = sin.(Fields.coordinate_field(center_space).z)
        cosz_c = cos.(Fields.coordinate_field(center_space).z)
        @test sum(sinz_c) ≈ FT(0.0) atol = 1e-2

        sinz_f = sin.(Fields.coordinate_field(face_space).z)
        cosz_f = cos.(Fields.coordinate_field(face_space).z)
        @test sum(sinz_f) ≈ FT(0.0) atol = 1e-2

        ∇ᶜ = Operators.GradientF2C()
        ∂sin = Geometry.WVector.(∇ᶜ.(sinz_f))
        @test ∂sin ≈ Geometry.WVector.(cosz_c) atol = 1e-2

        divᶜ = Operators.DivergenceF2C()
        ∂sin = divᶜ.(Geometry.WVector.(sinz_f))
        @test ∂sin ≈ cosz_c atol = 1e-2

        ∇ᶠ = Operators.GradientC2F()
        ∂cos = Geometry.WVector.(∇ᶠ.(cosz_c))
        @test ∂cos ≈ Geometry.WVector.(.-sinz_f) atol = 1e-1

        ∇ᶠ = Operators.GradientC2F()
        ∂cos = Geometry.WVector.(∇ᶠ.(cosz_c))
        @test ∂cos ≈ Geometry.WVector.(.-sinz_f) atol = 1e-2

        # test that broadcasting into incorrect field space throws an error
        empty_centers = zeros(FT, center_space)
        @test_throws Exception empty_centers .= ∇ᶠ.(cos.(centers))
    end
end

@testset "Test composed stencils" begin
    are_boundschecks_forced = Base.JLOptions().check_bounds == 1
    device = ClimaComms.device()
    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_names = (:left, :right),
        )
        @test eltype(domain) === Geometry.ZPoint{FT}

        mesh = Meshes.IntervalMesh(domain; nelems = 16)

        center_space = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        centers = getproperty(Fields.coordinate_field(center_space), :z)
        w = ones(FT, face_space)
        θ = sin.(centers)

        # 1) we set boundaries on the 2nd operator
        I = Operators.InterpolateC2F()
        ∂ = Operators.GradientF2C(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )

        ∂sin = Geometry.WVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.WVector.(cos.(centers)) atol = 1e-2

        # can't define Neumann conditions on GradientF2C
        ∂ = Operators.GradientF2C(
            left = Operators.SetGradient(FT(1)),
            right = Operators.SetGradient(FT(-1)),
        )

        if are_boundschecks_forced
            @test_throws Exception ∂.(w .* I.(θ))
        else
            @warn "Bounds check on BoundsError ∂.(w .* I.(θ)) not verified."
        end

        # 2) we set boundaries on the 1st operator
        I = Operators.InterpolateC2F(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )
        ∂ = Operators.GradientF2C()

        ∂sin = Geometry.WVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.WVector.(cos.(centers)) atol = 1e-2

        I = Operators.InterpolateC2F(
            left = Operators.SetGradient(Geometry.WVector(FT(1))),
            right = Operators.SetGradient(Geometry.WVector(FT(-1))),
        )
        ∂ = Operators.GradientF2C()

        ∂sin = Geometry.WVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.WVector.(cos.(centers)) atol = 1e-2

        # 3) we set boundaries on both: 2nd should take precedence
        I = Operators.InterpolateC2F(
            left = Operators.SetValue(FT(NaN)),
            right = Operators.SetValue(FT(NaN)),
        )
        ∂ = Operators.GradientF2C(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )

        ∂sin = Geometry.WVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.WVector.(cos.(centers)) atol = 1e-2

        # test that broadcasting into incorrect field space throws an error
        empty_faces = zeros(FT, face_space)
        @test_throws Exception empty_faces .= ∂.(w .* I.(θ))

        # 5) we set boundaries on neither
        I = Operators.InterpolateC2F()
        ∂ = Operators.GradientF2C()

        # TODO: should we throw something else?
        if are_boundschecks_forced
            @test_throws BoundsError ∂.(w .* I.(θ))
        else
            @warn "Bounds check on BoundsError ∂.(w .* I.(θ)) not verified."
        end
    end
end

@testset "Composite Field FiniteDifferenceSpaces" begin
    device = ClimaComms.device()
    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_names = (:left, :right),
        )

        @test eltype(domain) === Geometry.ZPoint{FT}
        mesh = Meshes.IntervalMesh(domain; nelems = 16)

        center_space = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

        FieldType = NamedTuple{(:a, :b), Tuple{FT, FT}}

        center_sum = sum(ones(FieldType, center_space))
        @test center_sum isa FieldType
        @test center_sum.a ≈ FT(pi)
        @test center_sum.b ≈ FT(pi)

        face_sum = sum(ones(FieldType, face_space))
        @test face_sum isa FieldType
        @test face_sum.a ≈ FT(pi)
        @test face_sum.b ≈ FT(pi)
    end
end

@testset "Biased interpolation" begin
    FT = Float64
    n_elems = 10
    device = ClimaComms.device()

    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(pi);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = n_elems)

    cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)

    zc = getproperty(Fields.coordinate_field(cs), :z)
    zf = getproperty(Fields.coordinate_field(fs), :z)

    field_vars() = (; y = FT(0))

    cfield = fill(field_vars(), cs)
    ffield = fill(field_vars(), fs)

    cy = cfield.y
    fy = ffield.y

    cyp = parent(cy).arrays[1]
    fyp = parent(fy).arrays[1]

    # C2F biased operators
    LBC2F = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(10))
    @. cy = cos(zc)
    @. fy = LBC2F(cy)
    fy_ref = [FT(10), [cyp[i] for i in 1:length(cyp)]...]
    @test all(fy_ref .== fyp)

    RBC2F = Operators.RightBiasedC2F(; top = Operators.SetValue(10))
    @. cy = cos(zc)
    @. fy = RBC2F(cy)
    fy_ref = [[cyp[i] for i in 1:length(cyp)]..., FT(10)]
    @test all(fy_ref .== fyp)

    # F2C biased operators
    LBF2C = Operators.LeftBiasedF2C(; bottom = Operators.SetValue(10))
    @. cy = cos(zc)
    @. cy = LBF2C(fy)
    cy_ref = [i == 1 ? FT(10) : fyp[i] for i in 1:length(cyp)]
    @test all(cy_ref .== cyp)

    RBF2C = Operators.RightBiasedF2C(; top = Operators.SetValue(10))
    @. cy = cos(zc)
    @. cy = RBF2C(fy)
    cy_ref = [i == length(cyp) ? FT(10) : fyp[i + 1] for i in 1:length(cyp)]
    @test all(cy_ref .== cyp)
end
