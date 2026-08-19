using Test
using StaticArrays, IntervalSets, LinearAlgebra
using ClimaCore
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
            left = Operators.SetGradient(Geometry.WVector(FT(0))),
            right = Operators.SetGradient(Geometry.WVector(FT(0))),
        )
        ∂cos = Geometry.WVector.(∇ᶠ.(cos.(centers)))
        @test ∂cos ≈ Geometry.WVector.(.-sin.(faces)) atol = 1e-2

        # Test that broadcasting into incorrect field space throws an error
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

        # Test that broadcasting into incorrect field space throws an error
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

        # Extrapolate is not accepted by GradientF2C
        @test_throws AssertionError Operators.GradientF2C(
            left = Operators.Extrapolate(),
            right = Operators.Extrapolate(),
        )

        # SetGradient prescribes the gradient at the boundary centers
        ∂ = Operators.GradientF2C(
            left = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
            right = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
        )
        ∂sin = Geometry.WVector.(∂.(w .* I.(θ)))
        ∂sin_arr = vec(Array(parent(∂sin)))
        @test ∂sin_arr[1] == 0
        @test ∂sin_arr[end] == 0
        @test maximum(
            abs.(
                ∂sin_arr[2:(end - 1)] .-
                cos.(vec(Array(parent(centers))))[2:(end - 1)]
            ),
        ) ≤ 1e-2

        # 2) we set boundaries on the 1st operator
        I = Operators.InterpolateC2F(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )
        ∂ = Operators.GradientF2C()

        ∂sin = Geometry.WVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.WVector.(cos.(centers)) atol = 1e-2

        # 3) we set boundaries on both: 2nd should take precedence
        I = Operators.InterpolateC2F(
            left = Operators.SetValue(FT(1)),
            right = Operators.SetValue(FT(1)),
        )
        ∂ = Operators.GradientF2C(
            left = Operators.SetValue(FT(0)),
            right = Operators.SetValue(FT(0)),
        )

        ∂sin = Geometry.WVector.(∂.(w .* I.(θ)))
        @test ∂sin ≈ Geometry.WVector.(cos.(centers)) atol = 1e-2

        # Test that broadcasting into incorrect field space throws an error
        empty_faces = zeros(FT, face_space)
        @test_throws Exception empty_faces .= ∂.(w .* I.(θ))
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

@testset "Biased interpolation [$FT]" for FT in (Float32, Float64)
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

    cyp = parent(cy)
    fyp = parent(fy)

    # C2F biased operators
    LBC2F = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(FT(10)))
    @. cy = cos(zc)
    @. fy = LBC2F(cy)
    fy_ref = ClimaComms.allowscalar(device) do
        [FT(10), [cyp[i] for i in 1:length(cyp)]...]
    end
    @test all(fy_ref .== parent(ClimaCore.to_cpu(fy)))

    RBC2F = Operators.RightBiasedC2F(; top = Operators.SetValue(FT(10)))
    @. cy = cos(zc)
    @. fy = RBC2F(cy)
    fy_ref = ClimaComms.allowscalar(device) do
        [[cyp[i] for i in 1:length(cyp)]..., FT(10)]
    end
    @test all(fy_ref .== parent(ClimaCore.to_cpu(fy)))

    # F2C biased operators
    LBF2C = Operators.LeftBiasedF2C(; bottom = Operators.SetValue(FT(10)))
    @. cy = cos(zc)
    @. cy = LBF2C(fy)
    cy_ref = ClimaComms.allowscalar(device) do
        [i == 1 ? FT(10) : fyp[i] for i in 1:length(cyp)]
    end
    @test all(cy_ref .== parent(ClimaCore.to_cpu(cy)))

    RBF2C = Operators.RightBiasedF2C(; top = Operators.SetValue(FT(10)))
    @. cy = cos(zc)
    @. cy = RBF2C(fy)
    cy_ref = ClimaComms.allowscalar(device) do
        [i == length(cyp) ? FT(10) : fyp[i + 1] for i in 1:length(cyp)]
    end
    @test all(cy_ref .== parent(ClimaCore.to_cpu(cy)))
end

# https://github.com/CliMA/ClimaCore.jl/issues/994
# TODO: make this test more low-level / granular (test `getidx`).
@testset "Spatially varying BC with Grad [$FT]" for FT in (Float32, Float64)
    zmin = FT(1.0)
    zmax = FT(2.0)
    xlim = FT.((0.0, 10.0))
    ylim = FT.((0.0, 1.0))
    zlim = FT.((zmin, zmax))
    nelements = (1, 1, 5)
    npolynomial = 3
    domain_x = Domains.IntervalDomain(
        Geometry.XPoint(xlim[1]),
        Geometry.XPoint(xlim[2]);
        periodic = true,
    )
    domain_y = Domains.IntervalDomain(
        Geometry.YPoint(ylim[1]),
        Geometry.YPoint(ylim[2]);
        periodic = true,
    )
    plane = Domains.RectangleDomain(domain_x, domain_y)
    context = ClimaComms.context()
    mesh = Meshes.RectilinearMesh(plane, nelements[1], nelements[2])
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = ClimaCore.Quadratures.GLL{npolynomial + 1}()
    horzspace = Spaces.SpectralElementSpace2D(grid_topology, quad)

    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zlim[1]),
        Geometry.ZPoint(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = nelements[3])
    vert_center_space =
        Spaces.CenterFiniteDifferenceSpace(ClimaComms.device(context), vertmesh)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)

    surface_field = Fields.zeros(horzspace)

    ψ = Fields.ones(hv_center_space)
    value = surface_field # same type/instance of underlying space as horizontal space of \psi

    gradc2f_no_bc = Operators.GradientC2F()
    divf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.WVector.(value)),
        bottom = Operators.SetValue(Geometry.WVector(FT(0.0))),
    )
    @. divf2c(gradc2f_no_bc(ψ)) # runs

    # A boundary condition whose value is a Field on the horizontal space,
    # rather than a constant.
    gradc2f = Operators.GradientC2F(;
        top = Operators.SetGradient(Geometry.WVector.(value)),
        bottom = Operators.SetGradient(Geometry.WVector(FT(0.0))),
    )
    ∇ψ = gradc2f.(ψ)
    @test all(isfinite, parent(ClimaCore.to_cpu(∇ψ)))
end

# The `SetValue` boundary conditions of `GradientC2F`, `DivergenceC2F`,
# `CurlC2F` and `UpwindBiasedProductC2F`, and the `AdvectionC2C`,
# `AdvectionF2F` and `FluxCorrectionC2C`/`FluxCorrectionF2F` operators, were
# removed in #2521. These tests pin the recommended replacements against the
# stencils that they replaced, on a stretched mesh (so that a dropped metric
# term would show up).
@testset "Boundary values and advection built from the primitive operators" begin
    FT = Float64
    n = 8
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(1.0);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(
        domain,
        Meshes.ExponentialStretching(FT(0.3));
        nelems = n,
    )
    cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
    fs = Spaces.face_space(cs)

    ᶜlg = Fields.local_geometry_field(cs)
    ᶠlg = Fields.local_geometry_field(fs)
    θᶜ = sin.(3 .* Fields.coordinate_field(cs).z)
    θᶠ = sin.(3 .* Fields.coordinate_field(fs).z)
    wᶜ = Geometry.WVector.(1 .+ sin.(Fields.coordinate_field(cs).z))
    wᶠ = Geometry.WVector.(1 .+ sin.(Fields.coordinate_field(fs).z))

    # the reference values below are built with scalar indexing, so the data
    # they are built from has to be moved to the CPU first
    cpu_parent(field) = Array(parent(field))

    # the quantities the reference stencils below are written in terms of
    tᶜ = cpu_parent(θᶜ)[:]
    tᶠ = cpu_parent(θᶠ)[:]
    w³ᶜ = cpu_parent(Geometry.contravariant3.(wᶜ, ᶜlg))[:]
    w³ᶠ = cpu_parent(Geometry.contravariant3.(wᶠ, ᶠlg))[:]
    Jᶜ = cpu_parent(ᶜlg.J)[:]
    Jᶠ = cpu_parent(ᶠlg.J)[:]
    θ₀ = FT(0.5) # boundary value
    interpf2c = Operators.InterpolateF2C()

    θ_bot = Fields.level(θᶜ, 1)
    θ_top = Fields.level(θᶜ, n)

    @testset "GradientC2F with a prescribed boundary value" begin
        # G(x)[1/2] = 2 (x[1] - x₀), G(x)[n+1/2] = 2 (x₀ - x[n])
        ref = [
            i == 1 ? 2 * (tᶜ[1] - θ₀) :
            i == n + 1 ? 2 * (θ₀ - tᶜ[n]) : tᶜ[i] - tᶜ[i - 1] for i in 1:(n + 1)
        ]
        gradc2f = Operators.GradientC2F(
            bottom = Operators.SetGradient(
                Geometry.Covariant3Vector.(2 .* (θ_bot .- θ₀)),
            ),
            top = Operators.SetGradient(
                Geometry.Covariant3Vector.(2 .* (θ₀ .- θ_top)),
            ),
        )
        @test cpu_parent(gradc2f.(θᶜ))[:] ≈ ref
    end

    @testset "DivergenceC2F with a zero boundary value" begin
        # D(v)[1/2] = (Jv³[1] - 0) 2 / J[1/2]
        ref = [
            i == 1 ? (Jᶜ[1] * w³ᶜ[1]) * 2 / Jᶠ[1] :
            i == n + 1 ? -(Jᶜ[n] * w³ᶜ[n]) * 2 / Jᶠ[n + 1] :
            (Jᶜ[i] * w³ᶜ[i] - Jᶜ[i - 1] * w³ᶜ[i - 1]) / Jᶠ[i] for i in 1:(n + 1)
        ]
        # `LeftBiasedF2C(x)[i] = x[i-half]`, so its first level is the bottom
        # face; `RightBiasedF2C(x)[i] = x[i+half]`, so its last level is the top.
        # These are applied to `J` rather than to the local geometry itself, so
        # that the stencil only ever operates on scalars.
        J_bot_face = Fields.level(Operators.LeftBiasedF2C().(ᶠlg.J), 1)
        J_top_face = Fields.level(Operators.RightBiasedF2C().(ᶠlg.J), n)
        set_bcs = Operators.SetBoundaryOperator(
            bottom = Operators.SetValue(
                Geometry.Jcontravariant3.(
                    Fields.level(wᶜ, 1),
                    Fields.level(ᶜlg, 1),
                ) .* (2 .* inv.(J_bot_face)),
            ),
            top = Operators.SetValue(
                Geometry.Jcontravariant3.(
                    Fields.level(wᶜ, n),
                    Fields.level(ᶜlg, n),
                ) .* (-2 .* inv.(J_top_face)),
            ),
        )
        divc2f = Operators.DivergenceC2F()
        @test cpu_parent(@. set_bcs(divc2f(wᶜ)))[:] ≈ ref
    end

    @testset "UpwindBiasedProductC2F with a prescribed boundary value" begin
        # U(v,x)[1/2] uses x₀ on the outside of the boundary
        ref = [
            i == 1 ?
            Operators.upwind_biased_product(w³ᶠ[1], θ₀, tᶜ[1]) :
            i == n + 1 ?
            Operators.upwind_biased_product(w³ᶠ[n + 1], tᶜ[n], θ₀) :
            Operators.upwind_biased_product(w³ᶠ[i], tᶜ[i - 1], tᶜ[i])
            for i in 1:(n + 1)
        ]
        v_bot = w³ᶠ[1]
        v_top = w³ᶠ[n + 1]
        set_bcs = Operators.SetBoundaryOperator(
            bottom = Operators.SetValue(
                Geometry.Contravariant3Vector(
                    Operators.upwind_biased_product(v_bot, θ₀, tᶜ[1]),
                ),
            ),
            top = Operators.SetValue(
                Geometry.Contravariant3Vector(
                    Operators.upwind_biased_product(v_top, tᶜ[n], θ₀),
                ),
            ),
        )
        upwind = Operators.UpwindBiasedProductC2F()
        @test cpu_parent(@. set_bcs(upwind(wᶠ, θᶜ)))[:] ≈ ref
    end

    @testset "Centered advection of a center field" begin
        # A(v,θ)[i] = (v³[i+1/2] ∂θ⁺ + v³[i-1/2] ∂θ⁻) / 2, with the boundary
        # difference taken over half a cell
        ref = map(1:n) do i
            ∂⁺ = i == n ? 2 * (θ₀ - tᶜ[n]) : tᶜ[i + 1] - tᶜ[i]
            ∂⁻ = i == 1 ? 2 * (tᶜ[1] - θ₀) : tᶜ[i] - tᶜ[i - 1]
            (w³ᶠ[i + 1] * ∂⁺ + w³ᶠ[i] * ∂⁻) / 2
        end
        gradc2f = Operators.GradientC2F(
            bottom = Operators.SetGradient(
                Geometry.Covariant3Vector.(2 .* (θ_bot .- θ₀)),
            ),
            top = Operators.SetGradient(
                Geometry.Covariant3Vector.(2 .* (θ₀ .- θ_top)),
            ),
        )
        new = @. interpf2c(
            Geometry.dot(Geometry.Contravariant3Vector(wᶠ), gradc2f(θᶜ)),
        )
        @test cpu_parent(new)[:] ≈ ref
    end

    @testset "Centered advection of a face field" begin
        # A(v,θ)[i] = v³[i] (θ[i+1] - θ[i-1]) / 2, interior only
        ref = [w³ᶠ[i] * (tᶠ[i + 1] - tᶠ[i - 1]) / 2 for i in 2:n]
        gradf2c = Operators.GradientF2C()
        interpc2f = Operators.InterpolateC2F()
        new = @. Geometry.dot(
            Geometry.Contravariant3Vector(wᶠ),
            interpc2f(gradf2c(θᶠ)),
        )
        @test cpu_parent(new)[2:n] ≈ ref
    end

    @testset "Diffusive flux correction of a center field" begin
        # A(v,θ)[i] = |v³[i+1/2]| ∂θ⁺ - |v³[i-1/2]| ∂θ⁻, where `Extrapolate`
        # drops the term outside of the boundary (zero flux through the face)
        ref = map(1:n) do i
            fc⁺ = i == n ? zero(FT) : abs(w³ᶠ[i + 1]) * (tᶜ[i + 1] - tᶜ[i])
            fc⁻ = i == 1 ? zero(FT) : abs(w³ᶠ[i]) * (tᶜ[i] - tᶜ[i - 1])
            fc⁺ - fc⁻
        end
        zero_gradient =
            Operators.SetGradient(Geometry.Covariant3Vector(zero(FT)))
        gradc2f = Operators.GradientC2F(
            bottom = zero_gradient,
            top = zero_gradient,
        )
        gradf2c = Operators.GradientF2C()
        flux = @. Geometry.dot(
            Geometry.Contravariant3Vector(
                abs(Geometry.contravariant3(wᶠ, ᶠlg)),
            ),
            gradc2f(θᶜ),
        )
        # `GradientF2C` returns a `Covariant3Vector`, whose component is the
        # difference of the fluxes across the cell
        @test cpu_parent(gradf2c.(flux))[:] ≈ ref
    end
end
