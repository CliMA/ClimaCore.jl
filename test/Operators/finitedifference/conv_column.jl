using Test
using StaticArrays, IntervalSets, LinearAlgebra

using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    ClimaCore, slab, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

device = ClimaComms.device()

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities: convergence_rate

# Visualization artifacts: the advection testsets below record the flux and its
# divergence `adv_wc` at every resolution, so that the convergence rates can be
# inspected alongside the profiles they were computed from.
#
# Plots and ClimaCorePlots are in the .buildkite environment but not in the one
# `Pkg.test` builds, and here they only write diagnostic PNGs — no assertion
# depends on them. Make them optional, as test/tabulated_tests.jl does for
# PrettyTables, so the test still runs in a plotting-free environment.
ENV["GKSwstype"] = "nul"
const HAVE_PLOTS = try
    import Plots
    import ClimaCorePlots
    Plots.GRBackend()
    true
catch
    false
end

const stretch_names = ("uniform", "stretched")
const plot_dir = joinpath(
    @__DIR__,
    "output",
    "conv_column",
    device isa ClimaComms.CUDADevice ? "GPU" : "CPU",
)

"""
    plot_flux_and_adv(name, results)

Write `output/conv_column/<device>/<name>.png`, showing the advecting velocity
`w` and the advected field `c` that go into the flux, the face-valued flux
itself, and the center-valued advective tendency `adv_wc`, at every resolution.
`results` is a vector of `(n, w, c, flux, adv_wc, adv_exact)` tuples, one per
element count, and the exact tendency of the finest resolution is drawn on top
of `adv_wc`. A no-op when the plotting packages are unavailable.
"""
function plot_flux_and_adv(name, results)
    HAVE_PLOTS || return nothing
    mkpath(plot_dir)
    # The upwind operators return a Contravariant3Vector, whose component scales
    # like 1/Δz; the physical WVector component is what is comparable across
    # resolutions. Plotting also requires scalar indexing, so the fields are
    # moved to the CPU.
    function plottable(field)
        field = ClimaCore.to_cpu(field)
        eltype(field) <: Geometry.AxisVector || return field
        return Geometry.WVector.(field, Fields.local_geometry_field(axes(field)))
    end

    w_plt, c_plt, flux_plt, adv_plt = ntuple(_ -> Plots.plot(), 4)
    for (n, w, c, flux, adv_wc, _) in results
        label = "n = $n"
        Plots.plot!(w_plt, plottable(w); label)
        Plots.plot!(c_plt, plottable(c); label)
        Plots.plot!(flux_plt, plottable(flux); label)
        Plots.plot!(adv_plt, plottable(adv_wc); label)
    end
    Plots.plot!(
        adv_plt,
        plottable(last(results)[6]);
        label = "exact",
        color = :black,
        linestyle = :dash,
    )
    # Set the guides after the series, so that they are not overwritten by the
    # defaults of the ClimaCorePlots recipe.
    Plots.plot!(w_plt; title = "w", xguide = "w (WVector)", yguide = "z faces")
    Plots.plot!(c_plt; title = "c", xguide = "c", yguide = "z centers")
    Plots.plot!(
        flux_plt;
        title = "flux",
        xguide = "flux (WVector)",
        yguide = "z faces",
    )
    Plots.plot!(adv_plt; title = "adv_wc", xguide = "adv_wc", yguide = "z centers")
    plts = (w_plt, c_plt, flux_plt, adv_plt)
    Plots.plot!.(plts, legend = :outerbottom, legendcolumns = 2)
    plt = Plots.plot(
        plts...;
        layout = (2, 2),
        size = (1000, 1200),
        plot_title = name,
        left_margin = 8Plots.PlotMeasures.mm,
    )
    Plots.png(plt, joinpath(plot_dir, "$name.png"))
    return plt
end


@testset "Face -> Center interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    device = ClimaComms.device()
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err = zeros(FT, length(n_elems_seq))
        werr = zeros(FT, length(n_elems_seq))
        Δh = zeros(FT, length(n_elems_seq))

        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a) .. Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_names = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            cent_field_exact = zeros(FT, cs)
            face_field = zeros(FT, fs)

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            face_field .= sin.(3π .* faces.z)
            face_J = Fields.local_geometry_field(fs).J

            cent_field_exact .= sin.(3π .* centers.z)
            operator = Operators.InterpolateF2C()
            woperator = Operators.WeightedInterpolateF2C()
            cent_field = operator.(face_field)
            wcent_field = woperator.(face_J, face_field)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end
            err[k] = norm(cent_field .- cent_field_exact)
            werr[k] = norm(wcent_field .- cent_field_exact)
        end

        conv = convergence_rate(err, Δh)
        wconv = convergence_rate(werr, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test all(1.8 .<= conv .<= 2)
        @test all(1.8 .<= wconv .<= 2)
    end
end

@testset "Center -> Face interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    device = ClimaComms.device()
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
        werr = zeros(FT, length(n_elems_seq))
        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a) .. Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_names = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            face_field_exact = zeros(FT, fs)
            cent_field = zeros(FT, cs)

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            cent_field .= sin.(3π .* centers.z)
            cent_J = Fields.local_geometry_field(cs).J
            face_field_exact .= sin.(3π .* faces.z)

            operator = Operators.InterpolateC2F(
                left = Operators.SetValue(0.0),
                right = Operators.SetValue(0.0),
            )
            woperator = Operators.WeightedInterpolateC2F(
                left = Operators.SetValue(0.0),
                right = Operators.SetValue(0.0),
            )
            face_field = operator.(cent_field)
            wface_field = woperator.(cent_J, cent_field)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end
            err[k] = norm(face_field .- face_field_exact)
            werr[k] = norm(wface_field .- face_field_exact)
        end
        conv = convergence_rate(err, Δh)
        wconv = convergence_rate(werr, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test all(1.8 .<= conv .<= 2)
        @test all(1.8 .<= wconv .<= 2)
    end
end

@testset "∂ Center -> Face interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    device = ClimaComms.device()
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a) .. Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_names = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            face_field_exact = Geometry.Covariant3Vector.(zeros(FT, fs))
            cent_field = zeros(FT, cs)
            face_field = Geometry.Covariant3Vector.(zeros(FT, fs))

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            cent_field .= sin.(3π .* centers.z)
            face_field_exact .=
                Geometry.CovariantVector.(
                    Geometry.WVector.(3π .* cos.(3π .* faces.z)),
                )

            operator = Operators.GradientC2F(
                left = Operators.SetGradient(Geometry.WVector(3π)),
                right = Operators.SetGradient(Geometry.WVector(-3π)),
            )

            face_field .= operator.(cent_field)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end
            err[k] = norm(face_field .- face_field_exact)
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
        @test conv[1] ≈ 2 atol = 0.1
        @test conv[2] ≈ 2 atol = 0.1
        @test conv[3] ≈ 2 atol = 0.1
        @test conv[1] ≤ conv[2] ≤ conv[3]
    end
end

@testset "∂ Face -> Center interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    device = ClimaComms.device()
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(0.5))
    for (i, stretch_fn) in enumerate(stretch_fns)
        err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
        for (k, n) in enumerate(n_elems_seq)
            interval = Geometry.ZPoint(a) .. Geometry.ZPoint(b)
            domain = Domains.IntervalDomain(
                interval;
                boundary_names = (:left, :right),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            cent_field_exact = Geometry.Covariant3Vector.(zeros(FT, cs))
            cent_field = Geometry.Covariant3Vector.(zeros(FT, cs))
            face_field = zeros(FT, fs)

            centers = Fields.coordinate_field(cs)
            faces = Fields.coordinate_field(fs)

            face_field .= sin.(3π .* faces.z)
            cent_field_exact .=
                Geometry.CovariantVector.(
                    Geometry.WVector.(3π .* cos.(3π .* centers.z)),
                )

            operator = Operators.GradientF2C()

            cent_field .= operator.(face_field)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end
            err[k] = norm(cent_field .- cent_field_exact)
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
        @test conv[1] ≈ 2 atol = 0.1
        @test conv[2] ≈ 2 atol = 0.1
        @test conv[3] ≈ 2 atol = 0.1
        @test conv[1] ≤ conv[2] ≤ conv[3]
    end
end

@testset "∂ Center -> Face and ∂ Face-> Center (uniform)" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_grad_sin_c = zeros(FT, length(n_elems_seq))
    err_div_sin_c = zeros(FT, length(n_elems_seq))
    err_grad_z_f = zeros(FT, length(n_elems_seq))
    err_grad_cos_f2 = zeros(FT, length(n_elems_seq))
    err_div_cos_f = zeros(FT, length(n_elems_seq))
    err_curl_sin_f = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(pi);
            boundary_names = (:left, :right),
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        faces = getproperty(Fields.coordinate_field(fs), :z)

        # Face -> Center operators:
        # GradientF2C
        # f(z) = sin(z)
        ∇ᶜ = Operators.GradientF2C()
        gradsinᶜ = Geometry.WVector.(∇ᶜ.(sin.(faces)))

        # DivergenceF2C
        # f(z) = sin(z)
        divᶜ = Operators.DivergenceF2C()
        divsinᶜ = divᶜ.(Geometry.WVector.(sin.(faces)))

        # Center -> Face operators:
        # GradientC2F, SetGradient
        # f(z) = z
        ∇ᶠ⁰ = Operators.GradientC2F(
            left = Operators.SetGradient(Geometry.WVector(one(FT))),
            right = Operators.SetGradient(Geometry.WVector(one(FT))),
        )
        ∂zᶠ = Geometry.WVector.(∇ᶠ⁰.(centers))

        # GradientC2F, SetGradient
        # f(z) = cos(z)
        ∇ᶠ² = Operators.GradientC2F(
            left = Operators.SetGradient(Geometry.WVector(FT(0))),
            right = Operators.SetGradient(Geometry.WVector(FT(0))),
        )
        gradcosᶠ² = Geometry.WVector.(∇ᶠ².(cos.(centers)))

        # DivergenceC2F, SetDivergence
        # f(z) = cos(z)
        divᶠ¹ = Operators.DivergenceC2F(
            left = Operators.SetDivergence(FT(0)),
            right = Operators.SetDivergence(FT(0)),
        )
        divcosᶠ = divᶠ¹.(Geometry.WVector.(cos.(centers)))

        curlᶠ = Operators.CurlC2F(
            left = Operators.SetCurl(
                Geometry.Contravariant12Vector(zero(FT), one(FT)),
            ),
            right = Operators.SetCurl(
                Geometry.Contravariant12Vector(zero(FT), -one(FT)),
            ),
        )
        curlsinᶠ = curlᶠ.(Geometry.Covariant12Vector.(sin.(centers), zero(FT)))


        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end
        # Errors
        err_grad_sin_c[k] = norm(gradsinᶜ .- Geometry.WVector.(cos.(centers)))
        err_div_sin_c[k] = norm(divsinᶜ .- cos.(centers))
        err_grad_z_f[k] = norm(∂zᶠ .- Geometry.WVector.(ones(FT, fs)))
        err_grad_cos_f2[k] = norm(gradcosᶠ² .- Geometry.WVector.(.-sin.(faces)))
        err_div_cos_f[k] = norm(
            divcosᶠ .- (Geometry.WVector.(.-sin.(faces))).components.data.:1,
        )
        err_curl_sin_f[k] =
            norm(curlsinᶠ.components.data.:2 .- cos.(faces))
    end

    # GradientF2C conv, with f(z) = sin(z)
    conv_grad_sin_c = convergence_rate(err_grad_sin_c, Δh)
    # DivergenceF2C conv, with f(z) = sin(z)
    conv_div_sin_c = convergence_rate(err_div_sin_c, Δh)
    # GradientC2F conv, with f(z) = z, SetGradient
    conv_grad_z = convergence_rate(err_grad_z_f, Δh)
    # GradientC2F conv, with f(z) = cos(z), SetGradient
    conv_grad_cos_f2 = convergence_rate(err_grad_cos_f2, Δh)
    # DivergenceC2F conv, with f(z) = cos(z), SetDivergence
    conv_div_cos_f = convergence_rate(err_div_cos_f, Δh)
    # CurlC2F with f(z) = sin(z), SetCurl
    conv_curl_sin_f = convergence_rate(err_curl_sin_f, Δh)

    # GradientF2C conv, with f(z) = sin(z)
    @test err_grad_sin_c[3] ≤ err_grad_sin_c[2] ≤ err_grad_sin_c[1] ≤ 0.1
    @test conv_grad_sin_c[1] ≈ 2 atol = 0.1
    @test conv_grad_sin_c[2] ≈ 2 atol = 0.1
    @test conv_grad_sin_c[3] ≈ 2 atol = 0.1
    @test conv_grad_sin_c[1] ≤ conv_grad_sin_c[2] ≤ conv_grad_sin_c[3]

    # DivergenceF2C conv, with f(z) = sin(z)
    @test err_div_sin_c[3] ≤ err_div_sin_c[2] ≤ err_div_sin_c[1] ≤ 0.1
    @test conv_div_sin_c[1] ≈ 2 atol = 0.1
    @test conv_div_sin_c[2] ≈ 2 atol = 0.1
    @test conv_div_sin_c[3] ≈ 2 atol = 0.1
    @test conv_div_sin_c[1] ≤ conv_div_sin_c[2] ≤ conv_div_sin_c[3]

    # GradientC2F conv, with f(z) = z, SetGradient
    @test norm(err_grad_z_f) ≤ 200 * eps(FT)
    # Convergence rate for this case is noisy because error very small

    # GradientC2F conv, with f(z) = cos(z), SetGradient
    @test err_grad_cos_f2[3] ≤ err_grad_cos_f2[2] ≤ err_grad_cos_f2[1] ≤ 0.1
    @test conv_grad_cos_f2[1] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[2] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[3] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[1] ≤ conv_grad_cos_f2[2] ≤ conv_grad_cos_f2[3]

    # DivergenceC2F conv, with f(z) = cos(z), SetDivergence
    @test err_div_cos_f[3] ≤ err_div_cos_f[2] ≤ err_div_cos_f[1] ≤ 0.1
    @test conv_div_cos_f[1] ≈ 2 atol = 0.1
    @test conv_div_cos_f[2] ≈ 2 atol = 0.1
    @test conv_div_cos_f[3] ≈ 2 atol = 0.1
    @test conv_div_cos_f[1] ≤ conv_div_cos_f[2] ≤ conv_div_cos_f[3]

    # CurlC2F with f(z) = sin(z), SetCurl
    @test err_curl_sin_f[3] ≤ err_curl_sin_f[2] ≤ err_curl_sin_f[1] ≤ 0.1
    @test conv_curl_sin_f[1] ≈ 2 atol = 0.1
    @test conv_curl_sin_f[2] ≈ 2 atol = 0.1
    @test conv_curl_sin_f[3] ≈ 2 atol = 0.1
    @test conv_curl_sin_f[1] ≤ conv_curl_sin_f[2] ≤ conv_curl_sin_f[3]
end

@testset "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on (uniform) periodic mesh, constant w" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()
    results = []

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)

        # Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F()
        third_order_fluxsinᶠ = third_order_fluxᶠ.(w, c)

        divf2c = Operators.DivergenceF2C()
        adv_wc = divf2c.(third_order_fluxsinᶠ)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
        push!(results, (n, w, c, third_order_fluxsinᶠ, adv_wc, cos.(centers)))
    end

    plot_flux_and_adv("upwind3_periodic_constant_w", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 5e-4
    @test conv_adv_wc[1] ≈ 3 atol = 0.1
    @test conv_adv_wc[2] ≈ 3 atol = 0.1
    @test conv_adv_wc[3] ≈ 3 atol = 0.1
    @test conv_adv_wc[1] ≤ conv_adv_wc[2] ≤ conv_adv_wc[2]
end

@testset "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on (uniform) periodic mesh, varying sign w" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()
    results = []

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        faces = getproperty(Fields.coordinate_field(fs), :z)

        # Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # w = cos(z), vertical velocity field defined at the faces
        w = Geometry.WVector.(cos.(faces))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F()
        third_order_fluxsinᶠ = third_order_fluxᶠ.(w, c)

        divf2c = Operators.DivergenceF2C()
        adv_wc = divf2c.(third_order_fluxsinᶠ)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end

        # Error
        adv_exact = cos.(centers) .^ 2 .- sin.(centers) .^ 2
        err_adv_wc[k] = norm(adv_wc .- adv_exact)
        push!(results, (n, w, c, third_order_fluxsinᶠ, adv_wc, adv_exact))
    end

    plot_flux_and_adv("upwind3_periodic_varying_w", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = cos(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 4e-3
    @test conv_adv_wc[1] ≈ 2 atol = 0.2
    @test conv_adv_wc[2] ≈ 2 atol = 0.1
    @test conv_adv_wc[3] ≈ 2 atol = 0.1
end

@testset "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on (uniform and stretched) non-periodic mesh, with FirstOrderOneSided + DivergenceF2C SetValue BCs, constant w" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(1.0))
    device = ClimaComms.device()

    for (i, stretch_fn) in enumerate(stretch_fns)
        err_adv_wc = zeros(FT, length(n_elems_seq))
        Δh = zeros(FT, length(n_elems_seq))
        results = []
        for (k, n) in enumerate(n_elems_seq)
            domain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(-pi),
                Geometry.ZPoint{FT}(pi);
                boundary_names = (:bottom, :top),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            centers = getproperty(Fields.coordinate_field(cs), :z)

            # Upwind3rdOrderBiasedProductC2F Center -> Face operator
            # Unitary, constant advective velocity
            w = Geometry.WVector.(ones(fs))
            # c = sin(z), scalar field defined at the centers
            Δz = FT(2pi / n)
            c = (cos.(centers .- Δz / 2) .- cos.(centers .+ Δz / 2)) ./ Δz
            s = sin.(centers)

            third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
                bottom = Operators.FirstOrderOneSided(),
                top = Operators.FirstOrderOneSided(),
            )

            divf2c = Operators.DivergenceF2C(
                bottom = Operators.SetValue(
                    Geometry.Contravariant3Vector(FT(0.0)),
                ),
                top = Operators.SetValue(
                    Geometry.Contravariant3Vector(FT(0.0)),
                ),
            )

            flux = third_order_fluxᶠ.(w, c)
            adv_wc = divf2c.(flux)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end

            # Error
            err_adv_wc[k] = norm(adv_wc .- cos.(centers))
            push!(results, (n, w, c, flux, adv_wc, cos.(centers)))
        end

        plot_flux_and_adv("upwind3_nonperiodic_constant_w_$(stretch_names[i])", results)

        # Check convergence rate
        conv_adv_wc = convergence_rate(err_adv_wc, Δh)
        # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = 1
        # The L2 error is dominated by the zeroth-order one-sided boundary
        # reconstruction, so the uniform-mesh rate is ~0.5; on the stretched
        # mesh the rate is ragged, so the measured values are asserted instead.
        @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 0.2006
        if stretch_fn isa Meshes.Uniform
            @test conv_adv_wc[1] ≈ 0.5 atol = 0.05
            @test conv_adv_wc[2] ≈ 0.5 atol = 0.05
            @test conv_adv_wc[3] ≈ 0.5 atol = 0.05
        else
            @test conv_adv_wc[1] ≈ 0.68 atol = 0.1
            @test conv_adv_wc[2] ≈ 0.17 atol = 0.1
            @test conv_adv_wc[3] ≈ 1.13 atol = 0.1
        end
    end
end

@testset "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on (uniform and stretched) non-periodic mesh, with ThirdOrderOneSided + DivergenceF2C SetValue BCs, varying sign w" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(1.0))
    device = ClimaComms.device()

    for (i, stretch_fn) in enumerate(stretch_fns)
        err_adv_wc = zeros(FT, length(n_elems_seq))
        Δh = zeros(FT, length(n_elems_seq))
        results = []
        for (k, n) in enumerate(n_elems_seq)
            domain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(-pi),
                Geometry.ZPoint{FT}(pi);
                boundary_names = (:bottom, :top),
            )
            mesh = Meshes.IntervalMesh(domain; nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            centers = getproperty(Fields.coordinate_field(cs), :z)
            faces = getproperty(Fields.coordinate_field(fs), :z)

            # Upwind3rdOrderBiasedProductC2F Center -> Face operator
            # w = cos(z), vertical velocity field defined at the faces
            w = Geometry.WVector.(cos.(faces))
            # c = sin(z), scalar field defined at the centers
            c = sin.(centers)#.^2 .+ 1

            third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
                bottom = Operators.ThirdOrderOneSided(),
                top = Operators.ThirdOrderOneSided(),
            )

            divf2c = Operators.DivergenceF2C(
                bottom = Operators.SetValue(Geometry.WVector(FT(0.0))),
                top = Operators.SetValue(Geometry.WVector(FT(0.0))),
            )
            flux = third_order_fluxᶠ.(w, c)
            adv_wc = divf2c.(flux)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end
            # Errors
            adv_exact = cos.(centers) .^ 2 .- sin.(centers) .^ 2
            err_adv_wc[k] = norm(adv_wc .- adv_exact)
            push!(results, (n, w, c, flux, adv_wc, adv_exact))

        end

        plot_flux_and_adv("upwind3_nonperiodic_varying_w_$(stretch_names[i])", results)

        # Check convergence rate
        conv_adv_wc = convergence_rate(err_adv_wc, Δh)
        # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = cos(z)
        @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 2e-1
        # the coarsest refinement step measures ~2.12
        @test conv_adv_wc[1] ≈ 2 atol = 0.15
        @test conv_adv_wc[2] ≈ 2 atol = 0.05
        @test conv_adv_wc[3] ≈ 2 atol = 0.05
    end
end

@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on (uniform) periodic mesh" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()
    results = []

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        first_order_fluxᶠ = Operators.UpwindBiasedProductC2F()
        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F()
        first_order_fluxsinᶠ = first_order_fluxᶠ.(w, c)
        third_order_fluxsinᶠ = third_order_fluxᶠ.(w, c)

        divf2c = Operators.DivergenceF2C()
        corrected_antidiff_flux =
            @. divf2c(C * (third_order_fluxsinᶠ - first_order_fluxsinᶠ))
        adv_wc = @. divf2c.(first_order_fluxsinᶠ) + corrected_antidiff_flux
        # The total flux whose divergence is adv_wc
        flux = @. first_order_fluxsinᶠ +
                  C * (third_order_fluxsinᶠ - first_order_fluxsinᶠ)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
        push!(results, (n, w, c, flux, adv_wc, cos.(centers)))
    end

    plot_flux_and_adv("fct_periodic", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 5e-4
    @test conv_adv_wc[1] ≈ 3 atol = 0.1
    @test conv_adv_wc[2] ≈ 3 atol = 0.1
    @test conv_adv_wc[3] ≈ 3 atol = 0.1
    @test conv_adv_wc[1] ≤ conv_adv_wc[2] ≤ conv_adv_wc[2]
end

@testset "Lin et al. (1994) van Leer class limiter (Mono5)" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8, 9, 10)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()
    results = []

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        SLMethod = Operators.LinVanLeerC2F(
            constraint = Operators.MonotoneLocalExtrema(),
        )

        divf2c = Operators.DivergenceF2C()
        flux = SLMethod.(w, c, FT(0))
        adv_wc = divf2c.(flux)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
        push!(results, (n, w, c, flux, adv_wc, cos.(centers)))
    end

    plot_flux_and_adv("linvanleer_mono5", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # LinVanLeer limited flux conv, with f(z) = sin(z)
    @test conv_adv_wc[1] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[2] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[3] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[4] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[5] ≈ 1.5 atol = 0.01

end

@testset "Lin et al. (1994) van Leer class limiter (Mono4)" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8, 9, 10)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()
    results = []

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        SLMethod = Operators.LinVanLeerC2F(;
            constraint = Operators.MonotoneHarmonic(),
        )

        divf2c = Operators.DivergenceF2C()
        flux = SLMethod.(w, c, FT(0))
        adv_wc = divf2c.(flux)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
        push!(results, (n, w, c, flux, adv_wc, cos.(centers)))
    end

    plot_flux_and_adv("linvanleer_mono4", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # LinVanLeer limited flux conv, with f(z) = sin(z)
    @test conv_adv_wc[1] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[2] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[3] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[4] ≈ 1.5 atol = 0.01
    @test conv_adv_wc[5] ≈ 1.5 atol = 0.01

end

@testset "Lin et al. (1994) van Leer class limiter (PosDef)" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8, 9, 10)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()
    results = []

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        SLMethod = Operators.LinVanLeerC2F(
            constraint = Operators.PositiveDefinite(),
        )

        divf2c = Operators.DivergenceF2C()
        flux = SLMethod.(w, c, FT(0))
        adv_wc = divf2c.(flux)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
        push!(results, (n, w, c, flux, adv_wc, cos.(centers)))
    end

    plot_flux_and_adv("linvanleer_posdef", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # LinVanLeer limited flux conv, with f(z) = sin(z)
    @test conv_adv_wc[1] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[2] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[3] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[4] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[5] ≈ 1.0 atol = 0.01

end

@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on (uniform and stretched) non-periodic mesh, finite-volume-averaged initial condition" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(1.0))
    device = ClimaComms.device()

    for (i, stretch_fn) in enumerate(stretch_fns)
        err_adv_wc = zeros(FT, length(n_elems_seq))
        Δh = zeros(FT, length(n_elems_seq))
        results = []
        for (k, n) in enumerate(n_elems_seq)
            domain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(-pi),
                Geometry.ZPoint{FT}(pi);
                boundary_names = (:bottom, :top),
            )
            mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            centers = getproperty(Fields.coordinate_field(cs), :z)
            C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

            # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face operator
            # Unitary, constant advective velocity
            w = Geometry.WVector.(ones(fs))
            # c = sin(z), scalar field defined at the centers
            Δz = FT(2pi / n)
            c = (cos.(centers .- Δz / 2) .- cos.(centers .+ Δz / 2)) ./ Δz
            s = sin.(centers)

            first_order_fluxᶠ = Operators.UpwindBiasedProductC2F(
                bottom = Operators.FirstOrderOneSided(),
                top = Operators.FirstOrderOneSided(),
            )
            third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
                bottom = Operators.FirstOrderOneSided(),
                top = Operators.FirstOrderOneSided(),
            )

            divf2c = Operators.DivergenceF2C(
                bottom = Operators.SetValue(
                    Geometry.Contravariant3Vector(FT(0.0)),
                ),
                top = Operators.SetValue(
                    Geometry.Contravariant3Vector(FT(0.0)),
                ),
            )

            first_order_flux = first_order_fluxᶠ.(w, c)
            third_order_flux = third_order_fluxᶠ.(w, c)
            corrected_antidiff_flux =
                @. divf2c(C * (third_order_flux - first_order_flux))
            adv_wc = @. divf2c(first_order_flux) + corrected_antidiff_flux
            # The total flux whose divergence is adv_wc
            flux = @. first_order_flux +
                      C * (third_order_flux - first_order_flux)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end

            # Error
            err_adv_wc[k] = norm(adv_wc .- cos.(centers))
            push!(results, (n, w, c, flux, adv_wc, cos.(centers)))
        end

        plot_flux_and_adv("fct_nonperiodic_fv_ic_$(stretch_names[i])", results)

        # Check convergence rate
        conv_adv_wc = convergence_rate(err_adv_wc, Δh)
        # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
        # The L2 error is dominated by the zeroth-order one-sided boundary
        # reconstruction, so the uniform-mesh rate is ~0.5; on the stretched
        # mesh the rate is ragged, so the measured values are asserted instead.
        @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 0.2006
        if stretch_fn isa Meshes.Uniform
            @test conv_adv_wc[1] ≈ 0.5 atol = 0.05
            @test conv_adv_wc[2] ≈ 0.5 atol = 0.05
            @test conv_adv_wc[3] ≈ 0.5 atol = 0.05
        else
            @test conv_adv_wc[1] ≈ 0.68 atol = 0.1
            @test conv_adv_wc[2] ≈ 0.17 atol = 0.1
            @test conv_adv_wc[3] ≈ 1.13 atol = 0.1
        end
    end
end

@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on uniform non-periodic mesh, pointwise initial condition" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(1.0))
    device = ClimaComms.device()

    for (i, stretch_fn) in enumerate(stretch_fns)
        err_adv_wc = zeros(FT, length(n_elems_seq))
        Δh = zeros(FT, length(n_elems_seq))
        results = []
        for (k, n) in enumerate(n_elems_seq)
            domain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(-pi),
                Geometry.ZPoint{FT}(pi);
                boundary_names = (:bottom, :top),
            )
            mesh = Meshes.IntervalMesh(domain; nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)

            centers = getproperty(Fields.coordinate_field(cs), :z)
            C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

            # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face operator
            # Unitary, constant advective velocity
            w = Geometry.WVector.(ones(fs))
            # c = sin(z), scalar field defined at the centers
            c = sin.(centers)

            first_order_fluxᶠ = Operators.UpwindBiasedProductC2F(
                bottom = Operators.FirstOrderOneSided(),
                top = Operators.FirstOrderOneSided(),
            )
            third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
                bottom = Operators.ThirdOrderOneSided(),
                top = Operators.ThirdOrderOneSided(),
            )

            divf2c = Operators.DivergenceF2C(
                bottom = Operators.SetValue(Geometry.WVector(FT(0.0))),
                top = Operators.SetValue(Geometry.WVector(FT(0.0))),
            )
            first_order_flux = first_order_fluxᶠ.(w, c)
            third_order_flux = third_order_fluxᶠ.(w, c)
            corrected_antidiff_flux =
                @. divf2c(C * (third_order_flux - first_order_flux))
            adv_wc = @. divf2c(first_order_flux) + corrected_antidiff_flux
            # The total flux whose divergence is adv_wc
            flux = @. first_order_flux +
                      C * (third_order_flux - first_order_flux)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end
            # Errors
            err_adv_wc[k] = norm(adv_wc .- cos.(centers))
            push!(results, (n, w, c, flux, adv_wc, cos.(centers)))

        end

        plot_flux_and_adv("fct_nonperiodic_pointwise_ic_$(stretch_names[i])", results)

        # Check convergence rate
        conv_adv_wc = convergence_rate(err_adv_wc, Δh)
        # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
        @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 5e-1
        @test conv_adv_wc[1] ≈ 2.5 atol = 0.05
        @test conv_adv_wc[2] ≈ 2.5 atol = 0.05
        @test conv_adv_wc[3] ≈ 2.5 atol = 0.05
    end
end

@testset "Center -> Face -> Center Advection" begin

    function advection(c, f, cs)
        adv = zeros(eltype(f), cs)
        gradc2f = Operators.GradientC2F(
            bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
            top = Operators.SetGradient(Geometry.WVector(FT(1))),
        )
        interpf2c = Operators.InterpolateF2C()
        return @. adv = interpf2c(
            LinearAlgebra.dot(Geometry.Contravariant3Vector(c), gradc2f(f)),
        )
    end

    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    err = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(0.0),
            Geometry.ZPoint{FT}(4π);
            boundary_names = (:bottom, :top),
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        # Advective velocity
        c = Geometry.WVector.(ones(Float64, fs))
        # Scalar-valued field to be advected
        f = sin.(Fields.coordinate_field(cs).z)

        # Call the advection operator
        adv = advection(c, f, cs)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end
        err[k] = norm(adv .- cos.(Fields.coordinate_field(cs).z))
    end
    # Center -> face -> center advection convergence rate
    conv_adv_c2c = convergence_rate(err, Δh)
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
    @test conv_adv_c2c[1] ≈ 2 atol = 0.1
    @test conv_adv_c2c[2] ≈ 2 atol = 0.1
    @test conv_adv_c2c[3] ≈ 2 atol = 0.1
end
