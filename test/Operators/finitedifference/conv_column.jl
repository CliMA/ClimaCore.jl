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

Write `output/conv_column/<device>/<name>.png` (`name` may contain a
subdirectory, which is created), showing the advecting velocity
`w` and the advected field `c` that go into the flux, the face-valued flux
itself, and the center-valued advective tendency `adv_wc`, at every resolution.
`results` is a vector of `(n, w, c, flux, adv_wc, adv_exact)` tuples, one per
element count, and the exact tendency of the finest resolution is drawn on top
of `adv_wc`. A no-op when the plotting packages are unavailable.
"""
function plot_flux_and_adv(name, results)
    HAVE_PLOTS || return nothing
    mkpath(dirname(joinpath(plot_dir, name)))
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

        centers = Fields.coordinate_field(cs).z
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

##### Advection operator + DivergenceF2C convergence #####
#
# The cases below share one experiment, parameterized by the advection
# operator (with its boundary conditions) and the functions that generate the
# velocity and advected input fields. At each resolution `n`, on the interval
# (-π, π):
#
#     w      = WVector(w_fn(z_face))
#     c      = c_fn(z_center, Δz)                 (Δz = 2π / n)
#     adv_wc = divf2c(op(w, c, op_args...))
#
# and the L2 error of `adv_wc` against `exact_fn(z_center)` gives one
# convergence rate per refinement step. Each case lists the meshes it runs on
# (periodic cases use a single uniform mesh) with the expected rate and
# tolerance of every refinement step; when `err_bound` is given, the errors of
# the first three resolutions must also decrease monotonically and stay below
# it. A mesh entry may also carry two more entries of the same
# `(; bound, rates)` form, each applying the same two checks to another error
# measure:
#  - `cons`: the total-mass tendency `|∫ adv_wc|`. For a flux-form tendency
#    this integral telescopes to the flux difference between the two boundary
#    faces, so when the divergence operator imposes the boundary fluxes with
#    `SetValue`, conservation is exact and the drift is roundoff noise: pass
#    an empty `rates` list, and the drift is only required to stay below
#    `bound` at every resolution (roundoff need not decrease with
#    resolution). With nonempty `rates`, the drift instead measures the
#    conservation error of the boundary fluxes the advection operator
#    produces, and the usual monotone-decrease, bound, and rate checks apply.
#  - `flux_conv`: the L2 error of the advection operator's own face-valued
#    output against the exact flux `flux_exact_fn(z_face)` (a case entry),
#    compared as physical WVector components, so that the convergence of the
#    operator is measured directly rather than through the divergence. The
#    wall faces are zeroed out of the error: an advection operator's output at
#    the wall faces is usually unused, because the enclosing divergence
#    imposes the boundary fluxes with its own boundary conditions (as these
#    cases' divergence does), so the operator is measured on the faces that
#    are actually consumed — the one-in and interior faces.

uniform_mesh(rates; cons = nothing, flux_conv = nothing) =
    (; stretch_name = "", stretch = Meshes.Uniform(), rates, cons, flux_conv)

# Zeroes the wall faces of the face-valued flux-error field (see `flux_conv`).
zero_wall_faces = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(Geometry.WVector(0.0)),
    top = Operators.SetValue(Geometry.WVector(0.0)),
)

"""
    op_plot_string(op)

A filename-friendly description of an advection operator, appended to every
plot name so that each plot records the operator and boundary conditions that
produced it: the operator's type name, its limiter constraint (for
`LinVanLeerC2F`), and each of its boundary conditions, e.g.
`Upwind3rdOrderBiasedProductC2F_bottom_Extrapolate1_top_Extrapolate1`. On
periodic meshes the boundary conditions in the name are the operator's
(unused) defaults.
"""
op_plot_string(op) = join(
    (
        string(nameof(typeof(op))),
        (
            op isa Operators.LinVanLeerC2F ?
            (string(nameof(typeof(op.constraint))),) : ()
        )...,
        (
            "$(name)_$(bc_plot_string(getproperty(op.bcs, name)))" for
            name in propertynames(op.bcs)
        )...,
    ),
    "_",
)
bc_plot_string(::Operators.Extrapolate{N}) where {N} = "Extrapolate$N"
bc_plot_string(bc) = string(nameof(typeof(bc)))

function test_advection_convergence(case)
    FT = Float64
    (; n_elems_seq, op, op_args, w_fn, c_fn, exact_fn, divf2c) = case
    for mesh_case in case.meshes
        (; stretch_name, stretch, rates) = mesh_case
        cons = get(mesh_case, :cons, nothing)
        flux_conv = get(mesh_case, :flux_conv, nothing)
        err_adv_wc = zeros(FT, length(n_elems_seq))
        err_mass = zeros(FT, length(n_elems_seq))
        err_flux = zeros(FT, length(n_elems_seq))
        Δh = zeros(FT, length(n_elems_seq))
        results = []
        for (k, n) in enumerate(n_elems_seq)
            domain = Domains.IntervalDomain(
                Geometry.ZPoint{FT}(-pi),
                Geometry.ZPoint{FT}(pi);
                (case.periodic ? (; periodic = true) :
                 (; boundary_names = (:bottom, :top)))...,
            )
            mesh = Meshes.IntervalMesh(domain, stretch; nelems = n)

            cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
            fs = Spaces.face_space(cs)
            centers = Fields.coordinate_field(cs).z
            faces = Fields.coordinate_field(fs).z

            w = Geometry.WVector.(w_fn.(faces))
            c = c_fn.(centers, FT(2pi / n))
            flux = op.(w, c, op_args...)
            adv_wc = divf2c.(flux)

            ClimaComms.allowscalar(device) do
                Δh[k] = Spaces.local_geometry_data(fs).J[1]
            end

            exact = exact_fn.(centers)
            err_adv_wc[k] = norm(adv_wc .- exact)
            cons === nothing || (err_mass[k] = abs(sum(adv_wc)))
            flux_conv === nothing || (
                err_flux[k] = norm(
                    zero_wall_faces.(
                        Geometry.WVector.(flux) .-
                        Geometry.WVector.(case.flux_exact_fn.(faces)),
                    ),
                )
            )
            push!(results, (n, w, c, flux, adv_wc, exact))
        end

        suffix = isempty(stretch_name) ? "" : "_$(stretch_name)"
        plot_flux_and_adv(
            "$(case.plot_name)_$(op_plot_string(op))$suffix",
            results,
        )

        if case.err_bound !== nothing
            @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ case.err_bound
        end
        conv_adv_wc = convergence_rate(err_adv_wc, Δh)
        for (i, (expected, tol)) in enumerate(rates)
            @test conv_adv_wc[i] ≈ expected atol = tol
        end
        if cons !== nothing
            if isempty(cons.rates)
                @test maximum(err_mass) ≤ cons.bound
            else
                @test err_mass[3] ≤ err_mass[2] ≤ err_mass[1] ≤ cons.bound
                conv_mass = convergence_rate(err_mass, Δh)
                for (i, (expected, tol)) in enumerate(cons.rates)
                    @test conv_mass[i] ≈ expected atol = tol
                end
            end
        end
        if flux_conv !== nothing
            @test err_flux[3] ≤ err_flux[2] ≤ err_flux[1] ≤ flux_conv.bound
            conv_flux = convergence_rate(err_flux, Δh)
            for (i, (expected, tol)) in enumerate(flux_conv.rates)
                @test conv_flux[i] ≈ expected atol = tol
            end
        end
    end
end

# Inputs for the conservation cases. The scalar vanishes at the boundaries
# z = ±π (with zero slope, since c'(z) = -sin(z) also vanishes there), and the
# velocity attains its extrema there (w'(z) = cos(z / 2) / 2 vanishes at ±π
# and nowhere else), so the exact boundary flux `w * c` is zero — which is the
# analytical value the cases' DivergenceF2C `SetValue` boundary conditions
# impose — and the exact total-mass tendency `∫ ∂(w c)/∂z` vanishes.
# `w_extrema_at_boundary` (w < 0 at the bottom, w > 0 at the top) makes both
# boundaries outflow boundaries, so the advection operators' near-boundary
# fluxes are dominated by the interior points of the upwind stencils;
# `w_inflow_at_boundary` reverses it, so the upwind side at each boundary is
# the ghost side and the near-boundary fluxes are built from the ghost-point
# extrapolations instead.
w_extrema_at_boundary(z) = sin(z / 2)
w_inflow_at_boundary(z) = -w_extrema_at_boundary(z)
c_zero_at_boundary(z, Δz) = 1 + cos(z)
c_zero_at_boundary_flux(z) = w_extrema_at_boundary(z) * c_zero_at_boundary(z, 0)
c_zero_at_boundary_tendency(z) =
    cos(z / 2) * (1 + cos(z)) / 2 - sin(z / 2) * sin(z)
c_inflow_at_boundary_flux(z) = -c_zero_at_boundary_flux(z)
c_inflow_at_boundary_tendency(z) = -c_zero_at_boundary_tendency(z)

# A scalar with a simple zero at the boundaries: sin(±π) = 0 but
# c'(±π) = -1 ≠ 0. `c_zero_at_boundary`'s double zero makes every ghost value
# near the wall O(Δz²) regardless of the extrapolation order, so it cannot
# distinguish the orders; with a simple zero, the constant (Extrapolate{0})
# ghost error is O(Δz) while the linear (Extrapolate{1}) one is O(Δz²), and
# under an inflow velocity — where the ghost side is the upwind side — the
# near-boundary errors separate the orders.
c_simple_zero_at_boundary(z, Δz) = sin(z)
c_simple_zero_inflow_flux(z) = w_inflow_at_boundary(z) * sin(z)
c_simple_zero_inflow_tendency(z) =
    -cos(z / 2) * sin(z) / 2 - sin(z / 2) * cos(z)

# The exact boundary flux of the conservation cases, imposed analytically.
zero_flux_divf2c = Operators.DivergenceF2C(
    bottom = Operators.SetValue(Geometry.WVector(0.0)),
    top = Operators.SetValue(Geometry.WVector(0.0)),
)

# `c_fn` receives the uniform grid spacing Δz so that an initial condition can
# be finite-volume averaged; most cases ignore it.
advection_convergence_cases = [
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on (uniform) \
                periodic mesh, constant w",
        plot_name = "periodic/constant_w",
        periodic = true,
        n_elems_seq = 2 .^ (5, 6, 7, 8),
        op = Operators.Upwind3rdOrderBiasedProductC2F(),
        op_args = (),
        w_fn = one,
        c_fn = (z, Δz) -> sin(z),
        exact_fn = cos,
        divf2c = Operators.DivergenceF2C(),
        err_bound = 5e-4,
        # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
        meshes = [uniform_mesh([(3, 0.1), (3, 0.1), (3, 0.1)])],
    ),
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on (uniform) \
                periodic mesh, varying sign w",
        plot_name = "periodic/varying_w",
        periodic = true,
        n_elems_seq = 2 .^ (5, 6, 7, 8),
        op = Operators.Upwind3rdOrderBiasedProductC2F(),
        op_args = (),
        w_fn = cos,
        c_fn = (z, Δz) -> sin(z),
        exact_fn = z -> cos(z)^2 - sin(z)^2,
        divf2c = Operators.DivergenceF2C(),
        err_bound = 4e-3,
        # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = cos(z)
        meshes = [uniform_mesh([(2, 0.2), (2, 0.1), (2, 0.1)])],
    ),
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on (uniform \
                and stretched) non-periodic mesh, with Extrapolate{0} + \
                DivergenceF2C SetValue BCs, constant w",
        plot_name = "nonperiodic/constant_w",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.Extrapolate(0),
            top = Operators.Extrapolate(0),
        ),
        op_args = (),
        w_fn = one,
        # finite-volume average of sin(z) over each cell
        c_fn = (z, Δz) -> (cos(z - Δz / 2) - cos(z + Δz / 2)) / Δz,
        exact_fn = cos,
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
            top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        ),
        err_bound = 0.2006,
        # The L2 error is dominated by the zeroth-order one-sided boundary
        # extrapolation, so the uniform-mesh rate is ~0.5; on the stretched
        # mesh the rate is ragged, so the measured values are asserted instead.
        meshes = [
            (;
                stretch_name = "uniform",
                stretch = Meshes.Uniform(),
                rates = [(0.5, 0.05), (0.5, 0.05), (0.5, 0.05)],
            ),
            (;
                stretch_name = "stretched",
                stretch = Meshes.ExponentialStretching(1.0),
                rates = [(0.68, 0.1), (0.17, 0.1), (1.13, 0.1)],
            ),
        ],
    ),
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on uniform \
                non-periodic mesh, with Extrapolate{1} + DivergenceF2C \
                SetValue BCs, varying sign w",
        plot_name = "nonperiodic/varying_w",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.Extrapolate(1),
            top = Operators.Extrapolate(1),
        ),
        op_args = (),
        w_fn = cos,
        c_fn = (z, Δz) -> sin(z),
        exact_fn = z -> cos(z)^2 - sin(z)^2,
        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.WVector(0.0)),
            top = Operators.SetValue(Geometry.WVector(0.0)),
        ),
        err_bound = 2e-1,
        # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = cos(z);
        # the coarsest refinement step measures ~2.12. (The pre-refactor version
        # of this testset looped over a stretched mesh as well, but never
        # actually applied the stretching when building the mesh, so only the
        # uniform mesh is kept here.)
        meshes = [uniform_mesh([(2, 0.15), (2, 0.05), (2, 0.05)])],
    ),
    # Lin et al. (1994) van Leer class limiters, with f(z) = sin(z); the
    # `mismatch` Δ𝜙 = 0 (dt = 0) reduces each constraint to its limiting
    # reconstruction.
    (;
        name = "Lin et al. (1994) van Leer class limiter (Mono5)",
        plot_name = "periodic/constant_w",
        periodic = true,
        n_elems_seq = 2 .^ (5, 6, 7, 8, 9, 10),
        op = Operators.LinVanLeerC2F(
            constraint = Operators.MonotoneLocalExtrema(),
        ),
        op_args = (0.0,), # dt
        w_fn = one,
        c_fn = (z, Δz) -> sin(z),
        exact_fn = cos,
        divf2c = Operators.DivergenceF2C(),
        err_bound = nothing,
        meshes = [uniform_mesh([((1.5, 0.01) for _ in 1:5)...])],
    ),
    (;
        name = "Lin et al. (1994) van Leer class limiter (Mono4)",
        plot_name = "periodic/constant_w",
        periodic = true,
        n_elems_seq = 2 .^ (5, 6, 7, 8, 9, 10),
        op = Operators.LinVanLeerC2F(constraint = Operators.MonotoneHarmonic()),
        op_args = (0.0,), # dt
        w_fn = one,
        c_fn = (z, Δz) -> sin(z),
        exact_fn = cos,
        divf2c = Operators.DivergenceF2C(),
        err_bound = nothing,
        meshes = [uniform_mesh([((1.5, 0.01) for _ in 1:5)...])],
    ),
    (;
        name = "Lin et al. (1994) van Leer class limiter (PosDef)",
        plot_name = "periodic/constant_w",
        periodic = true,
        n_elems_seq = 2 .^ (5, 6, 7, 8, 9, 10),
        op = Operators.LinVanLeerC2F(constraint = Operators.PositiveDefinite()),
        op_args = (0.0,), # dt
        w_fn = one,
        c_fn = (z, Δz) -> sin(z),
        exact_fn = cos,
        divf2c = Operators.DivergenceF2C(),
        err_bound = nothing,
        meshes = [uniform_mesh([((1.0, 0.01) for _ in 1:5)...])],
    ),
    # Conservation and operator-convergence cases (see
    # `w_extrema_at_boundary`): the advected scalar vanishes at the
    # boundaries, so the exact boundary flux is zero, and DivergenceF2C
    # imposes that analytical value with `SetValue`. The total-mass tendency
    # then telescopes to exactly zero, so the drift `|∫ adv_wc|` must stay at
    # roundoff level at every resolution (`cons` with empty `rates`), and the
    # operators' own outputs also converge (`flux_conv`). The tendency L2
    # rates are limited by the upwind biasing (~1 first-order, ~1.5
    # third-order and van Leer, from the measured values).
    (;
        name = "UpwindBiasedProductC2F + DivergenceF2C conservation on \
                uniform non-periodic mesh, c -> 0 and w extremal at the \
                boundaries",
        plot_name = "conservation/outflow",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.UpwindBiasedProductC2F(),
        op_args = (),
        w_fn = w_extrema_at_boundary,
        c_fn = c_zero_at_boundary,
        exact_fn = c_zero_at_boundary_tendency,
        flux_exact_fn = c_zero_at_boundary_flux,
        divf2c = zero_flux_divf2c,
        err_bound = 0.12,
        meshes = [
            uniform_mesh(
                [(1, 0.05), (1, 0.05), (1, 0.05)];
                cons = (; bound = 1e-12, rates = []),
                flux_conv = (;
                    bound = 0.11,
                    rates = [(1, 0.05), (1, 0.05), (1, 0.05)],
                ),
            ),
        ],
    ),
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C conservation \
                on uniform non-periodic mesh, with Extrapolate{1}, c -> 0 \
                and w extremal at the boundaries",
        plot_name = "conservation/outflow",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.Extrapolate(1),
            top = Operators.Extrapolate(1),
        ),
        op_args = (),
        w_fn = w_extrema_at_boundary,
        c_fn = c_zero_at_boundary,
        exact_fn = c_zero_at_boundary_tendency,
        flux_exact_fn = c_zero_at_boundary_flux,
        divf2c = zero_flux_divf2c,
        err_bound = 0.006,
        meshes = [
            uniform_mesh(
                [(1.5, 0.05), (1.5, 0.05), (1.5, 0.05)];
                cons = (; bound = 1e-12, rates = []),
                # the interior stencil is third-order, but the L2 error is
                # dominated by the second-order one-in faces, so the measured
                # rates approach 2 from below
                flux_conv = (;
                    bound = 0.003,
                    rates = [(2, 0.1), (2, 0.1), (2, 0.1)],
                ),
            ),
        ],
    ),
    (;
        name = "Lin et al. (1994) van Leer class limiter (Mono5) + \
                DivergenceF2C conservation on uniform non-periodic mesh, \
                c -> 0 and w extremal at the boundaries",
        plot_name = "conservation/outflow",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.LinVanLeerC2F(
            constraint = Operators.MonotoneLocalExtrema(),
        ),
        op_args = (0.0,), # dt
        w_fn = w_extrema_at_boundary,
        c_fn = c_zero_at_boundary,
        exact_fn = c_zero_at_boundary_tendency,
        flux_exact_fn = c_zero_at_boundary_flux,
        divf2c = zero_flux_divf2c,
        err_bound = 0.017,
        meshes = [
            uniform_mesh(
                [(1.5, 0.05), (1.5, 0.05), (1.5, 0.05)];
                cons = (; bound = 1e-12, rates = []),
                flux_conv = (;
                    bound = 0.008,
                    rates = [(1.91, 0.05), (1.96, 0.05), (1.99, 0.05)],
                ),
            ),
        ],
    ),
    # Inflow at the boundaries: with `w_inflow_at_boundary` the upwind side at
    # each boundary is the ghost side, so the near-boundary fluxes are built
    # from the ghost-point extrapolations, which this case exercises directly
    # (in the outflow cases above they barely enter).
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C conservation \
                on uniform non-periodic mesh, with Extrapolate{1}, c -> 0 \
                and w inflow at the boundaries",
        plot_name = "conservation/inflow",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.Extrapolate(1),
            top = Operators.Extrapolate(1),
        ),
        op_args = (),
        w_fn = w_inflow_at_boundary,
        c_fn = c_zero_at_boundary,
        exact_fn = c_inflow_at_boundary_tendency,
        flux_exact_fn = c_inflow_at_boundary_flux,
        divf2c = zero_flux_divf2c,
        err_bound = 0.03,
        meshes = [
            uniform_mesh(
                [(1.5, 0.02), (1.5, 0.02), (1.5, 0.02)];
                cons = (; bound = 1e-12, rates = []),
                flux_conv = (;
                    bound = 0.008,
                    rates = [(2.3, 0.1), (2.18, 0.1), (2.06, 0.1)],
                ),
            ),
        ],
    ),
    # Extrapolation-order sensitivity (see `c_simple_zero_at_boundary`): with
    # inflow at the boundaries and a scalar whose zero at the walls is simple,
    # the O(Δz) one-in ghost error of Extrapolate{0} collapses the tendency
    # rate to ~0.5 near the boundary and the flux rate to ~1.5, while
    # Extrapolate{1} keeps them at ~2. Extrapolate{2} is indistinguishable
    # from Extrapolate{1} here, so it is not tested: at the boundary face the
    # order is capped at 1 (the two are bitwise identical there, and that face
    # is excluded from the flux error anyway), and at the one-in face sin's
    # inflection at its boundary zero (c''(±π) = 0) degenerates
    # Extrapolate{1}'s O(Δz²) ghost error to O(Δz³), the same order as
    # Extrapolate{2}'s. A scalar with a simple zero and nonzero curvature at
    # the walls would separate them.
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on uniform \
                non-periodic mesh, with Extrapolate{0}, c with a simple zero \
                and w inflow at the boundaries",
        plot_name = "conservation/inflow_simple_zero",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.Extrapolate(0),
            top = Operators.Extrapolate(0),
        ),
        op_args = (),
        w_fn = w_inflow_at_boundary,
        c_fn = c_simple_zero_at_boundary,
        exact_fn = c_simple_zero_inflow_tendency,
        flux_exact_fn = c_simple_zero_inflow_flux,
        divf2c = zero_flux_divf2c,
        err_bound = 0.09,
        meshes = [
            uniform_mesh(
                [(0.5, 0.05), (0.5, 0.05), (0.5, 0.05)];
                cons = (; bound = 1e-12, rates = []),
                flux_conv = (;
                    bound = 0.025,
                    rates = [(1.5, 0.05), (1.5, 0.05), (1.5, 0.05)],
                ),
            ),
        ],
    ),
    (;
        name = "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on uniform \
                non-periodic mesh, with Extrapolate{1}, c with a simple zero \
                and w inflow at the boundaries",
        plot_name = "conservation/inflow_simple_zero",
        periodic = false,
        n_elems_seq = 2 .^ (4, 6, 8, 10),
        op = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.Extrapolate(1),
            top = Operators.Extrapolate(1),
        ),
        op_args = (),
        w_fn = w_inflow_at_boundary,
        c_fn = c_simple_zero_at_boundary,
        exact_fn = c_simple_zero_inflow_tendency,
        flux_exact_fn = c_simple_zero_inflow_flux,
        divf2c = zero_flux_divf2c,
        err_bound = 0.013,
        meshes = [
            uniform_mesh(
                [(2.41, 0.1), (2.23, 0.1), (2.09, 0.1)];
                cons = (; bound = 1e-12, rates = []),
                flux_conv = (;
                    bound = 0.004,
                    rates = [(2.13, 0.1), (1.96, 0.05), (1.99, 0.05)],
                ),
            ),
        ],
    ),
]

for case in advection_convergence_cases
    @testset "$(case.name)" begin
        test_advection_convergence(case)
    end
end

@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on (uniform) periodic mesh" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()
    results = []

    # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face
    # operators (resolution-independent, so they can also name the plot)
    first_order_fluxᶠ = Operators.UpwindBiasedProductC2F()
    third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F()
    divf2c = Operators.DivergenceF2C()

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = Fields.coordinate_field(cs).z
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        first_order_fluxsinᶠ = first_order_fluxᶠ.(w, c)
        third_order_fluxsinᶠ = third_order_fluxᶠ.(w, c)
        corrected_antidiff_flux =
            @. divf2c(C * (third_order_fluxsinᶠ - first_order_fluxsinᶠ))
        adv_wc = @. divf2c(first_order_fluxsinᶠ) + corrected_antidiff_flux
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

    fct_ops = "$(op_plot_string(first_order_fluxᶠ))_plus_$(op_plot_string(third_order_fluxᶠ))"
    plot_flux_and_adv("fct/periodic_$fct_ops", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 5e-4
    @test conv_adv_wc[1] ≈ 3 atol = 0.1
    @test conv_adv_wc[2] ≈ 3 atol = 0.1
    @test conv_adv_wc[3] ≈ 3 atol = 0.1
end


@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on (uniform and stretched) non-periodic mesh, finite-volume-averaged initial condition" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(1.0))
    device = ClimaComms.device()

    # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face
    # operators (resolution-independent, so they can also name the plot)
    first_order_fluxᶠ = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(0),
        top = Operators.Extrapolate(0),
    )
    third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.Extrapolate(0),
        top = Operators.Extrapolate(0),
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
    )
    fct_ops = "$(op_plot_string(first_order_fluxᶠ))_plus_$(op_plot_string(third_order_fluxᶠ))"

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

            centers = Fields.coordinate_field(cs).z
            C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

            # Unitary, constant advective velocity
            w = Geometry.WVector.(ones(fs))
            # c = sin(z), scalar field defined at the centers
            Δz = FT(2pi / n)
            c = (cos.(centers .- Δz / 2) .- cos.(centers .+ Δz / 2)) ./ Δz

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

        plot_flux_and_adv(
            "fct/nonperiodic_fv_ic_$(fct_ops)_$(stretch_names[i])",
            results,
        )

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
    device = ClimaComms.device()

    err_adv_wc = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    results = []

    # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face
    # operators (resolution-independent, so they can also name the plot)
    first_order_fluxᶠ = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(0),
        top = Operators.Extrapolate(0),
    )
    third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.Extrapolate(1),
        top = Operators.Extrapolate(1),
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0.0))),
        top = Operators.SetValue(Geometry.WVector(FT(0.0))),
    )

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            boundary_names = (:bottom, :top),
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.face_space(cs)

        centers = Fields.coordinate_field(cs).z
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        first_order_flux = first_order_fluxᶠ.(w, c)
        third_order_flux = third_order_fluxᶠ.(w, c)
        corrected_antidiff_flux =
            @. divf2c(C * (third_order_flux - first_order_flux))
        adv_wc = @. divf2c(first_order_flux) + corrected_antidiff_flux
        # The total flux whose divergence is adv_wc
        flux = @. first_order_flux + C * (third_order_flux - first_order_flux)

        ClimaComms.allowscalar(device) do
            Δh[k] = Spaces.local_geometry_data(fs).J[1]
        end
        # Errors
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
        push!(results, (n, w, c, flux, adv_wc, cos.(centers)))
    end

    fct_ops = "$(op_plot_string(first_order_fluxᶠ))_plus_$(op_plot_string(third_order_fluxᶠ))"
    plot_flux_and_adv("fct/nonperiodic_pointwise_ic_$fct_ops", results)

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)
    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 5e-1
    @test conv_adv_wc[1] ≈ 2.5 atol = 0.05
    @test conv_adv_wc[2] ≈ 2.5 atol = 0.05
    @test conv_adv_wc[3] ≈ 2.5 atol = 0.05
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
