#=
julia --check-bounds=yes --project=.buildkite

# Uniform mesh:
using Revise; include("test/Operators/finitedifference/convergence_column.jl")

# stretched mesh:
empty!(ARGS)
push!(ARGS, "--stretch", "ExponentialStretching")
using Revise; include("test/Operators/finitedifference/convergence_column.jl")
=#
using Test
using StaticArrays, IntervalSets, LinearAlgebra

using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: slab, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Domains: Geometry
import ClimaCore.DataLayouts: vindex
using ClimaCore.CommonSpaces

import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--stretch"
        help = "Float type"
        arg_type = String
        default = "uniform"
    end
    return ArgParse.parse_args(ARGS, s)
end

function stretch_function(FT, parsed_args, param = 0.5)
    if parsed_args["stretch"] == "ExponentialStretching"
        return Meshes.ExponentialStretching(FT(param))
    elseif parsed_args["stretch"] == "uniform"
        return Meshes.Uniform()
    else
        error("Unexpected stretch passed to parsed_args. Using uniform")
    end
end

"""
    (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

Returns the face space and coordinate information for the column convergence
tests.
"""
function coord_info(ᶜspace)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶜJ = Fields.local_geometry_field(ᶜspace).J
    ᶠJ = Fields.local_geometry_field(ᶠspace).J
    ᶠJdata = Spaces.local_geometry_data(ᶠspace).J
    ᶜJdata = Spaces.local_geometry_data(ᶜspace).J
    return (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata)
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


@testset "Face -> Center interpolation" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    stretch = stretch_function(FT, parse_commandline())
    err = zeros(FT, length(n_elems_seq))
    werr = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        ᶜspace = ColumnSpace(FT;
            z_elem=n,
            z_min=0,
            z_max=1,
            stretch,
            staggering=CellCenter()
        )
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        face_field = sin.(3π .* ᶠz)

        cent_field_exact = sin.(3π .* ᶜz)
        operator = Operators.InterpolateF2C()
        woperator = Operators.WeightedInterpolateF2C()
        cent_field = operator.(face_field)
        wcent_field = woperator.(ᶠJ, face_field)

        Δh[k] = ᶠJdata[vindex(1)]
        err[k] = norm(cent_field .- cent_field_exact)
        werr[k] = norm(wcent_field .- cent_field_exact)
    end

    conv = convergence_rate(err, Δh)
    wconv = convergence_rate(werr, Δh)
    # conv should be approximately 2 for second order-accurate stencil.
    @test all(1.8 .<= conv .<= 2)
    @test all(1.8 .<= wconv .<= 2)
end

@testset "Center -> Face interpolation" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    device = ClimaComms.device()
    stretch = stretch_function(FT, parse_commandline())
    err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
    werr = zeros(FT, length(n_elems_seq))
    for (k, n) in enumerate(n_elems_seq)
        ᶜspace = ColumnSpace(FT;
            z_elem=n,
            z_min=0,
            z_max=1,
            stretch,
            staggering=CellCenter()
        )
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        cent_field = sin.(3π .* ᶜz)
        face_field_exact = sin.(3π .* ᶠz)

        operator = Operators.InterpolateC2F(
            bottom = Operators.SetValue(0.0),
            top = Operators.SetValue(0.0),
        )
        woperator = Operators.WeightedInterpolateC2F(
            bottom = Operators.SetValue(0.0),
            top = Operators.SetValue(0.0),
        )
        face_field = operator.(cent_field)
        wface_field = woperator.(ᶜJ, cent_field)

        Δh[k] = ᶠJdata[vindex(1)]
        err[k] = norm(face_field .- face_field_exact)
        werr[k] = norm(wface_field .- face_field_exact)
    end
    conv = convergence_rate(err, Δh)
    wconv = convergence_rate(werr, Δh)
    # conv should be approximately 2 for second order-accurate stencil.
    @test all(1.8 .<= conv .<= 2)
    @test all(1.8 .<= wconv .<= 2)
end

@testset "∂ Center -> Face interpolation" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    stretch = stretch_function(FT, parse_commandline())
    err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
    for (k, n) in enumerate(n_elems_seq)
        ᶜspace = ColumnSpace(FT;
            z_elem=n,
            z_min=0,
            z_max=1,
            stretch,
            staggering=CellCenter()
        )
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        cent_field = sin.(3π .* ᶜz)
        face_field_exact =
            Geometry.CovariantVector.(
                Geometry.WVector.(3π .* cos.(3π .* ᶠz)),
            )

        operator = Operators.GradientC2F(
            bottom = Operators.SetGradient(Geometry.WVector(3π)),
            top = Operators.SetGradient(Geometry.WVector(-3π)),
        )

        face_field = operator.(cent_field)

        Δh[k] = ᶠJdata[vindex(1)]
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

@testset "∂ Face -> Center interpolation" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)
    stretch = stretch_function(FT, parse_commandline())
    err, Δh = zeros(FT, length(n_elems_seq)), zeros(FT, length(n_elems_seq))
    for (k, n) in enumerate(n_elems_seq)
        ᶜspace = ColumnSpace(FT;
            z_elem=n,
            z_min=0,
            z_max=1,
            stretch,
            staggering=CellCenter()
        )
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        face_field = sin.(3π .* ᶠz)
        cent_field_exact =
            Geometry.CovariantVector.(
                Geometry.WVector.(3π .* cos.(3π .* ᶜz)),
            )

        operator = Operators.GradientF2C()

        cent_field = operator.(face_field)

        Δh[k] = ᶠJdata[vindex(1)]
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

@testset "∂ Center -> Face and ∂ Face-> Center (uniform)" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_grad_sin_c = zeros(FT, length(n_elems_seq))
    err_div_sin_c = zeros(FT, length(n_elems_seq))
    err_grad_z_f = zeros(FT, length(n_elems_seq))
    err_grad_cos_f1 = zeros(FT, length(n_elems_seq))
    err_grad_cos_f2 = zeros(FT, length(n_elems_seq))
    err_div_sin_f = zeros(FT, length(n_elems_seq))
    err_div_cos_f = zeros(FT, length(n_elems_seq))
    err_curl_sin_f = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    stretch = stretch_function(FT, parse_commandline())

    for (k, n) in enumerate(n_elems_seq)
        ᶜspace = ColumnSpace(FT;
            z_elem=n,
            z_min=0,
            z_max=pi,
            stretch,
            staggering=CellCenter()
        )
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        # Face -> Center operators:
        # GradientF2C
        # f(z) = sin(z)
        ∇ᶜ = Operators.GradientF2C()
        gradsinᶜ = Geometry.WVector.(∇ᶜ.(sin.(ᶠz)))

        # DivergenceF2C
        # f(z) = sin(z)
        divᶜ = Operators.DivergenceF2C()
        divsinᶜ = divᶜ.(Geometry.WVector.(sin.(ᶠz)))

        # Center -> Face operators:
        # GradientC2F, SetValue
        # f(z) = z
        ∇ᶠ⁰ = Operators.GradientC2F(
            bottom = Operators.SetValue(FT(0)),
            top = Operators.SetValue(FT(pi)),
        )
        ∂zᶠ = Geometry.WVector.(∇ᶠ⁰.(ᶜz))

        # GradientC2F, SetValue
        # f(z) = cos(z)
        ∇ᶠ¹ = Operators.GradientC2F(
            bottom = Operators.SetValue(FT(1)),
            top = Operators.SetValue(FT(-1)),
        )
        gradcosᶠ¹ = Geometry.WVector.(∇ᶠ¹.(cos.(ᶜz)))

        # GradientC2F, SetGradient
        # f(z) = cos(z)
        ∇ᶠ² = Operators.GradientC2F(
            bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
            top = Operators.SetGradient(Geometry.WVector(FT(0))),
        )
        gradcosᶠ² = Geometry.WVector.(∇ᶠ².(cos.(ᶜz)))

        # DivergenceC2F, SetValue
        # f(z) = sin(z)
        divᶠ⁰ = Operators.DivergenceC2F(
            bottom = Operators.SetValue(Geometry.WVector(zero(FT))),
            top = Operators.SetValue(Geometry.WVector(zero(FT))),
        )
        divsinᶠ = divᶠ⁰.(Geometry.WVector.(sin.(ᶜz)))

        # DivergenceC2F, SetDivergence
        # f(z) = cos(z)
        divᶠ¹ = Operators.DivergenceC2F(
            bottom = Operators.SetDivergence(FT(0)),
            top = Operators.SetDivergence(FT(0)),
        )
        divcosᶠ = divᶠ¹.(Geometry.WVector.(cos.(ᶜz)))

        curlᶠ = Operators.CurlC2F(
            bottom = Operators.SetValue(Geometry.Covariant1Vector(zero(FT))),
            top = Operators.SetValue(Geometry.Covariant1Vector(zero(FT))),
        )
        curlsinᶠ = curlᶠ.(Geometry.Covariant1Vector.(sin.(ᶜz)))


        Δh[k] = ᶠJdata[vindex(1)]
        # Errors
        err_grad_sin_c[k] = norm(gradsinᶜ .- Geometry.WVector.(cos.(ᶜz)))
        err_div_sin_c[k] = norm(divsinᶜ .- cos.(ᶜz))
        err_grad_z_f[k] = norm(∂zᶠ .- Geometry.WVector.(ones(FT, ᶠspace)))
        err_grad_cos_f1[k] = norm(gradcosᶠ¹ .- Geometry.WVector.(.-sin.(ᶠz)))
        err_grad_cos_f2[k] = norm(gradcosᶠ² .- Geometry.WVector.(.-sin.(ᶠz)))
        err_div_sin_f[k] =
            norm(divsinᶠ .- (Geometry.WVector.(cos.(ᶠz))).components.data.:1)
        err_div_cos_f[k] = norm(
            divcosᶠ .- (Geometry.WVector.(.-sin.(ᶠz))).components.data.:1,
        )
        err_curl_sin_f[k] =
            norm(curlsinᶠ .- Geometry.Contravariant2Vector.(cos.(ᶠz)))
    end

    # GradientF2C conv, with f(z) = sin(z)
    conv_grad_sin_c = convergence_rate(err_grad_sin_c, Δh)
    # DivergenceF2C conv, with f(z) = sin(z)
    conv_div_sin_c = convergence_rate(err_div_sin_c, Δh)
    # GradientC2F conv, with f(z) = z, SetValue
    conv_grad_z = convergence_rate(err_grad_z_f, Δh)
    # GradientC2F conv, with f(z) = cos(z), SetValue
    conv_grad_cos_f1 = convergence_rate(err_grad_cos_f1, Δh)
    # GradientC2F conv, with f(z) = cos(z), SetGradient
    conv_grad_cos_f2 = convergence_rate(err_grad_cos_f2, Δh)
    # DivergenceC2F conv, with f(z) = sin(z), SetValue
    conv_div_sin_f = convergence_rate(err_div_sin_f, Δh)
    # DivergenceC2F conv, with f(z) = cos(z), SetDivergence
    conv_div_cos_f = convergence_rate(err_div_cos_f, Δh)
    # CurlC2F with f(z) = sin(z), SetValue
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

    # GradientC2F conv, with f(z) = z, SetValue
    @test norm(err_grad_z_f) ≤ 200 * eps(FT)
    # Convergence rate for this case is noisy because error very small

    # GradientC2F conv, with f(z) = cos(z), SetValue
    @test err_grad_cos_f1[3] ≤ err_grad_cos_f1[2] ≤ err_grad_cos_f1[1] ≤ 0.1
    @test conv_grad_cos_f1[1] ≈ 1.5 atol = 0.1
    @test conv_grad_cos_f1[2] ≈ 1.5 atol = 0.1
    @test conv_grad_cos_f1[3] ≈ 1.5 atol = 0.1
    # @test conv_grad_cos_f1[1] ≤ conv_grad_cos_f1[2] ≤ conv_grad_cos_f1[3]

    # GradientC2F conv, with f(z) = cos(z), SetGradient
    @test err_grad_cos_f2[3] ≤ err_grad_cos_f2[2] ≤ err_grad_cos_f2[1] ≤ 0.1
    @test conv_grad_cos_f2[1] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[2] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[3] ≈ 2 atol = 0.1
    @test conv_grad_cos_f2[1] ≤ conv_grad_cos_f2[2] ≤ conv_grad_cos_f2[3]

    # DivergenceC2F conv, with f(z) = sin(z), SetValue
    @test err_div_sin_f[3] ≤ err_div_sin_f[2] ≤ err_div_sin_f[1] ≤ 0.1
    @test conv_div_sin_f[1] ≈ 2 atol = 0.1
    @test conv_div_sin_f[2] ≈ 2 atol = 0.1
    @test conv_div_sin_f[3] ≈ 2 atol = 0.1
    @test conv_div_sin_f[1] ≤ conv_div_sin_f[2] ≤ conv_div_sin_f[3]

    # DivergenceC2F conv, with f(z) = cos(z), SetDivergence
    @test err_div_cos_f[3] ≤ err_div_cos_f[2] ≤ err_div_cos_f[1] ≤ 0.1
    @test conv_div_cos_f[1] ≈ 2 atol = 0.1
    @test conv_div_cos_f[2] ≈ 2 atol = 0.1
    @test conv_div_cos_f[3] ≈ 2 atol = 0.1
    @test conv_div_cos_f[1] ≤ conv_div_cos_f[2] ≤ conv_div_cos_f[3]

    # CurlC2F with f(z) = sin(z), SetValue
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
    device = ClimaComms.device()

    Δh = zeros(FT, length(n_elems_seq))

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true, # not yet supported by CommonSpaces
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        ᶜspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        # Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(axes(ᶠz)))
        # c = sin(z), scalar field defined at the centers
        c = sin.(ᶜz)

        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F()
        third_order_fluxsinᶠ = third_order_fluxᶠ.(w, c)

        divf2c = Operators.DivergenceF2C()
        adv_wc = divf2c.(third_order_fluxsinᶠ)

        Δh[k] = ᶠJdata[vindex(1)]

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(ᶜz))
    end

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

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        ᶜspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        # Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # w = cos(z), vertical velocity field defined at the faces
        w = Geometry.WVector.(cos.(ᶠz))
        # c = sin(z), scalar field defined at the centers
        c = sin.(ᶜz)

        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F()
        third_order_fluxsinᶠ = third_order_fluxᶠ.(w, c)

        divf2c = Operators.DivergenceF2C()
        adv_wc = divf2c.(third_order_fluxsinᶠ)

        Δh[k] = ᶠJdata[vindex(1)]

        # Error
        err_adv_wc[k] =
            norm(adv_wc .- ((cos.(ᶜz)) .^ 2 .- (sin.(ᶜz)) .^ 2))
    end

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = cos(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 4e-3
    @test conv_adv_wc[1] ≈ 2 atol = 0.2
    @test conv_adv_wc[2] ≈ 2 atol = 0.1
    @test conv_adv_wc[3] ≈ 2 atol = 0.1
end

@testset "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on non-periodic mesh, with FirstOrderOneSided + DivergenceF2C SetValue BCs, constant w" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch = stretch_function(FT, parse_commandline(), 1.0)

    err_adv_wc = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    for (k, n) in enumerate(n_elems_seq)
        ᶜspace = ColumnSpace(FT;
            z_elem=n,
            z_min=-π,
            z_max=π,
            stretch,
            staggering=CellCenter()
        )
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        # Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(axes(ᶠz)))
        # c = sin(z), scalar field defined at the centers
        Δz = FT(2pi / n)
        c = (cos.(ᶜz .- Δz / 2) .- cos.(ᶜz .+ Δz / 2)) ./ Δz
        s = sin.(ᶜz)

        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.FirstOrderOneSided(),
            top = Operators.FirstOrderOneSided(),
        )

        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
            top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
        )

        adv_wc = divf2c.(third_order_fluxᶠ.(w, c))

        Δh[k] = ᶠJdata[vindex(1)]

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(ᶜz))
    end

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)
    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = 1
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 0.2006
    @test conv_adv_wc[1] ≈ 0.5 atol = 0.2
    @test conv_adv_wc[2] ≈ 0.5 atol = 0.3
    @test conv_adv_wc[3] ≈ 1.0 atol = 0.55
end

@testset "Upwind3rdOrderBiasedProductC2F + DivergenceF2C on non-periodic mesh, with ThirdOrderOneSided + DivergenceF2C SetValue BCs, varying sign w" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch = stretch_function(FT, parse_commandline(), 1.0)

    err_adv_wc = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    for (k, n) in enumerate(n_elems_seq)
        ᶜspace = ColumnSpace(FT;
            z_elem=n,
            z_min=-π,
            z_max=π,
            stretch,
            staggering=CellCenter()
        )
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        # Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # w = cos(z), vertical velocity field defined at the faces
        w = Geometry.WVector.(cos.(ᶠz))
        # c = sin(z), scalar field defined at the centers
        c = sin.(ᶜz)

        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.ThirdOrderOneSided(),
            top = Operators.ThirdOrderOneSided(),
        )

        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.WVector(FT(0.0))),
            top = Operators.SetValue(Geometry.WVector(FT(0.0))),
        )
        adv_wc = divf2c.(third_order_fluxᶠ.(w, c))

        Δh[k] = ᶠJdata[vindex(1)]
        # Errors
        err_adv_wc[k] =
            norm(adv_wc .- ((cos.(ᶜz)) .^ 2 .- (sin.(ᶜz)) .^ 2))

    end

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)
    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z), w(z) = cos(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 2e-1
    @test conv_adv_wc[1] ≈ 2 atol = 0.1
    @test conv_adv_wc[2] ≈ 2 atol = 0.1
    @test conv_adv_wc[3] ≈ 2 atol = 0.1
end

@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on (uniform) periodic mesh" begin
    FT = Float64
    n_elems_seq = 2 .^ (5, 6, 7, 8)

    err_adv_wc = zeros(FT, length(n_elems_seq))

    Δh = zeros(FT, length(n_elems_seq))
    device = ClimaComms.device()

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        ᶜspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        (; ᶠspace, ᶜz, ᶠz, ᶜJ, ᶠJ, ᶠJdata, ᶜJdata) = coord_info(ᶜspace)

        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(axes(ᶠz)))
        # c = sin(z), scalar field defined at the centers
        c = sin.(ᶜz)

        first_order_fluxᶠ = Operators.UpwindBiasedProductC2F()
        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F()
        first_order_fluxsinᶠ = first_order_fluxᶠ.(w, c)
        third_order_fluxsinᶠ = third_order_fluxᶠ.(w, c)

        divf2c = Operators.DivergenceF2C()
        corrected_antidiff_flux =
            @. divf2c(C * (third_order_fluxsinᶠ - first_order_fluxsinᶠ))
        adv_wc = @. divf2c.(first_order_fluxsinᶠ) + corrected_antidiff_flux

        Δh[k] = ᶠJdata[vindex(1)]

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(ᶜz))
    end

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

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.FaceFiniteDifferenceSpace(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        SLMethod = Operators.LinVanLeerC2F(
            bottom = Operators.FirstOrderOneSided(),
            top = Operators.FirstOrderOneSided(),
            constraint = Operators.MonotoneLocalExtrema(),
        )

        divf2c = Operators.DivergenceF2C()
        flux = SLMethod.(w, c, FT(0))
        adv_wc = divf2c.(flux)

        Δh[k] = Spaces.local_geometry_data(fs).J[vindex(1)]

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
    end

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

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.FaceFiniteDifferenceSpace(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        SLMethod = Operators.LinVanLeerC2F(;
            bottom = Operators.FirstOrderOneSided(),
            top = Operators.FirstOrderOneSided(),
            constraint = Operators.MonotoneHarmonic(),
        )

        divf2c = Operators.DivergenceF2C()
        flux = SLMethod.(w, c, FT(0))
        adv_wc = divf2c.(flux)

        Δh[k] = Spaces.local_geometry_data(fs).J[vindex(1)]

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
    end

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

    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            periodic = true,
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.FaceFiniteDifferenceSpace(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        SLMethod = Operators.LinVanLeerC2F(
            bottom = Operators.FirstOrderOneSided(),
            top = Operators.FirstOrderOneSided(),
            constraint = Operators.PositiveDefinite(),
        )

        divf2c = Operators.DivergenceF2C()
        flux = SLMethod.(w, c, FT(0))
        adv_wc = divf2c.(flux)

        Δh[k] = Spaces.local_geometry_data(fs).J[vindex(1)]

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
    end

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)

    # LinVanLeer limited flux conv, with f(z) = sin(z)
    @test conv_adv_wc[1] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[2] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[3] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[4] ≈ 1.0 atol = 0.01
    @test conv_adv_wc[5] ≈ 1.0 atol = 0.01

end

@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on non-periodic mesh, with FirstOrderOneSided BCs" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch = stretch_function(FT, parse_commandline(), 1.0)
    device = ClimaComms.device()

    err_adv_wc = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            boundary_names = (:bottom, :top),
        )
        mesh = Meshes.IntervalMesh(domain, stretch; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.FaceFiniteDifferenceSpace(cs)

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
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.FirstOrderOneSided(),
            top = Operators.FirstOrderOneSided(),
        )

        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
            top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
        )

        corrected_antidiff_flux =
            @. divf2c(C * (third_order_fluxᶠ(w, c) - first_order_fluxᶠ(w, c)))
        adv_wc = @. divf2c.(first_order_fluxᶠ(w, c)) + corrected_antidiff_flux

        Δh[k] = Spaces.local_geometry_data(fs).J[vindex(1)]

        # Error
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))
    end

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)
    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 0.2006
    @test conv_adv_wc[1] ≈ 0.5 atol = 0.2
    @test conv_adv_wc[2] ≈ 0.5 atol = 0.3
    @test conv_adv_wc[3] ≈ 1.0 atol = 0.55
end

@testset "Simple FCT: lin combination of UpwindBiasedProductC2F + Upwind3rdOrderBiasedProductC2F on non-periodic mesh, with ThirdOrderOneSided BCs" begin
    FT = Float64
    n_elems_seq = 2 .^ (4, 6, 8, 10)
    stretch = stretch_function(FT, parse_commandline(), 1.0)
    device = ClimaComms.device()

    err_adv_wc = zeros(FT, length(n_elems_seq))
    Δh = zeros(FT, length(n_elems_seq))
    for (k, n) in enumerate(n_elems_seq)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(-pi),
            Geometry.ZPoint{FT}(pi);
            boundary_names = (:bottom, :top),
        )
        mesh = Meshes.IntervalMesh(domain; nelems = n)

        cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
        fs = Spaces.FaceFiniteDifferenceSpace(cs)

        centers = getproperty(Fields.coordinate_field(cs), :z)
        C = FT(1.0) # flux-correction coefficient (falling back to third-order upwinding)

        # UpwindBiasedProductC2F & Upwind3rdOrderBiasedProductC2F Center -> Face operator
        # Unitary, constant advective velocity
        w = Geometry.WVector.(ones(fs))
        # c = sin(z), scalar field defined at the centers
        c = sin.(centers)

        first_order_fluxᶠ = Operators.UpwindBiasedProductC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.ThirdOrderOneSided(),
            top = Operators.ThirdOrderOneSided(),
        )

        divf2c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.WVector(FT(0.0))),
            top = Operators.SetValue(Geometry.WVector(FT(0.0))),
        )
        corrected_antidiff_flux =
            @. divf2c(C * (third_order_fluxᶠ(w, c) - first_order_fluxᶠ(w, c)))
        adv_wc = @. divf2c.(first_order_fluxᶠ(w, c)) + corrected_antidiff_flux

        Δh[k] = Spaces.local_geometry_data(fs).J[vindex(1)]
        # Errors
        err_adv_wc[k] = norm(adv_wc .- cos.(centers))

    end

    # Check convergence rate
    conv_adv_wc = convergence_rate(err_adv_wc, Δh)
    # Upwind3rdOrderBiasedProductC2F conv, with f(z) = sin(z)
    @test err_adv_wc[3] ≤ err_adv_wc[2] ≤ err_adv_wc[1] ≤ 5e-1
    @test conv_adv_wc[1] ≈ 2.5 atol = 0.1
    @test conv_adv_wc[2] ≈ 2.5 atol = 0.1
    @test conv_adv_wc[3] ≈ 2.5 atol = 0.1
end

@testset "Center -> Center Advection" begin

    function advection(c, f, cs)
        adv = zeros(eltype(f), cs)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(0.0),
            top = Operators.Extrapolate(),
        )
        return @. adv = A(c, f)
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
        fs = Spaces.FaceFiniteDifferenceSpace(cs)

        # advective velocity
        c = Geometry.WVector.(ones(Float64, fs),)
        # scalar-valued field to be advected
        f = sin.(Fields.coordinate_field(cs).z)

        # Call the advection operator
        adv = advection(c, f, cs)

        Δh[k] = Spaces.local_geometry_data(fs).J[vindex(1)]
        err[k] = norm(adv .- cos.(Fields.coordinate_field(cs).z))
    end
    # AdvectionC2C convergence rate
    conv_adv_c2c = convergence_rate(err, Δh)
    @test err[3] ≤ err[2] ≤ err[1] ≤ 0.1
    @test conv_adv_c2c[1] ≈ 2 atol = 0.1
    @test conv_adv_c2c[2] ≈ 2 atol = 0.1
    @test conv_adv_c2c[3] ≈ 2 atol = 0.1
end
