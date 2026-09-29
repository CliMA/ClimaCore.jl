# Baroclinic wave (Ullrich et al., 2014) with the horizontal pressure-gradient
# force computed by the cuTile KronGEMM kernel from `benchmarks/cutile`, for
# timing the kernel inside a full simulation rather than in isolation. The
# fused ClimaCore baseline runs from the same file with `PGRAD=fused`, so an
# A/B comparison shares one environment and configuration.
#
# Requires the `benchmarks/cutile` environment (Julia 1.11, CUDA 13 driver;
# see benchmarks/cutile/README.md):
#
#     export CLIMACOMMS_DEVICE=CUDA
#     export TEST_NAME=sphere/baroclinic_wave_rhoe_cutile
#     PGRAD=cutile julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
#     PGRAD=fused  julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
#
# Configuration (env): H_ELEM (30), Z_ELEM (63), DT, T_END (21600 s), KAPPA_4,
# TV (cuTile vertical tile size), FLOAT_TYPE (Float32).
#
# `npoly` is fixed at 3 (Nq = 4): cuTile tile extents must be powers of two,
# so Nq² must be a power of two. This matches the microbenchmark configuration.
using Test
using ClimaCore.DataLayouts

include("baroclinic_wave_utils.jl")

const pgrad_name = get(ENV, "PGRAD", "cutile")
pgrad_name in ("cutile", "fused") ||
    error("PGRAD must be \"cutile\" or \"fused\"; got $(repr(pgrad_name))")

# Variables required for driver.jl
h_elem = parse(Int, get(ENV, "H_ELEM", "30"))
horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = h_elem)
npoly = 3
z_max = FT(30e3)
z_elem = parse(Int, get(ENV, "Z_ELEM", "63"))
t_end = FT(parse(Float64, get(ENV, "T_END", "21600")))
# dt and κ₄ are tuned for h_elem = 4 in `baroclinic_wave_rhoe.jl`; scale them
# to the (finer) resolutions this benchmark runs at.
dt = FT(parse(Float64, get(ENV, "DT", string(400 * 4 / h_elem))))
const κ₄ = FT(parse(Float64, get(ENV, "KAPPA_4", string(2e17 * (4 / h_elem)^3))))
dt_save_to_sol = t_end
dt_save_to_disk = FT(0)
ode_algorithm = CTS.SSP333
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

if pgrad_name == "cutile"
    import CUDA
    include(joinpath(@__DIR__, "../../../benchmarks/cutile/gradient_kernels.jl"))
    include(
        joinpath(@__DIR__, "../../../benchmarks/cutile/gradient_kernels_cutile.jl"),
    )

    VERSION >= v"1.11" || error(
        "cuTile requires Julia >= 1.11 (this is $VERSION); use `julia +1.11`.",
    )
    CUDA.functional() ||
        error("PGRAD=cutile requires a GPU node with CLIMACOMMS_DEVICE=CUDA.")
    CUDA.driver_version() >= v"13" || error(
        "cuTile requires an NVIDIA driver supporting CUDA 13 (driver >= 580).",
    )
    CUDA.capability(CUDA.device()) >= v"8.0" ||
        error("cuTile requires compute capability >= 8.0 (Ampere+).")

    struct CuTilePressureGradient{W, V, S}
        Wt::W
        ∇p::V
        ∇b::V
        bernoulli::S
        tv::Int
        Kq::Int
        Cq::Int
    end

    # Strong scalar gradient via the KronGEMM kernel; `dest` is a
    # Covariant12Vector field, `src` a scalar field on the same space.
    function cutile_scalar_grad!(dest, src, s::CuTilePressureGradient)
        pf = parent(Fields.field_values(src))
        po = parent(Fields.field_values(dest))
        (Nv, _, _, _, Nh) = size(pf)
        launch_grad_cutile!(
            reshape(pf, Nv, s.Kq, Nh),
            s.Wt,
            reshape(po, Nv, s.Cq, Nh);
            tv = s.tv,
            K = s.Kq,
            C = s.Cq,
        )
        return nothing
    end

    function pressure_gradient_tendency!(
        Yₜ,
        ᶜρ,
        ᶜp,
        ᶜK,
        ᶜΦ,
        s::CuTilePressureGradient,
    )
        (; ∇p, ∇b, bernoulli) = s
        @. bernoulli = ᶜK + ᶜΦ
        cutile_scalar_grad!(∇p, ᶜp, s)
        cutile_scalar_grad!(∇b, bernoulli, s)
        @. Yₜ.c.uₕ -= ∇p / ᶜρ + ∇b
        return nothing
    end

    function cutile_pgrad_scheme(ᶜlocal_geometry)
        space = axes(ᶜlocal_geometry)
        quad = Spaces.quadrature_style(space)
        Nq = Quadratures.degrees_of_freedom(quad)
        Kq = Nq * Nq
        Cq = 2 * Kq
        ispow2(Kq) || error("cuTile tile extents must be powers of two; \
                             Nq² = $Kq. Use npoly = 3.")
        D = Quadratures.differentiation_matrix(FT, quad)
        Wt = CUDA.CuArray(gradient_weight(D))
        ∇p = similar(ᶜlocal_geometry, Geometry.Covariant12Vector{FT})
        Nv = size(parent(Fields.field_values(∇p)), 1)
        tv = min(parse(Int, get(ENV, "TV", "64")), nextpow(2, Nv))
        ispow2(tv) || error("TV must be a power of two; got $tv")
        s = CuTilePressureGradient(
            Wt,
            ∇p,
            similar(∇p),
            similar(ᶜlocal_geometry, FT),
            tv,
            Kq,
            Cq,
        )

        # One-time on-device check against ClimaCore's Gradient (also warms up
        # the kernel). Both run on the GPU, so they differ only by contraction
        # order; a weight-ordering or masked-store bug is O(1) wrong.
        coords = Fields.coordinate_field(space)
        χ = @. sind(coords.long) * cosd(coords.lat) * (1 + coords.z / z_max)
        ref = @. gradₕ(χ)
        out = similar(ref)
        cutile_scalar_grad!(out, χ, s)
        CUDA.synchronize()
        maxerr = maximum(
            abs.(
                parent(Fields.field_values(out)) .-
                parent(Fields.field_values(ref))
            ),
        )
        scale = maximum(abs, parent(Fields.field_values(ref)))
        rtol = FT == Float64 ? 1e-11 : 1e-3
        maxerr <= rtol * scale || error(
            "cuTile gradient disagrees with ClimaCore Gradient: \
             max abs err $maxerr, gate $(rtol * scale)",
        )
        @info "cuTile pressure gradient enabled" Nq Nv tv maxerr
        return s
    end
end

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry; κ₄),
    pgrad_name == "cutile" ?
    (; pgrad_scheme = cutile_pgrad_scheme(ᶜlocal_geometry)) : (;),
)
additional_tendency!(Yₜ, Y, p, t) = hyperdiffusion_tendency!(Yₜ, Y, p, t)

center_initial_condition(local_geometry) =
    sphere_center_initial_condition(local_geometry)

function postprocessing(sol, output_dir)
    # No plots: the point of this case is the walltime printed by driver.jl.
    # The norms let the PGRAD=fused and PGRAD=cutile runs be compared.
    @info "PGRAD = $pgrad_name"
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"
    v_end = maximum(abs, Geometry.UVVector.(sol.u[end].c.uₕ).components.data.:2)
    @info "max |v| at t = $(sol.t[end]): $v_end"
end
