# Baroclinic wave (Ullrich et al., 2014) with the horizontal spectral
# operators of two tendencies swapped for single-kernel cuTile versions
# (`benchmarks/cutile/fused_kernels_cutile.jl`), for timing them inside a full
# simulation rather than in isolation:
#
#   PGRAD      the pressure-gradient force  uₜ -= grad(p)/ρ + grad(K + Φ)
#   HYPERDIFF  the two scalar Laplacian passes of the energy hyperdiffusion,
#              χ = ∇²((ρe + p)/ρ)  and  ρeₜ -= κ₄ ∇·(ρ ∇χ)
#
# Each is `fused` (the ClimaCore broadcast) or `cutile` (one cuTile kernel
# with the pointwise work fused around per-element GEMMs). HYPERDIFF defaults
# to the value of PGRAD, so `PGRAD=fused` is the pure baseline and
# `PGRAD=cutile` moves both tendencies. Momentum hyperdiffusion and every
# other operator stay on ClimaCore in all configurations, so the walltimes
# printed by driver.jl are directly comparable.
#
# Requires the `benchmarks/cutile` environment (Julia 1.11, CUDA 13 driver;
# see benchmarks/cutile/README.md):
#
#     export CLIMACOMMS_DEVICE=CUDA
#     export TEST_NAME=sphere/baroclinic_wave_rhoe_cutile
#     PGRAD=fused  julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
#     PGRAD=cutile julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
#     PGRAD=cutile HYPERDIFF=fused julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
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
const hyperdiff_name = get(ENV, "HYPERDIFF", pgrad_name)
for (var, name) in (("PGRAD", pgrad_name), ("HYPERDIFF", hyperdiff_name))
    name in ("cutile", "fused") ||
        error("$var must be \"cutile\" or \"fused\"; got $(repr(name))")
end
const use_cutile = pgrad_name == "cutile" || hyperdiff_name == "cutile"

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

if use_cutile
    import CUDA
    include(
        joinpath(@__DIR__, "../../../benchmarks/cutile/fused_kernels_cutile.jl"),
    )

    VERSION >= v"1.11" || error(
        "cuTile requires Julia >= 1.11 (this is $VERSION); use `julia +1.11`.",
    )
    CUDA.functional() ||
        error("the cutile schemes require a GPU node with CLIMACOMMS_DEVICE=CUDA.")
    CUDA.driver_version() >= v"13" || error(
        "cuTile requires an NVIDIA driver supporting CUDA 13 (driver >= 580).",
    )
    CUDA.capability(CUDA.device()) >= v"8.0" ||
        error("cuTile requires compute capability >= 8.0 (Ampere+).")

    # Pressure-gradient force, in place on the momentum tendency.
    pressure_gradient_tendency!(Yₜ, ᶜρ, ᶜp, ᶜK, ᶜΦ, s::CuTileSpectral) =
        cutile_pressure_gradient!(Yₜ.c.uₕ, ᶜp, ᶜK, ᶜΦ, ᶜρ, s)

    # Energy hyperdiffusion: χ = ∇²((ρe + p) / ρ) with the enthalpy as the
    # kernel prologue, then ρeₜ -= κ₄ ∇·(ρ ∇χ) accumulated in place.
    scalar_hyperdiffusion_first_pass!(ᶜχ, Y, ᶜp, s::CuTileSpectral) =
        cutile_scalar_laplacian!(ᶜχ, Y.c.ρe, s; energy = (ᶜp, Y.c.ρ))
    scalar_hyperdiffusion_second_pass!(Yₜ, ᶜχ, ᶜρ, κ₄, s::CuTileSpectral) =
        cutile_scalar_laplacian!(
            Yₜ.c.ρe,
            ᶜχ,
            s;
            weight = ᶜρ,
            scale = -κ₄,
            accumulate = true,
        )

    # One-time on-device check of every kernel configuration used above
    # against ClimaCore's operators on the same GPU (also warms them up).
    # Operands carry production-like node-constant offsets, whose exact
    # horizontal gradient is zero, so the gate has the rounding floor
    # eps · ‖D‖∞ · ‖operand‖ of that cancellation besides rtol · ‖result‖.
    function check_cutile_spectral(s::CuTileSpectral, space)
        maxabs(x) = maximum(abs, parent(Fields.field_values(x)))
        D = Quadratures.differentiation_matrix(FT, Spaces.quadrature_style(space))
        Dnorm = maximum(sum(abs, Matrix(D); dims = 2))
        rtol = FT == Float64 ? 1e-12 : 1e-4
        function gate!(name, result, oracle, operand_scale)
            scale = max(maxabs(oracle), floatmin(FT))
            gate = max(8 * eps(FT) * Dnorm * operand_scale, rtol * scale)
            err = maximum(
                abs.(
                    parent(Fields.field_values(result)) .-
                    parent(Fields.field_values(oracle)),
                ),
            )
            err <= gate || error(
                "cuTile $name disagrees with ClimaCore: max abs err $err, gate $gate",
            )
            return nothing
        end

        coords = Fields.coordinate_field(space)
        wdiv = Operators.Divergence{Operators.WeakForm}()
        grad = Operators.Gradient()
        p = @. FT(1e5) * (1 + FT(0.1) * sind(coords.long) * cosd(coords.lat))
        ρ = @. 1 + FT(0.05) * cosd(coords.lat) + coords.z / z_max
        Kin = @. FT(100) * sind(coords.long)^2
        Φ = @. grav * coords.z
        χ = @. sind(coords.long) * cosd(coords.lat) * (1 + coords.z / z_max)
        ρe = @. ρ * (FT(2e5) + FT(1e4) * cosd(2 * coords.long) * sind(coords.lat))

        uₜ0 = @. Geometry.Covariant12Vector(FT(0.25) * cosd(coords.lat), FT(-0.75))
        uₜ_ref = @. uₜ0 - Geometry.Covariant12Vector(grad(p) / ρ + grad(Kin + Φ))
        uₜ = copy(uₜ0)
        cutile_pressure_gradient!(uₜ, p, Kin, Φ, ρ, s)
        CUDA.synchronize()
        gate!(
            "pressure gradient",
            uₜ,
            uₜ_ref,
            maxabs(p) / minimum(parent(ρ)) + maxabs(Kin) + maxabs(Φ),
        )

        h_tot = @. (ρe + p) / ρ
        χ1_ref = @. wdiv(grad(h_tot))
        χ1 = similar(χ)
        cutile_scalar_laplacian!(χ1, ρe, s; energy = (p, ρ))
        CUDA.synchronize()
        lap_gain = maxabs(χ1_ref) / max(maxabs(@. grad(h_tot)), eps(FT))
        gate!("enthalpy Laplacian", χ1, χ1_ref, lap_gain * maxabs(h_tot) + maxabs(χ1_ref))

        κ = FT(3e15)
        acc0 = @. FT(0.5) * sind(3 * coords.long)
        acc_ref = @. acc0 - κ * wdiv(ρ * grad(χ))
        acc = copy(acc0)
        cutile_scalar_laplacian!(acc, χ, s; weight = ρ, scale = -κ, accumulate = true)
        CUDA.synchronize()
        lap_gain = maxabs(acc_ref) / max(maxabs(@. grad(χ)), eps(FT))
        gate!("weighted Laplacian", acc, acc_ref, lap_gain * maxabs(χ) + maxabs(acc_ref))

        @info "cuTile spectral kernels verified against ClimaCore" s.tv s.mtv s.Nv
        return nothing
    end

    function cutile_spectral_cache(ᶜlocal_geometry)
        space = axes(ᶜlocal_geometry)
        s = CuTileSpectral(space; tv = parse(Int, get(ENV, "TV", "64")))
        check_cutile_spectral(s, space)
        return s
    end
end

function additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt)
    cache = hyperdiffusion_cache(ᶜlocal_geometry; κ₄)
    use_cutile || return cache
    spectral = cutile_spectral_cache(ᶜlocal_geometry)
    pgrad_name == "cutile" && (cache = merge(cache, (; pgrad_scheme = spectral)))
    hyperdiff_name == "cutile" &&
        (cache = merge(cache, (; hyperdiff_scheme = spectral)))
    return cache
end
additional_tendency!(Yₜ, Y, p, t) = hyperdiffusion_tendency!(Yₜ, Y, p, t)

center_initial_condition(local_geometry) =
    sphere_center_initial_condition(local_geometry)

function postprocessing(sol, output_dir)
    # No plots: the point of this case is the walltime printed by driver.jl.
    # The norms let the configurations be compared.
    @info "PGRAD = $pgrad_name, HYPERDIFF = $hyperdiff_name"
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"
    v_end = maximum(abs, Geometry.UVVector.(sol.u[end].c.uₕ).components.data.:2)
    @info "max |v| at t = $(sol.t[end]): $v_end"
end
