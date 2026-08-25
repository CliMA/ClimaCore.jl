# TOPO=earth: real Earth orography from ETOPO2022 (60 arc-second), regridded
# onto the cubed sphere with ClimaUtilities.SpaceVaryingInput (the GPU-capable
# InterpolationsRegridder — bilinear), Laplacian-smoothed via ClimaCore
# Hypsography, and with the ocean clamped to sea level. Isolated here and
# `include`d only when TOPO=earth (sphere_dg_fd_model.jl) so that flat/hj runs
# don't load ClimaUtilities / Interpolations / NCDatasets or touch the artifact.
#
# Requires (added to .buildkite): ClimaUtilities, Interpolations. The
# earth_orography_60arcseconds artifact is declared in the repo-root
# Artifacts.toml and lazily downloaded on first use.
import ClimaUtilities.SpaceVaryingInputs: SpaceVaryingInput
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import Interpolations   # activates ClimaUtilities' InterpolationsRegridder ext
import NCDatasets        # activates ClimaUtilities' NetCDF reader ext
import LazyArtifacts     # enables on-demand (lazy) artifact download

# ETOPO_DAMPING sets the number of Laplacian smoothing iterations via
# maxiter = round(log(damping) / CFL) with a diffusion Courant number CFL =
# 0.05 (κ = CFL·Δx²). Larger damping ⇒ smoother terrain. Real orography MUST be
# smoothed or the spectral-element grid Gibbs-oscillates into negative layer
# thicknesses.
const etopo_damping = parse(FT, get(ENV, "ETOPO_DAMPING", "8"))

"""
    earth_z_surface(horzspace) -> Field

ETOPO2022 surface elevation (m) on `horzspace`: regrid → smooth → clamp ocean
to 0. Returns a scalar `Field`; the caller wraps it in `Geometry.ZPoint`s for
`Hypsography.LinearAdaption`.
"""
function earth_z_surface(horzspace)
    context = ClimaComms.context(horzspace)
    # The artifact contains a single NetCDF (currently
    # ETOPO_2022_v1_60s_N90W180_surface.nc); glob it rather than hardcode the
    # name (it has varied across artifact versions).
    dir = @clima_artifact("earth_orography_60arcseconds", context)
    etopo = joinpath(dir, only(filter(f -> endswith(f, ".nc"), readdir(dir))))
    z_surface = SpaceVaryingInput(
        etopo,
        "z",
        horzspace;
        regridder_type = :InterpolationsRegridder,
    )
    diff_courant = parse(FT, get(ENV, "ETOPO_CFL", "0.05"))
    # node_horizontal_length_scale returns the ELEMENT scale; divide by npoly
    # for the true node spacing. Using the element scale makes the explicit
    # diffusion CFL ~npoly² too large (unstable — the smoother blows up) and
    # over-smooths by ~npoly². Total smoothing = log(damping)·Δx² is CFL-independent.
    Δx = Spaces.node_horizontal_length_scale(horzspace) / npoly
    κ = diff_courant * Δx^2
    maxiter = Int(round(log(etopo_damping) / diff_courant))
    Hypsography.diffuse_surface_elevation!(z_surface; κ, dt = FT(1), maxiter)
    @. z_surface = max(z_surface, FT(0))   # ocean → sea level
    return z_surface
end
