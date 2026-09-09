import CUDA, Adapt
import ClimaComms
import ClimaCore
import ClimaCore: DataLayouts, Grids, Meshes, Topologies

Adapt.adapt_structure(
    ::CUDA.KernelAdaptor,
    ::ClimaComms.AbstractCommsContext,
) = ClimaCore.DeviceSideContext()

Adapt.adapt_structure(::CUDA.KernelAdaptor, ::Meshes.AbstractMesh) = nothing

Adapt.adapt_structure(::CUDA.KernelAdaptor, ::Topologies.Topology2D) = nothing

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.SpectralElementGrid1D,
) = Grids.SpectralElementGrid1D(
    Adapt.adapt(to, grid.topology),
    Adapt.adapt(to, grid.quadrature_style),
    Adapt.adapt(to, grid.global_geometry),
    Adapt.adapt(to, grid.local_geometry),
    nothing, # dss_weights
    grid.discretization,
)

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.SpectralElementGrid2D,
) = Grids.SpectralElementGrid2D(
    Adapt.adapt(to, grid.topology),
    Adapt.adapt(to, grid.quadrature_style),
    Adapt.adapt(to, grid.global_geometry),
    Adapt.adapt(to, grid.local_geometry),
    nothing, # dss_weights
    nothing, # interior_surface_geometry
    nothing, # boundary_surface_geometries
    DataLayouts.NoMask(), # mask
    grid.enable_bubble,
    grid.autodiff_metric,
    grid.discretization,
)

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    lim::Limiters.QuasiMonotoneLimiter,
) = Limiters.QuasiMonotoneLimiter(
    Adapt.adapt(to, lim.q_bounds),
    Adapt.adapt(to, lim.q_bounds_nbr),
    Adapt.adapt(to, lim.ghost_buffer),
    lim.rtol,
    Limiters.NoConvergenceStats(),
)
