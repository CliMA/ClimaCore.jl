import CUDA, Adapt
import ClimaComms
import ClimaCore
import ClimaCore: DataLayouts, Grids, Meshes, Topologies

Adapt.adapt_storage(
    ::CUDA.KernelAdaptor,
    ::ClimaComms.AbstractCommsContext,
) = ClimaCore.DeviceSideContext()

Adapt.adapt_storage(::CUDA.KernelAdaptor, ::Meshes.AbstractMesh) = nothing
Adapt.adapt_structure(::CUDA.KernelAdaptor, ::Topologies.Topology2D) = nothing

# Kernels receive the immutable device twin of each grid (see
# `Grids.@host_device_struct`); the spectral element grids also drop the DSS
# weights, surface geometries and mask, which kernels do not use.
Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.HostSpectralElementGrid1D,
) = Grids.DeviceSpectralElementGrid1D(
    Adapt.adapt(to, grid.topology),
    Adapt.adapt(to, grid.quadrature_style),
    Adapt.adapt(to, grid.global_geometry),
    Adapt.adapt(to, grid.local_geometry),
    nothing, # dss_weights
    grid.discretization,
)

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.HostSpectralElementGrid2D,
) = Grids.DeviceSpectralElementGrid2D(
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

# One method per grid to be more specific in dispatch than the adapt
# methods `@host_device_struct` defines
Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.HostFiniteDifferenceGrid,
) = Grids.device_twin(to, grid)
Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.HostExtrudedFiniteDifferenceGrid,
) = Grids.device_twin(to, grid)

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
