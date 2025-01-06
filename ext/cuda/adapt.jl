import Adapt
import CUDA
import ClimaCore: Grids

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    device::ClimaComms.CPUSingleThreaded,
) = ClimaComms.CUDADevice()

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    device::ClimaComms.CPUMultiThreaded,
) = ClimaComms.CUDADevice()

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    context::ClimaComms.SingletonCommsContext,
) = ClimaComms.context(Adapt.adapt(to, context.device))

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.SpectralElementGrid2D,
) = Grids.DeviceSpectralElementGrid2D(
    Adapt.adapt(to, grid.topology),
    Adapt.adapt(to, grid.quadrature_style),
    Adapt.adapt(to, grid.global_geometry),
    Adapt.adapt(to, grid.local_geometry),
    Adapt.adapt(to, grid.local_dss_weights),
    Adapt.adapt(to, grid.internal_surface_geometry),
    Adapt.adapt(to, grid.boundary_surface_geometries),
    Adapt.adapt(to, grid.enable_bubble),
)

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    topology::Topologies.IntervalTopology,
) = IntervalTopology(
    Adapt.adapt_structure(to, topology.context),
    Adapt.adapt_structure(to, topology.mesh),
    Adapt.adapt_structure(to, topology.boundaries),
)
