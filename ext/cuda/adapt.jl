import CUDA, Adapt
import ClimaCore
import ClimaCore: Grids, Spaces, Topologies

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.ExtrudedFiniteDifferenceGrid,
) = Grids.DeviceExtrudedFiniteDifferenceGrid(
    Adapt.adapt(to, Grids.vertical_topology(grid)),
    Adapt.adapt(to, grid.horizontal_grid.quadrature_style),
    Adapt.adapt(to, grid.global_geometry),
    Adapt.adapt(to, grid.center_local_geometry),
    Adapt.adapt(to, grid.face_local_geometry),
)

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.FiniteDifferenceGrid,
) = Grids.DeviceFiniteDifferenceGrid(
    Adapt.adapt(to, grid.topology),
    Adapt.adapt(to, grid.global_geometry),
    Adapt.adapt(to, grid.center_local_geometry),
    Adapt.adapt(to, grid.face_local_geometry),
)

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    grid::Grids.SpectralElementGrid2D,
) = Grids.DeviceSpectralElementGrid2D(
    Adapt.adapt(to, grid.quadrature_style),
    Adapt.adapt(to, grid.global_geometry),
    Adapt.adapt(to, grid.local_geometry),
)

Adapt.adapt_structure(to::CUDA.KernelAdaptor, space::Spaces.PointSpace) =
    Spaces.PointSpace(
        ClimaCore.DeviceSideContext(),
        Adapt.adapt(to, Spaces.local_geometry_data(space)),
    )

Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    topology::Topologies.IntervalTopology,
) = Topologies.DeviceIntervalTopology(topology.boundaries)

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
