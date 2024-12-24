import Adapt
import ClimaCore.DataLayouts: ToCUDA, ToCPU

function Adapt.adapt(to::ToCUDA, space::Spaces.FiniteDifferenceSpace)
    Spaces.FiniteDifferenceSpace(
        Adapt.adapt(CUDA.CuArray, Spaces.grid(space)),
        space.staggering,
    )
end

function Adapt.adapt(to::ToCUDA, space::Spaces.SpectralElementSpace1D)
    Spaces.SpectralElementSpace1D(Adapt.adapt(CUDA.CuArray, Spaces.grid(space)))
end

function Adapt.adapt(to::ToCUDA, space::Spaces.SpectralElementSpace2D)
    Spaces.SpectralElementSpace2D(Adapt.adapt(CUDA.CuArray, Spaces.grid(space)))
end

function Adapt.adapt(to::ToCUDA, space::Spaces.SpectralElementSpaceSlab)
    Spaces.SpectralElementSpaceSlab(
        space.quadrature_style,
        Adapt.adapt(CUDA.CuArray, space.local_geometry),
    )
end

function Adapt.adapt(::ToCUDA, context::ClimaComms.AbstractCommsContext)
    return context(ClimaComms.CUDADevice())
end

function Adapt.adapt(to::ToCUDA, space::Grids.LevelGrid)
    return Grids.LevelGrid(
        Adapt.adapt(CUDA.CuArray, space.full_grid),
        Adapt.adapt(CUDA.CuArray, space.level),
    )
end

function Adapt.adapt(to::ToCUDA, space::Grids.SpectralElementGrid1D)
    return Grids.SpectralElementGrid1D(
        Adapt.adapt(CUDA.CuArray, space.topology),
        Adapt.adapt(CUDA.CuArray, space.quadrature_style),
        Adapt.adapt(CUDA.CuArray, space.global_geometry),
        Adapt.adapt(CUDA.CuArray, space.local_geometry),
        Adapt.adapt(CUDA.CuArray, space.dss_weights),
    )
end

function Adapt.adapt(::ToCUDA, space::Grids.SpectralElementGrid2D)
    return Grids.SpectralElementGrid2D(
        Adapt.adapt(CUDA.CuArray, space.topology),
        Adapt.adapt(CUDA.CuArray, space.quadrature_style),
        Adapt.adapt(CUDA.CuArray, space.global_geometry),
        Adapt.adapt(CUDA.CuArray, space.local_geometry),
        Adapt.adapt(CUDA.CuArray, space.local_dss_weights),
        Adapt.adapt(CUDA.CuArray, space.internal_surface_geometry),
        Adapt.adapt(CUDA.CuArray, space.boundary_surface_geometries),
        space.enable_bubble,
    )
end

function Adapt.adapt(to::ToCUDA, grid::Grids.FiniteDifferenceGrid)
    return Grids.FiniteDifferenceGrid(
        Adapt.adapt(CUDA.CuArray, grid.topology),
        Adapt.adapt(CUDA.CuArray, grid.global_geometry),
        Adapt.adapt(CUDA.CuArray, grid.center_local_geometry),
        Adapt.adapt(CUDA.CuArray, grid.face_local_geometry),
    )
end
function Adapt.adapt(to::ToCUDA, grid::Grids.ExtrudedFiniteDifferenceGrid)
    return Grids.ExtrudedFiniteDifferenceGrid(
        Adapt.adapt(CUDA.CuArray, grid.horizontal_grid),
        Adapt.adapt(CUDA.CuArray, grid.vertical_grid),
        Adapt.adapt(CUDA.CuArray, grid.hypsography),
        Adapt.adapt(CUDA.CuArray, grid.global_geometry),
        Adapt.adapt(CUDA.CuArray, grid.center_local_geometry),
        Adapt.adapt(CUDA.CuArray, grid.face_local_geometry),
    )
end

function Adapt.adapt(to::ToCUDA, data::DataLayouts.AbstractData)
    DataLayouts.union_all(DataLayouts.singleton(data))(
        Adapt.adapt(CUDA.CuArray, parent(data)),
    )
end
