#=
While we recommend users to use our composable
methods, this file contains some convenience
constructors, which are meant to improve
developer experience.
=#

"""
    ExtrudedCubedSphereGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer = 10,
        z_min::Real = FT(0),
        z_max::Real = FT(1),
        boundary_names = (:bottom, :top),
        device = ClimaComms.device(),
        context = ClimaComms.context(device),
        stretch = Meshes.Uniform(),
        hypsography = Flat(),
        radius::Real = FT(6.371229e6),
        global_geometry = Geometry.ShallowSphericalGlobalGeometry(radius),
        h_elem::Integer = 4,
        Nq::Integer = 4,
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
        h_domain = Domains.SphereDomain(radius),
        h_mesh = Meshes.EquiangularCubedSphere(h_domain, h_elem)
    )

A convenience constructor, which builds a
`ExtrudedFiniteDifferenceGrid`.

All arguments are optional.
"""
ExtrudedCubedSphereGrid(; kwargs...) =
    ExtrudedCubedSphereGrid(Float64; kwargs...)

function ExtrudedCubedSphereGrid(
    ::Type{FT};
    z_elem::Integer = 10,
    z_min::Real = FT(0),
    z_max::Real = FT(1),
    boundary_names = (:bottom, :top),
    device = ClimaComms.device(),
    context = ClimaComms.context(device),
    stretch = Meshes.Uniform(),
    hypsography = Flat(),
    radius::Real = FT(6.371229e6),
    global_geometry = Geometry.ShallowSphericalGlobalGeometry(radius),
    h_elem::Integer = 4,
    Nq::Integer = 4,
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    h_domain = Domains.SphereDomain(radius),
    h_mesh = Meshes.EquiangularCubedSphere(h_domain, h_elem),
) where {FT}

    h_topology = Topologies.Topology2D(context, h_mesh)
    h_grid = Grids.SpectralElementGrid2D(h_topology, quad)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(z_min),
        Geometry.ZPoint(z_max);
        boundary_names,
    )
    z_mesh = Meshes.IntervalMesh(z_domain, stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    vertical_grid = FiniteDifferenceGrid(z_topology)
    return ExtrudedFiniteDifferenceGrid(
        h_grid,
        vertical_grid,
        hypsography,
        global_geometry,
    )
end

"""
    CubedSphereGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        device = ClimaComms.device(),
        context = ClimaComms.context(device),
        radius::Real = FT(6.371229e6),
        Nq::Integer = 4,
        h_elem::Integer = 10,
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
        h_domain = Domains.SphereDomain(radius),
        h_mesh = Meshes.EquiangularCubedSphere(h_domain, h_elem)
    )

A convenience constructor, which builds a
`SpectralElementGrid2D`.

All arguments are optional.
"""
CubedSphereGrid(; kwargs...) = CubedSphereGrid(Float64; kwargs...)
function CubedSphereGrid(
    ::Type{FT};
    device = ClimaComms.device(),
    context = ClimaComms.context(device),
    radius::Real = FT(6.371229e6),
    Nq::Integer = 4,
    h_elem::Integer = 10,
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    h_domain = Domains.SphereDomain(radius),
    h_mesh = Meshes.EquiangularCubedSphere(h_domain, h_elem),
) where {FT}
    h_topology = Topologies.Topology2D(context, h_mesh)
    return Grids.SpectralElementGrid2D(h_topology, quad)
end

"""
    ColumnGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer = 10,
        z_min::Real = FT(0),
        z_max::Real = FT(1),
        boundary_names = (:bottom, :top),
        device = ClimaComms.device(),
        context = ClimaComms.context(device),
        stretch = Meshes.Uniform(),
    )

A convenience constructor, which builds a
`FiniteDifferenceGrid` given.

All arguments are optional.
"""
ColumnGrid(; kwargs...) = ColumnGrid(Float64; kwargs...)
function ColumnGrid(
    ::Type{FT};
    z_elem::Integer = 10,
    z_min::Real = FT(0),
    z_max::Real = FT(1),
    boundary_names = (:bottom, :top),
    device = ClimaComms.device(),
    context = ClimaComms.context(device),
    stretch = Meshes.Uniform(),
) where {FT}
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(z_min),
        Geometry.ZPoint(z_max);
        boundary_names,
    )
    z_mesh = Meshes.IntervalMesh(z_domain, stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    return FiniteDifferenceGrid(z_topology)
end

"""
    Box3DGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer = 10,
        x_min::Real = FT(0),
        x_max::Real = FT(1),
        y_min::Real = FT(0),
        y_max::Real = FT(1),
        z_min::Real = FT(0),
        z_max::Real = FT(1),
        boundary_names = (:bottom, :top),
        x1periodic = true,
        x2periodic = true,
        x1boundary = (:east, :west),
        x2boundary = (:south, :north),
        device = ClimaComms.device(),
        context = ClimaComms.context(device),
        stretch = Meshes.Uniform(),
        hypsography = Flat(),
        global_geometry = Geometry.CartesianGlobalGeometry(),
        Nq::Integer = 4,
        n1::Integer = 1, # number of horizontal elements
        n2::Integer = 1, # number of horizontal elements
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    )

A convenience constructor, which builds a
`ExtrudedFiniteDifferenceGrid` with a
`FiniteDifferenceGrid` vertical grid and a
`SpectralElementGrid2D` horizontal grid.

All arguments are optional.
"""
Box3DGrid(; kwargs...) = Box3DGrid(Float64; kwargs...)
function Box3DGrid(
    ::Type{FT};
    z_elem::Integer = 10,
    x_min::Real = FT(0),
    x_max::Real = FT(1),
    y_min::Real = FT(0),
    y_max::Real = FT(1),
    z_min::Real = FT(0),
    z_max::Real = FT(1),
    boundary_names = (:bottom, :top),
    x1periodic = true,
    x2periodic = true,
    x1boundary = (:east, :west),
    x2boundary = (:south, :north),
    device = ClimaComms.device(),
    context = ClimaComms.context(device),
    stretch = Meshes.Uniform(),
    hypsography = Flat(),
    global_geometry = Geometry.CartesianGlobalGeometry(),
    Nq::Integer = 4,
    n1::Integer = 1,
    n2::Integer = 1,
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
) where {FT}

    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint{FT}(x_min),
            Geometry.XPoint{FT}(x_max);
            periodic = x1periodic,
            boundary_names = x1boundary,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint{FT}(y_min),
            Geometry.YPoint{FT}(y_max);
            periodic = x2periodic,
            boundary_names = x2boundary,
        ),
    )
    h_mesh = Meshes.RectilinearMesh(domain, n1, n2)
    h_topology = Topologies.Topology2D(context, h_mesh)
    h_grid = Grids.SpectralElementGrid2D(h_topology, quad)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(z_min),
        Geometry.ZPoint(z_max);
        boundary_names,
    )
    z_mesh = Meshes.IntervalMesh(z_domain, stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    vertical_grid = FiniteDifferenceGrid(z_topology)
    return ExtrudedFiniteDifferenceGrid(
        h_grid,
        vertical_grid,
        hypsography,
        global_geometry,
    )
end

"""
    SliceXZGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer = 10,
        x_min::Real = FT(0),
        x_max::Real = FT(1),
        z_min::Real = FT(0),
        z_max::Real = FT(1),
        boundary_names = (:bottom, :top),
        x1periodic = true,
        x1boundary = (:east, :west),
        device = ClimaComms.device(),
        context = ClimaComms.context(device),
        stretch = Meshes.Uniform(),
        hypsography = Flat(),
        global_geometry = Geometry.CartesianGlobalGeometry(),
        Nq::Integer = 4,
        n1::Integer = 1, # number of horizontal elements
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    )

A convenience constructor, which builds a
`ExtrudedFiniteDifferenceGrid` with a
`FiniteDifferenceGrid` vertical grid and a
`SpectralElementGrid1D` horizontal grid.

All arguments are optional.
 - ``
"""
SliceXZGrid(; kwargs...) = SliceXZGrid(Float64; kwargs...)
function SliceXZGrid(
    ::Type{FT};
    z_elem::Integer = 10,
    x_min::Real = FT(0),
    x_max::Real = FT(1),
    z_min::Real = FT(0),
    z_max::Real = FT(1),
    boundary_names = (:bottom, :top),
    x1periodic = true,
    x1boundary = (:east, :west),
    device = ClimaComms.device(),
    context = ClimaComms.context(device),
    stretch = Meshes.Uniform(),
    hypsography = Flat(),
    global_geometry = Geometry.CartesianGlobalGeometry(),
    Nq::Integer = 4,
    n1::Integer = 1,
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
) where {FT}

    h_domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(x_min),
        Geometry.XPoint{FT}(x_max);
        periodic = x1periodic,
        boundary_names = x1boundary,
    )
    h_mesh = Meshes.IntervalMesh(h_domain; nelems = n1)
    h_topology = Topologies.IntervalTopology(context, h_mesh)
    h_grid = Grids.SpectralElementGrid1D(h_topology, quad)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(z_min),
        Geometry.ZPoint(z_max);
        boundary_names,
    )
    z_mesh = Meshes.IntervalMesh(z_domain, stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    vertical_grid = FiniteDifferenceGrid(z_topology)
    return ExtrudedFiniteDifferenceGrid(
        h_grid,
        vertical_grid,
        hypsography,
        global_geometry,
    )
end

"""
    RectangleXYGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        x_min::Real = FT(0),
        x_max::Real = FT(1),
        y_min::Real = FT(0),
        y_max::Real = FT(1),
        boundary_names = (:bottom, :top),
        x1periodic = true,
        x2periodic = true,
        x1boundary = (:east, :west),
        x2boundary = (:south, :north),
        device = ClimaComms.device(),
        context = ClimaComms.context(device),
        hypsography = Flat(),
        global_geometry = Geometry.CartesianGlobalGeometry(),
        Nq::Integer = 4,
        n1::Integer = 1, # number of horizontal elements
        n2::Integer = 1, # number of horizontal elements
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    )

A convenience constructor, which builds a
`SpectralElementGrid2D` with a horizontal
`RectilinearMesh` mesh.

All arguments are optional.
"""
RectangleXYGrid(; kwargs...) = RectangleXYGrid(Float64; kwargs...)
function RectangleXYGrid(
    ::Type{FT};
    x_min::Real = FT(0),
    x_max::Real = FT(1),
    y_min::Real = FT(0),
    y_max::Real = FT(1),
    boundary_names = (:bottom, :top),
    x1periodic = true,
    x2periodic = true,
    x1boundary = (:east, :west),
    x2boundary = (:south, :north),
    device = ClimaComms.device(),
    context = ClimaComms.context(device),
    hypsography = Flat(),
    global_geometry = Geometry.CartesianGlobalGeometry(),
    Nq::Integer = 4,
    n1::Integer = 1,
    n2::Integer = 1,
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
) where {FT}

    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint{FT}(x_min),
            Geometry.XPoint{FT}(x_max);
            periodic = x1periodic,
            boundary_names = x1boundary,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint{FT}(y_min),
            Geometry.YPoint{FT}(y_max);
            periodic = x2periodic,
            boundary_names = x2boundary,
        ),
    )
    h_mesh = Meshes.RectilinearMesh(domain, n1, n2)
    h_topology = Topologies.Topology2D(context, h_mesh)
    return Grids.SpectralElementGrid2D(h_topology, quad)
end
