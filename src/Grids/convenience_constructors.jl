#=
While we recommend users to use our composable
methods, this file contains some convenience
constructors, which are meant to improve
developer experience.
=#
check_device_context(context, device) =
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
"""
    ExtrudedCubedSphereGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer,
        z_min::Real,
        z_max::Real,
        radius::Real,
        h_elem::Integer,
        Nq::Integer,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
        hypsography::HypsographyAdaption = Flat(),
        global_geometry::Geometry.AbstractGlobalGeometry = Geometry.ShallowSphericalGlobalGeometry(radius),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
        h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(radius), h_elem),
    )

A convenience constructor, which builds a
`ExtrudedFiniteDifferenceGrid`.
"""
ExtrudedCubedSphereGrid(; kwargs...) =
    ExtrudedCubedSphereGrid(Float64; kwargs...)

function ExtrudedCubedSphereGrid(
    ::Type{FT};
    z_elem::Integer,
    z_min::Real,
    z_max::Real,
    radius::Real,
    h_elem::Integer,
    Nq::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    hypsography::HypsographyAdaption = Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.ShallowSphericalGlobalGeometry(
        radius,
    ),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    h_mesh = Meshes.EquiangularCubedSphere(
        Domains.SphereDomain{FT}(radius),
        h_elem,
    ),
) where {FT}
    check_device_context(context, device)

    z_boundary_names = (:bottom, :top)
    h_topology = Topologies.Topology2D(context, h_mesh)
    h_grid = Grids.SpectralElementGrid2D(h_topology, quad)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(z_min),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = z_boundary_names,
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
        radius::Real,
        Nq::Integer,
        h_elem::Integer,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
        h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(radius), h_elem),
    )

A convenience constructor, which builds a
`SpectralElementGrid2D`.
"""
CubedSphereGrid(; kwargs...) = CubedSphereGrid(Float64; kwargs...)
function CubedSphereGrid(
    ::Type{FT};
    radius::Real,
    Nq::Integer,
    h_elem::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    h_mesh = Meshes.EquiangularCubedSphere(
        Domains.SphereDomain{FT}(radius),
        h_elem,
    ),
) where {FT}
    check_device_context(context, device)
    h_topology = Topologies.Topology2D(context, h_mesh)
    return Grids.SpectralElementGrid2D(h_topology, quad)
end

"""
    ColumnGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer,
        z_min::Real,
        z_max::Real,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
    )

A convenience constructor, which builds a
`FiniteDifferenceGrid` given.
"""
ColumnGrid(; kwargs...) = ColumnGrid(Float64; kwargs...)
function ColumnGrid(
    ::Type{FT};
    z_elem::Integer,
    z_min::Real,
    z_max::Real,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
) where {FT}
    check_device_context(context, device)
    z_boundary_names = (:bottom, :top)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(z_min),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = z_boundary_names,
    )
    z_mesh = Meshes.IntervalMesh(z_domain, stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    return FiniteDifferenceGrid(z_topology)
end

"""
    Box3DGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer,
        x_min::Real,
        x_max::Real,
        y_min::Real,
        y_max::Real,
        z_min::Real,
        z_max::Real,
        x1periodic::Bool,
        x2periodic::Bool,
        Nq::Integer,
        n1::Integer,
        n2::Integer,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
        hypsography::HypsographyAdaption = Flat(),
        global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    )

A convenience constructor, which builds a
`ExtrudedFiniteDifferenceGrid` with a
`FiniteDifferenceGrid` vertical grid and a
`SpectralElementGrid2D` horizontal grid.
"""
Box3DGrid(; kwargs...) = Box3DGrid(Float64; kwargs...)
function Box3DGrid(
    ::Type{FT};
    z_elem::Integer,
    x_min::Real,
    x_max::Real,
    y_min::Real,
    y_max::Real,
    z_min::Real,
    z_max::Real,
    x1periodic::Bool,
    x2periodic::Bool,
    Nq::Integer,
    n1::Integer,
    n2::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    hypsography::HypsographyAdaption = Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
) where {FT}
    check_device_context(context, device)
    x1boundary = (:east, :west)
    x2boundary = (:south, :north)
    z_boundary_names = (:bottom, :top)
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
        Geometry.ZPoint{FT}(z_min),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = z_boundary_names,
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
        z_elem::Integer,
        x_min::Real,
        x_max::Real,
        z_min::Real,
        z_max::Real,
        x1periodic::Bool,
        Nq::Integer,
        n1::Integer,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
        hypsography::HypsographyAdaption = Flat(),
        global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    )

A convenience constructor, which builds a
`ExtrudedFiniteDifferenceGrid` with a
`FiniteDifferenceGrid` vertical grid and a
`SpectralElementGrid1D` horizontal grid.
 - ``
"""
SliceXZGrid(; kwargs...) = SliceXZGrid(Float64; kwargs...)
function SliceXZGrid(
    ::Type{FT};
    z_elem::Integer,
    x_min::Real,
    x_max::Real,
    z_min::Real,
    z_max::Real,
    x1periodic::Bool,
    Nq::Integer,
    n1::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    hypsography::HypsographyAdaption = Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
) where {FT}
    check_device_context(context, device)

    x1boundary = (:east, :west)
    z_boundary_names = (:bottom, :top)
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
        Geometry.ZPoint{FT}(z_min),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = z_boundary_names,
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
        x_min::Real,
        x_max::Real,
        y_min::Real,
        y_max::Real,
        x1periodic::Bool,
        x2periodic::Bool,
        Nq::Integer,
        n1::Integer, # number of horizontal elements
        n2::Integer, # number of horizontal elements
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        hypsography::HypsographyAdaption = Flat(),
        global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
    )

A convenience constructor, which builds a
`SpectralElementGrid2D` with a horizontal
`RectilinearMesh` mesh.
"""
RectangleXYGrid(; kwargs...) = RectangleXYGrid(Float64; kwargs...)
function RectangleXYGrid(
    ::Type{FT};
    x_min::Real,
    x_max::Real,
    y_min::Real,
    y_max::Real,
    x1periodic::Bool,
    x2periodic::Bool,
    Nq::Integer,
    n1::Integer, # number of horizontal elements
    n2::Integer, # number of horizontal elements
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    hypsography::HypsographyAdaption = Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{Nq}(),
) where {FT}
    check_device_context(context, device)

    x1boundary = (:east, :west)
    x2boundary = (:south, :north)
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
