push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
import ClimaCore.Domains.Geometry: Cartesian2DPoint
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# set up function space
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 50,
    npoly = 4;
    vert_stretching_function = (ξ) -> ξ,
    topography_file = nothing,
)

    # build vertical mesh information
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)


    # build horizontal mesh information
    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)


    if topography_file === nothing
        topography = [zeros(npoly + 1) for i in 1:helem]
    else
        # todo Akshay
        # read topography on each horizontal mesh point
        # topography = topography_interpolation(horzspace, topography_file)
    end
    # todo do we seperate hv_center_space & hv_face_space
    # construct hv center/face spaces, recompute metric terms
    Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vertmesh,
        vert_stretching_function,
        topography,
    )

    # hv_center_space =
    #     Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    # hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    # return (hv_center_space, hv_face_space)

end

# set up rhs!
hvspace_2D((-500, 500), (0, 1000))
