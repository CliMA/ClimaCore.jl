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
    Operators,
    Topographies
using ClimaCore.Geometry




function warp_agnesi_peak(coord; a = 1 / 2)
    8 * a^3 / (coord.x^2 + 4 * a^2)
end


# set up function space
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 50,
    npoly = 4;
    stretch = Meshes.Uniform(),
    warp_fn = warp_agnesi_peak,
)

    # build vertical mesh information with stretching in [0, H]
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)




    # build horizontal mesh information
    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)


    z_surface = warp_fn.(Fields.coordinate_field(horzspace))

    hv_face_space = Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_face_space,
        z_surface,
        Topographies.LinearAdaption(),
    )
end

# set up rhs!
space = hvspace_2D((-500, 500), (0, 1000))
