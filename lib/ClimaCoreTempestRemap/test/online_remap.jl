import ClimaCore
using ClimaComms
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields
using NCDatasets
using Test
using ClimaCoreTempestRemap
using LinearAlgebra

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

"""
    reshape_sparse_to_field!(field, in_array, R)

reshapes and broadcasts a sparse matrix data array (e.g., output from TempestRemap) into a Field object
"""
function reshape_sparse_to_field!(field::Fields.Field, in_array::Array, R)
    field_array = parent(field)

    fill!(field_array, zero(eltype(field_array)))
    Nf = size(field_array, 3)

    f = 1
    for (n, row) in enumerate(R.row_indices)
        it, jt, et = (
            view(R.target_local_idxs[1], n),
            view(R.target_local_idxs[2], n),
            view(R.target_local_idxs[3], n),
        )
        for f in 1:Nf
            field_array[it, jt, f, et] .= in_array[row]
        end
    end
    # broadcast to the redundant nodes using unweighted dss
    topology = Spaces.topology(axes(field))
    hspace = Spaces.horizontal_space(axes(field))
    quadrature_style = Spaces.quadrature_rule(hspace)
    Spaces.dss2!(Fields.field_values(field), topology, quadrature_style)
    return field
end


@testset "online remap 2D sphere data" begin

    # domain
    R = 1.0 # unit sphere
    domain = ClimaCore.Domains.SphereDomain(R)

    # source grid params
    ne_i = 20 # #elements
    nq_i = 3 # polynomial order for SE discretization

    # target grid params
    ne_o = 5
    nq_o = 3

    # construct source mesh
    mesh_i = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_i)
    topology_i = ClimaCore.Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(),
        mesh_i,
    )
    space_i = Spaces.SpectralElementSpace2D(
        topology_i,
        Spaces.Quadratures.GLL{nq_i}(),
    )
    coords_i = Fields.coordinate_field(space_i)

    # construct target mesh
    mesh_o = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_o)
    topology_o = ClimaCore.Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(),
        mesh_o,
    )
    space_o = Spaces.SpectralElementSpace2D(
        topology_o,
        Spaces.Quadratures.GLL{nq_o}(),
    )
    coords_o = Fields.coordinate_field(space_o)

    # generate test data in the Field format
    field_i = sind.(Fields.coordinate_field(space_i).long)

    # use TempestRemap to generate map weights
    weightfile = tempname()
    R = ClimaCoreTempestRemap.generate_map(
        space_o,
        space_i,
        weightfile = weightfile,
        in_type = "cgll",
        out_type = "cgll",
    )

    # apply the remap
    field_o = ClimaCoreTempestRemap.remap(R, field_i)

    # TEST_1: error between our `apply!` in ClimaCoe and `apply_remap` in TempestRemap

    # write test data for offline map apply for comparison
    datafile_in = joinpath(OUTPUT_DIR, "data_in.nc")

    NCDataset(datafile_in, "c") do nc
        def_space_coord(nc, space_i, ; type = "cgll") # only cgll supported by generate_map
        nc_time = def_time_coord(nc)
        nc_sinlong = defVar(nc, "sinlong", Float64, space_i, ("time",))
        nc_sinlong[:, 1] = field_i
        nothing
    end

    ## for test below, apply offline map, read in the resulting field and reshape it to the IJFH format
    datafile_out = joinpath(OUTPUT_DIR, "data_out.nc")
    apply_remap(datafile_out, datafile_in, weightfile, ["sinlong"])

    offline_outarray = NCDataset(datafile_out, "r") do ds_wt
        ds_wt["sinlong"][:][:, 1]
    end
    offline_outarray = Float64.(offline_outarray)

    offline_field = similar(field_o)
    reshape_sparse_to_field!(offline_field, offline_outarray, R)

    @test offline_field ≈ field_o rtol = 1e-6

    # TEST_2: error compared to the analytical solution

    # reference
    field_ref = sind.(Fields.coordinate_field(space_o).long)
    @test field_ref ≈ field_o atol = 0.05

end
