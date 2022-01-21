#module ClimaCoreExodus

import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies
using NCDatasets

using IntervalSets

include("map_mesh_helper.jl")

function write_exodus_identical(filename, topology::Topologies.Topology2D)
    """
    generates a mesh that is identical to TempestRemap
    """

    mesh = topology.mesh
    ne = topology.mesh.ne

    # get CC mesh Cartesian coordinates
    coord = [[Meshes.coordinates(mesh, elem, i) for i in collect(1:1:4)] for elem in Meshes.elements(mesh)] # [ne, ne, 6][4][3]

    # reorder coord to be consistent with tempest face numbering and rotations
    coord_f_n_n_v_x = reorder_cc2tr_coord(coord, ne) # [6, ne, ne, 4, 3] 

    # produce connectivity tensor which maps Exodus unique nodes onto the face-wise indices 
    connect1_v_fnn, connect1_f_n_n_v = calc_connect1(ne) # tempest map from unique
    connect1_4tr = Int32.(connect1_v_fnn)

    # reorder coord_on_tfaces (4 x 54) in the same order as tempest unique nodes (56 x 3)
    coord_4tr = f2u_coord(connect1_f_n_n_v, coord_f_n_n_v_x, ne) # = coord in tempest

    len_string = 33
    len_line = 81
    four = 4
    time_step = 0
    num_dim = 3
    num_nodes = Int32((ne * ne * 6 * 4 - 3 * 8) / 4 + 8 )
    num_elem = Int32(ne * ne * 6)
    num_qa_rec = 1
    num_el_blk = 1
    num_el_in_blk1 = Int32(ne * ne * 6)
    num_nod_per_el1 = 4
    num_att_in_blk1 = 1

    # init_data
    dts = NCDataset(filename, "c")

    # dimensions
    defDim(dts, "len_string", len_string)
    defDim(dts, "len_line", len_line)
    defDim(dts, "four", four)
    defDim(dts, "time_step", time_step)
    defDim(dts, "num_dim", num_dim)
    defDim(dts, "num_nodes", num_nodes)
    defDim(dts, "num_elem", num_elem)
    defDim(dts, "num_qa_rec", num_qa_rec)
    defDim(dts, "num_el_blk", num_el_blk)
    defDim(dts, "num_el_in_blk1", num_el_in_blk1)
    defDim(dts, "num_nod_per_el1", num_nod_per_el1)
    defDim(dts, "num_att_in_blk1", num_att_in_blk1)

    # global attibutes
    dts.attrib["title"] = "ClimaCore_CSEAmesh"
    dts.attrib["api_version"] = Float32(5.0)
    dts.attrib["version"] = Float32(5.0)
    dts.attrib["floating_point_word_size"] = 8
    dts.attrib["file_size"] = 0

    # variables
    var_time_whole = defVar(dts, "time_whole", Float64, ("time_step",))
    var_qa_records = defVar(dts, "qa_records", Char, ("len_string","four","num_qa_rec")) # quality assurance record (code name, QA descriptor, date, time) - here '\0's
    var_coor_names = defVar(dts, "coor_names", Char, ("len_string","num_dim"))
    var_eb_names = defVar(dts, "eb_names", Char, ("len_string","num_el_blk"))
    var_eb_status = defVar(dts, "eb_status", Int32, ("num_el_blk",))
    var_eb_prop1 = defVar(dts, "eb_prop1", Int32, ("num_el_blk",))
    var_attrib1 = defVar(dts, "attrib1", Float64, ("num_att_in_blk1","num_el_in_blk1"))
    var_connect1 = defVar(dts, "connect1", Int32, ("num_nod_per_el1","num_el_in_blk1")) 
    var_global_id1 = defVar(dts,"global_id1",Int32,("num_el_in_blk1",))
    var_edge_type1 = defVar(dts, "edge_type1", Int32, ("num_nod_per_el1","num_el_in_blk1")) # tempest specific
    var_coord = defVar(dts, "coord", Float64, ("num_nodes","num_dim"))

    var_coord[:,:] =  coord_4tr[:,:]
    var_connect1[:,:] = connect1_4tr[:,:]
    var_coor_names[1,:] = [only("x"),only("y"),only("z")]
    var_eb_prop1[:] = Int32(1)
    var_eb_status[:] = Int32(1)
    var_global_id1[:] = Int32.(collect(1:1:num_el_in_blk1))
    var_attrib1[:,:] .= 9.96921e36
    var_edge_type1[:,:] .= 0

    close(dts)
end

function write_exodus_general(filename, topology::Topologies.Topology2D)
    """
    generates a mesh that is identical to TempestRemap
    """
    mesh = topology.mesh
    ne = topology.mesh.ne

    # get CC mesh Cartesian coordinates
    coord_f_n_n_v_x = zeros(6, ne,ne, 4, 3)
    for e_x in collect(1:1:ne)  
        for e_y in collect(1:1:ne) 
            for (f_i, f) in enumerate([1,2,4,5,6,3])
                elem =  Meshes.elements(mesh)[e_x, e_y, f]
                for v in collect(1:1:4)
                    coord_f_n_n_v_x[f, e_y, e_x, v, :] = unwrap_cc_coord(Meshes.coordinates(mesh, elem, v)) # [6, ne, ne, 4, 3] 
                end
            end
        end
    end 

    # produce connectivity tensor which maps Exodus unique nodes onto the face-wise indices 
    connect1_v_fnn, connect1_f_n_n_v = calc_connect1(ne) # tempest map from unique
    connect1_4tr = Int32.(connect1_v_fnn)

    # reorder coord_on_tfaces (4 x 54) in the same order as tempest unique nodes (56 x 3)
    coord_4tr = f2u_coord(connect1_f_n_n_v, coord_f_n_n_v_x, ne) # = coord in tempest

    len_string = 33
    len_line = 81
    four = 4
    time_step = 0
    num_dim = 3
    num_nodes = Int32((ne * ne * 6 * 4 - 3 * 8) / 4 + 8 )
    num_elem = Int32(ne * ne * 6)
    num_qa_rec = 1
    num_el_blk = 1
    num_el_in_blk1 = Int32(ne * ne * 6)
    num_nod_per_el1 = 4
    num_att_in_blk1 = 1

    # init_data
    dts = NCDataset(filename, "c")

    # dimensions
    defDim(dts, "len_string", len_string)
    defDim(dts, "len_line", len_line)
    defDim(dts, "four", four)
    defDim(dts, "time_step", time_step)
    defDim(dts, "num_dim", num_dim)
    defDim(dts, "num_nodes", num_nodes)
    defDim(dts, "num_elem", num_elem)
    defDim(dts, "num_qa_rec", num_qa_rec)
    defDim(dts, "num_el_blk", num_el_blk)
    defDim(dts, "num_el_in_blk1", num_el_in_blk1)
    defDim(dts, "num_nod_per_el1", num_nod_per_el1)
    defDim(dts, "num_att_in_blk1", num_att_in_blk1)

    # global attibutes
    dts.attrib["title"] = "ClimaCore_CSEAmesh"
    dts.attrib["api_version"] = Float32(5.0)
    dts.attrib["version"] = Float32(5.0)
    dts.attrib["floating_point_word_size"] = 8
    dts.attrib["file_size"] = 0

    # variables
    var_time_whole = defVar(dts, "time_whole", Float64, ("time_step",))
    var_qa_records = defVar(dts, "qa_records", Char, ("len_string","four","num_qa_rec")) # quality assurance record (code name, QA descriptor, date, time) - here '\0's
    var_coor_names = defVar(dts, "coor_names", Char, ("len_string","num_dim"))
    var_eb_names = defVar(dts, "eb_names", Char, ("len_string","num_el_blk"))
    var_eb_status = defVar(dts, "eb_status", Int32, ("num_el_blk",))
    var_eb_prop1 = defVar(dts, "eb_prop1", Int32, ("num_el_blk",))
    var_attrib1 = defVar(dts, "attrib1", Float64, ("num_att_in_blk1","num_el_in_blk1"))
    var_connect1 = defVar(dts, "connect1", Int32, ("num_nod_per_el1","num_el_in_blk1")) 
    var_global_id1 = defVar(dts,"global_id1",Int32,("num_el_in_blk1",))
    var_edge_type1 = defVar(dts, "edge_type1", Int32, ("num_nod_per_el1","num_el_in_blk1")) # tempest specific
    var_coord = defVar(dts, "coord", Float64, ("num_nodes","num_dim"))

    var_coord[:,:] =  coord_4tr[:,:]
    var_connect1[:,:] = connect1_4tr[:,:]
    var_coor_names[1,:] = [only("x"),only("y"),only("z")]
    var_eb_prop1[:] = Int32(1)
    var_eb_status[:] = Int32(1)
    var_global_id1[:] = Int32.(collect(1:1:num_el_in_blk1))
    var_attrib1[:,:] .= 9.96921e36
    var_edge_type1[:,:] .= 0

    close(dts)

end



#     mesh = topology.mesh
#     ne = topology.mesh.ne

#     # get CC mesh Cartesian coordinates
#     coord = [[Meshes.coordinates(mesh, elem, i) for i in collect(1:1:4)] for elem in Meshes.elements(mesh)] # [3×3×6][4]

#     for elem in Meshes.elements(mesh)
#         for (f_i, f) in enumerate([1,2,4,5,6,3])
#             for e_y in collect(1:1:ne) 
#                 for e_x in collect(1:1:ne)   
#         coord = unwrap_cc_coord()
    

#     # reorder coord to be consistent with tempest faces
#     coord_on_tfaces = reorder_coord_cc2tempest(coord, ne)

    
    
#     # produce tempest mapping tensor which maps tempest unique nodes onto the face-wise indices 
#     connect1_v_fnn, connect1_f_n_n_v = calc_connect1(ne) # tempest map from unique
#     connect1_4tempest = Int32.(connect1_v_fnn)
    


#     # reorder coord_on_tfaces (4 x 54) in the same order as tempest unique nodes (56 x 3)
#     coord_4tempest = tempest_fnodes_to_unodes(connect1_f_n_n_v, coord_on_tfaces, ne) # = coord in tempest

#     len_string = 33
#     len_line = 81
#     four = 4
#     time_step = 0
#     num_dim = 3
#     num_nodes = Int32((ne * ne * 6 * 4 - 3 * 8) / 4 + 8 )
#     num_elem = Int32(ne * ne * 6)
#     num_qa_rec = 1
#     num_el_blk = 1
#     num_el_in_blk1 = Int32(ne * ne * 6)
#     num_nod_per_el1 = 4
#     num_att_in_blk1 = 1

#     # init_data
#     dts = NCDataset(filename, "c")

#     # dimensions
#     defDim(dts, "len_string", len_string)
#     defDim(dts, "len_line", len_line)
#     defDim(dts, "four", four)
#     defDim(dts, "time_step", time_step)
#     defDim(dts, "num_dim", num_dim)
#     defDim(dts, "num_nodes", num_nodes)
#     defDim(dts, "num_elem", num_elem)
#     defDim(dts, "num_qa_rec", num_qa_rec)
#     defDim(dts, "num_el_blk", num_el_blk)
#     defDim(dts, "num_el_in_blk1", num_el_in_blk1)
#     defDim(dts, "num_nod_per_el1", num_nod_per_el1)
#     defDim(dts, "num_att_in_blk1", num_att_in_blk1)

#     # global attibutes
#     dts.attrib["title"] = "cc_test"
#     dts.attrib["api_version"] = Float32(5.0)
#     dts.attrib["version"] = Float32(5.0)
#     dts.attrib["floating_point_word_size"] = 8
#     dts.attrib["file_size"] = 0

#     # variables
#     var_time_whole = defVar(dts, "time_whole", Float64, ("time_step",))
#     var_qa_records = defVar(dts, "qa_records", Char, ("len_string","four","num_qa_rec"))
#     var_coor_names = defVar(dts, "coor_names", Char, ("len_string","num_dim"))
#     var_eb_names = defVar(dts, "eb_names", Char, ("len_string","num_el_blk"))
#     var_eb_status = defVar(dts, "eb_status", Int32, ("num_el_blk",))
#     var_eb_prop1 = defVar(dts, "eb_prop1", Int32, ("num_el_blk",))
#     var_attrib1 = defVar(dts, "attrib1", Float64, ("num_att_in_blk1","num_el_in_blk1"))
#     var_connect1 = defVar(dts, "connect1", Int32, ("num_nod_per_el1","num_el_in_blk1")) # 4 x (ne*ne*6)
#     var_global_id1 = defVar(dts,"global_id1",Int32,("num_el_in_blk1",))
#     var_edge_type1 = defVar(dts, "edge_type1", Int32, ("num_nod_per_el1","num_el_in_blk1"))
#     var_coord = defVar(dts, "coord", Float64, ("num_nodes","num_dim")) # 56 x 3

#     var_coord[:,:] =  coord_4tempest[:,:]
#     var_connect1[:,:] = connect1_4tempest[:,:]
#     var_coor_names[1,:] = [only("x"),only("y"),only("z")]
#     var_eb_prop1[:] = Int32(1)
#     var_eb_status[:] = Int32(1)
#     var_global_id1[:] = Int32.(collect(1:1:num_el_in_blk1))
#     var_attrib1[:,:] .= 9.96921e36
#     var_edge_type1[:,:] .= 0

#     close(dts)
# end
# #end # module

# TODO (though most likely won't need this)
# - expand to more blocks for MPI / easier storage  
# 
# 