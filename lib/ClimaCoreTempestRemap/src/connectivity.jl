""" 
    get_cc_gll_connect(ne_i::Int, nq::Int)
loop through CC GLL points and generate GLL connectivity matrix

"""
function get_cc_gll_connect(ne_i::Int, nq::Int)

    ntot = ne_i*(nq-1)
    face1 = collect(-ntot/2:1:ntot/2) * collect(-ntot/2:1:ntot/2)'
    face2 = collect(ntot/2:-1:-ntot/2)*collect(-ntot/2:1:ntot/2)'
    
    face3 = collect(ntot/2:-1:-ntot/2) * collect(ntot/2:-1:-ntot/2)'
    face4 = collect(ntot/2:-1:-ntot/2)* collect(ntot/2:-1:-ntot/2)'
    
    face5 = collect(ntot/2:-1:-ntot/2) * collect(-ntot/2:1:ntot/2)' 
    face6 = collect(-ntot/2:1:ntot/2) * collect(-ntot/2:1:ntot/2)' 
    
    face1_c = [ones(ntot+1)*ntot/2,  collect(-ntot/2:1:ntot/2), collect(-ntot/2:1:ntot/2)]
    face2_c = [collect(ntot/2:-1:-ntot/2),  ones(ntot+1)*ntot/2 , collect(-ntot/2:1:ntot/2)]
    
    face3_c = [collect(ntot/2:-1:-ntot/2), collect(ntot/2:-1:-ntot/2), ones(ntot+1)*ntot/2]
    face4_c = [-ones(ntot+1)*ntot/2, collect(ntot/2:-1:-ntot/2), collect(ntot/2:-1:-ntot/2)]
    
    face5_c = [collect(-ntot/2:1:ntot/2), -ones(ntot+1)*ntot/2, collect(ntot/2:-1:-ntot/2) ]
    face6_c = [collect(-ntot/2:1:ntot/2), collect(-ntot/2:1:ntot/2),  -ones(ntot+1)*ntot/2 ]
    
    six_faces_dc = [face1_c,face2_c,face3_c,face4_c,face5_c,face6_c]
    six_faces = [face1,face2,face3,face4,face5,face6]

    connect = zeros(nq, nq, 6 * ne_i * ne_i )

    node_ct = 0
    coords = []
    for f in collect(1:1:6)
        for e_y in collect(1:1:ne_i)
            for e_x in collect(1:1:ne_i)
                for nq_y in collect(1:1:nq)
                    for nq_x in collect(1:1:nq)
                        if f in [1]
                            x = six_faces_dc[f][1][1]
                            y = six_faces_dc[f][2][(e_x-1)*(nq) + nq_x - (e_x-1)]
                            z = six_faces_dc[f][3][(e_y-1)*(nq) + nq_y - (e_y-1)]
                            if (x,y,z) ∉ coords
                                push!(coords, (x,y,z))
                                node_ct +=1
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = node_ct
                            else
                                ui = findfirst(isequal((x,y,z)), coords)
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = ui
                            end
                        end
                        if f in [4]
                            x = six_faces_dc[f][1][1]
                            y = six_faces_dc[f][2][(e_y-1)*(nq) + nq_y - (e_y-1)]
                            z = six_faces_dc[f][3][(e_x-1)*(nq) + nq_x - (e_x-1)]
                            if (x,y,z) ∉ coords
                                push!(coords, (x,y,z))
                                node_ct +=1
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = node_ct
                            else
                                ui = findfirst(isequal((x,y,z)), coords)
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = ui

                            end
                        end
                        if f in [2]
                            x = six_faces_dc[f][1][(e_x-1)*(nq) + nq_x - (e_x-1)]
                            y = six_faces_dc[f][2][1]
                            z = six_faces_dc[f][3][(e_y-1)*(nq) + nq_y - (e_y-1)]
                            if (x,y,z) ∉ coords
                                push!(coords, (x,y,z))
                                node_ct +=1
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = node_ct
                            else
                                ui = findfirst(isequal((x,y,z)), coords)
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = ui
                                println(ui)
                            end     
                        end
                        if f in [5]
                            x = six_faces_dc[f][1][(e_y-1)*(nq) + nq_y - (e_y-1)]
                            y = six_faces_dc[f][2][1]
                            z = six_faces_dc[f][3][(e_x-1)*(nq) + nq_x - (e_x-1)]
                            if (x,y,z) ∉ coords
                                push!(coords, (x,y,z))
                                node_ct +=1
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = node_ct
                            else
                                ui = findfirst(isequal((x,y,z)), coords)
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = ui

                            end     
                        end
                        if f in [3]
                            x = six_faces_dc[f][1][(e_x-1)*(nq) + nq_x - (e_x-1)]
                            y = six_faces_dc[f][2][(e_y-1)*(nq) + nq_y - (e_y-1)]
                            z = six_faces_dc[f][3][1]
                            if (x,y,z) ∉ coords
                                push!(coords, (x,y,z))
                                node_ct +=1
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = node_ct
                            else
                                ui = findfirst(isequal((x,y,z)), coords)
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = ui
                            end              
                        end
                        if f in [6]
                            x = six_faces_dc[f][1][(e_y-1)*(nq) + nq_y - (e_y-1)]
                            y = six_faces_dc[f][2][(e_x-1)*(nq) + nq_x - (e_x-1)]
                            z = six_faces_dc[f][3][1]
                            if (x,y,z) ∉ coords
                                push!(coords, (x,y,z))
                                node_ct +=1
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = node_ct
                            else
                                ui = findfirst(isequal((x,y,z)), coords)
                                connect[nq_x, nq_y, (f-1)*ne_i^2 + (e_y-1)*ne_i + e_x]  = ui
                            end              
                        end
                    end
                end
            end
        end
    end
    return (coords, node_ct, connect)
end


"""
    project_IJFH_to_unique_nodes

transform an IJFH data array to a vector that follows the order of the Tempest unique GLL nodes 
"""
function project_IJFH_to_unique(var::ClimaCore.DataLayouts.IJFH)
    nq = size(var)[1]
    ne = Int(sqrt(size(var)[end] / 6))
    _, node_ct, connect = get_cc_gll_connect(ne, nq)
    # map(x ->  getindex(view(parent(var)[:,:,1,:]),unwrap_cc_coord(findfirst(isequal(x), connect))...), collect(1:1:node_ct)) # doesn't work if not subarray...

    unique = zeros(node_ct)
    for x in collect(1:1:node_ct)
        unique[x] = getindex(parent(var)[:,:,1,:],unwrap_cc_coord(findfirst(isequal(x), connect))...)
    end
    return unique
 end
unwrap_cc_coord(coord) = [coord[1] , coord[2], coord[3]]


"""
    project_unique_nodes_to_IJFH(var::Vector, nq::Int, ne::Int)

transform a vector that follows the order of the Tempest unique GLL nodes to an IJFH data array
"""
function project_unique_to_IJFH(var::Vector, nq::Int, ne::Int)
    num_elem = ne^2 * 6
    _, _, connect = get_cc_gll_connect(ne, nq)

    out = Array{Float64}(undef, nq, nq, 1, 1, num_elem) 
    for e in collect(1:1:num_elem)
        for nq_y in collect(1:1:nq) 
            for nq_x in collect(1:1:nq) 
                c = connect[nq_y, nq_x, e]
                out[nq_x,nq_y,1,1,e] = var[Int(c)]
            end
        end
    end
    return out
end

"""
    LinearTempestRemap{T, S, M, C}

stores info on the TempestRemap map and the source and target data
"""
struct LinearTempestRemap{T, S, M, C} # make consistent with / move to regridding.jl
    target::T
    source::S
    map::M # linear mapping operator
    col::C # source indices mapping source GLL nodes to each overlap grid point
    row::C # target indices mapping source GLL nodes to each overlap grid point
end

"""
    remap!(target, R, source)

applies the remapping
"""
# function remap!(target, R, source)
#     map, col, row = (R.map, R.col, R.row)
#     @assert size(map) == size(col) == size(row)

#     for (i, wt) in enumerate(map) # could resize to matrix and do matrix multiply?
#         target[row[i]] += wt * source[col[i]]
#     end

# end
function remap!(R)

    target, source, map, col, row = (R.target, R.source, R.map, R.col, R.row)
    
    target, source, map, col, row = (RemapInfo.target, RemapInfo.source, RemapInfo.map, RemapInfo.col, RemapInfo.row)

    @assert size(map) == size(col) == size(row)

    source_unique = project_IJFH_to_unique(getfield(source,:values)) # maybe can operate on fields?

    nq_t = size(parent(target))[1]
    ne_t = Int(sqrt(size(parent(target))[end] / 6))

    n_nodes_t = (ne_t^2 * 6 * (nq_t-1)*(nq_t-1))  - (8*3 ) / 4 + 8
    target_unique = zeros(Int(n_nodes_t))

    for (i, wt) in enumerate(map) # could resize to matrix and do matrix multiply?
        target_unique[row[i]] += wt * source_unique[col[i]]
    end

    target .= project_unique_to_IJFH(target_unique, nq_t, ne_t)
end


"""
    generate_map

offline generation of remapping weights using TempestRemap
"""
function generate_map(remap_weights_filename::String, topology_source, topology_target, nq_source, nq_target)

    OUTPUT_DIR = dirname(remap_weights_filename)
    # write meshes
    meshfile_cc_source = joinpath(OUTPUT_DIR, "mesh_cc_source.g")
    write_exodus(meshfile_cc_source, topology_source)

    meshfile_cc_target = joinpath(OUTPUT_DIR, "mesh_cc_target.g")
    write_exodus(meshfile_cc_target, topology_target)

    meshfile_cc_overlap = joinpath(OUTPUT_DIR, "mesh_cc_overlap.g")
    overlap_mesh(meshfile_cc_overlap, meshfile_cc_source, meshfile_cc_target)
    
    # calculate map weights 
    weightfile = joinpath(OUTPUT_DIR, remap_weights_filename)
    remap_weights(
        weightfile,
        meshfile_cc_source,
        meshfile_cc_target,
        meshfile_cc_overlap;
        in_type = "cgll",
        in_np = nq_source,
        out_type = "cgll",
        out_np = nq_target,
    )

    ds_wt = NCDataset(remap_weights_filename,"r")
    map = ds_wt["S"][:]
    row = ds_wt["row"][:]
    col = ds_wt["col"][:]
    close(ds_wt)

    # add optional clean up of redundant files

    return (map, col, row)
end