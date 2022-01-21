# tempest unique nodes on cube

mutable struct itr{T}
    n::T
end
function add!(ci, n)
    ci.n = ci.n + n
end

function calc_connect1(ne)
    """
    generate connectivity tensor
    """
    # gen tempest_unique_node_order
    n1 = ne+1
    cube_dims = zeros(n1,n1,n1)

    # corners
    n_1 = n1 - 1
    n_2 = n1 - 2

    cube_dims[n1,1,1] = 1 # + - -
    cube_dims[n1,n1,1] = 2 # + + -
    cube_dims[1,n1,1] = 3# - + -
    cube_dims[1,1,1] = 4# - - -
    cube_dims[n1,1,n1] = 5 # + - +
    cube_dims[n1,n1,n1] = 6 # + + +
    cube_dims[1,n1,n1] = 7# - + +
    cube_dims[1,1,n1] = 8# - - +

    # edges
    ci = itr(9)
    cube_dims[n1,2:n_1,1] = collect(ci.n:1:add!(ci, n_2)-1)
    cube_dims[2:n_1,n1,1] = collect(ci.n:1:add!(ci, n_2)-1)[end:-1:1]
    cube_dims[1,2:n_1,1] = collect(ci.n:1:add!(ci, n_2)-1)[end:-1:1]
    cube_dims[2:n_1,1,1] = collect(ci.n:1:add!(ci, n_2)-1)

    cube_dims[n1,1,2:n_1] = collect(ci.n:1:add!(ci, n_2)-1)
    cube_dims[n1,n1,2:n_1] = collect(ci.n:1:add!(ci, n_2)-1)
    cube_dims[1,n1,2:n_1] = collect(ci.n:1:add!(ci, n_2)-1)
    cube_dims[1,1,2:n_1] = collect(ci.n:1:add!(ci, n_2)-1)

    cube_dims[n1,2:n_1,n1] = collect(ci.n:1:add!(ci, n_2)-1)
    cube_dims[2:n_1,n1,n1] = collect(ci.n:1:add!(ci, n_2)-1)[end:-1:1]
    cube_dims[1,2:n_1,n1] = collect(ci.n:1:add!(ci, n_2)-1)[end:-1:1]
    cube_dims[2:n_1,1,n1] = collect(ci.n:1:add!(ci, n_2)-1)

    # internal vertices
    reshape_to_face(collect, n_2) = reshape(collect,n_2,n_2)

    cube_dims[n1,2:n_1,2:n_1] = reshape_to_face( collect(ci.n:1:add!(ci, n_2*n_2 )-1), n_2 ) #Eq1
    cube_dims[2:n_1,n1,2:n_1] = reshape_to_face( collect(ci.n:1:add!(ci, n_2*n_2 )-1), n_2 )[end:-1:1,:] #Eq2
    cube_dims[1,2:n_1,2:n_1] = reshape_to_face( collect(ci.n:1:add!(ci, n_2*n_2 )-1), n_2 )[end:-1:1,:] #Eq3
    cube_dims[2:n_1,1,2:n_1] = reshape_to_face( collect(ci.n:1:add!(ci, n_2*n_2 )-1), n_2 ) #Eq4
    cube_dims[2:n_1,2:n_1,1] = reshape_to_face( collect(ci.n:1:add!(ci, n_2*n_2 )-1), n_2 )' #Po1
    cube_dims[2:n_1,2:n_1,n1] = reshape_to_face( collect(ci.n:1:add!(ci, n_2*n_2 )-1), n_2 )[end:-1:1,:]'[end:-1:1,end:-1:1]  #Po2

    # collect only faces + transpose for below
    f_eq1 =  cube_dims[n1,:,:]
    f_eq2 =  cube_dims[:,n1,:]
    f_eq3 =  cube_dims[1,:,:]
    f_eq4 =  cube_dims[:,1,:]
    f_po1 =  cube_dims[:,:,1]
    f_po2 =  cube_dims[:,:,n1]

    faces = (f_eq1, f_eq2[end:-1:1,:], f_eq3[end:-1:1,:], f_eq4, f_po1',f_po2[end:-1:1,end:-1:1]'[end:-1:1,:])

    # get connect1 as 6 x 3 x 3 x 4
    tempest_nodes_idx_on_tfaces = zeros(6, ne,ne,4)
    for f in collect(1:1:6)  
        for e_y in collect(1:1:ne) 
            for e_x in collect(1:1:ne)   
                tempest_nodes_idx_on_tfaces[f,e_y,e_x,1] = faces[f][e_x,e_y]
                tempest_nodes_idx_on_tfaces[f,e_y,e_x,2] = faces[f][e_x+1,e_y]
                tempest_nodes_idx_on_tfaces[f,e_y,e_x,3] = faces[f][e_x+1,e_y+1]
                tempest_nodes_idx_on_tfaces[f,e_y,e_x,4] = faces[f][e_x,e_y+1]
            end
        end
    end
    connect1_f_n_n_v = tempest_nodes_idx_on_tfaces[:,:,:,:] # 6×3×3×4

    # get connect1 as 4 x 54
    tempest_nodes_idx_on_tfaces = zeros(6 * ne * ne,4)
    for f in collect(1:1:6)  
        for e_y in collect(1:1:ne) 
            for e_x in collect(1:1:ne)   
                tempest_nodes_idx_on_tfaces[(f-1)*ne^2+((e_y-1)*ne+e_x),1] = faces[f][e_x,e_y]
                tempest_nodes_idx_on_tfaces[(f-1)*ne^2+((e_y-1)*ne+e_x),2] = faces[f][e_x+1,e_y]
                tempest_nodes_idx_on_tfaces[(f-1)*ne^2+((e_y-1)*ne+e_x),3]= faces[f][e_x+1,e_y+1]
                tempest_nodes_idx_on_tfaces[(f-1)*ne^2+((e_y-1)*ne+e_x),4] = faces[f][e_x,e_y+1]
            end
        end
    end

    connect1_v_fnn = tempest_nodes_idx_on_tfaces' # 4 x 54 - same as connect1 in tempest
    # check = sum((connect1_v_fnn .- connect).^2) # test
    return connect1_v_fnn, connect1_f_n_n_v
end

unwrap_cc_coord(coord) = [coord.x1 , coord.x2, coord.x3]
function reorder_cc2tr_coord(coord, ne)
    """
    reorder CC Cartesian coordinates as in tempest (different face numbers and face orientaitons)
    """
    coord_on_tfaces = zeros(6, ne,ne,4,3);
    for (f_i, f) in enumerate([1,2,4,5,6,3])
        for e_y in collect(1:1:ne) 
            for e_x in collect(1:1:ne)   
                if f_i in [3,4]
                    e_x_new = e_y + 0
                    e_y_new = ne - e_x + 1 
                    coord_on_tfaces[f_i,e_y_new,e_x_new,1,:] .= unwrap_cc_coord(coord[e_x,e_y,f][2])
                    coord_on_tfaces[f_i,e_y_new,e_x_new,2,:] .= unwrap_cc_coord(coord[e_x, e_y,f][3])
                    coord_on_tfaces[f_i,e_y_new,e_x_new,3,:] .= unwrap_cc_coord(coord[e_x, e_y,f][4])
                    coord_on_tfaces[f_i,e_y_new,e_x_new,4,:] .= unwrap_cc_coord(coord[e_x, e_y,f][1])
                elseif f_i == 6
                    e_x_new = ne - e_y + 1
                    e_y_new = e_x + 0
                    coord_on_tfaces[f_i,e_y_new,e_x_new,1,:] .= unwrap_cc_coord(coord[e_x, e_y,f][4])
                    coord_on_tfaces[f_i,e_y_new,e_x_new,2,:] .= unwrap_cc_coord(coord[e_x, e_y,f][1])
                    coord_on_tfaces[f_i,e_y_new,e_x_new,3,:] .= unwrap_cc_coord(coord[e_x, e_y,f][2])
                    coord_on_tfaces[f_i,e_y_new,e_x_new,4,:] .= unwrap_cc_coord(coord[e_x, e_y,f][3])
                else
                    coord_on_tfaces[f_i,e_y,e_x,1,:] .= unwrap_cc_coord(coord[e_x,e_y,f][1])
                    coord_on_tfaces[f_i,e_y,e_x,2,:] .= unwrap_cc_coord(coord[e_x,e_y,f][2])
                    coord_on_tfaces[f_i,e_y,e_x,3,:] .= unwrap_cc_coord(coord[e_x,e_y,f][3])
                    coord_on_tfaces[f_i,e_y,e_x,4,:] .= unwrap_cc_coord(coord[e_x,e_y,f][4])
                end
            end
        end
    end
    return coord_on_tfaces
end

function f2u_coord(connect1_, coord_on_tfaces, ne)
    """
    map coordinates from face nodes to unique nodes
    """
    n_unique = Int((ne*ne*6*4 - 3*8)/4 + 8 )
    new_coord = zeros(n_unique, 3)
    for r in collect(1:1:n_unique)
        for f in collect(1:1:6)  
            for e_y in collect(1:1:ne) 
                for e_x in collect(1:1:ne)             
                    for v in collect(1:1:4)
                        if r == connect1_[f,e_y,e_x,v]
                            new_coord[r,:] = coord_on_tfaces[f,e_y,e_x,v,:]
                        end
                    end        
                end        
            end
        end
    end
    return new_coord
end
                    
