using ClimaComms, DataStructures
using GilbertCurves

"""
    Topology2D(mesh::AbstractMesh2D, elemorder=Mesh.elements(mesh))

This is a distributed topology for 2D meshes. `elemorder` is a vector or other
linear ordering of the `Mesh.elements(mesh)`. `elempid` is a sorted vector of
the same length as `elemorder`, each element of which contains the `pid` of the
owning process.

Internally, we can refer to elements in several different ways:
- `elem`: an element of the `mesh`. Often a `CartesianIndex` object.
- `gidx`: "global index": an enumeration of all elements:
    - `elemorder[gidx] == elem`
    - `orderindex[elem] == gidx`
- `lidx`: "local index": an enumeration of local elements.
    - `local_elem_gidx[lidx] == gidx`
- `sidx`: "send index": an index into the send buffer of a local element. A
  single local element may have multiple `sidx`s if it needs to be send to
  multiple processes.
    - `send_elem_lidx[sidx] == lidx`
- `ridx`: "receive index": an index into the receive buffer of a ghost element.
    - `recv_elem_gidx[ridx] == gidx`
"""
struct Topology2D{
    C <: ClimaComms.AbstractCommsContext,
    M <: Meshes.AbstractMesh{2},
    EO,
    OI,
    IF,
    GF,
    LV,
    LVO,
    GV,
    GVO,
    VI,
    BF,
    RGV,
} <: AbstractDistributedTopology
    "the ClimaComms context on which the topology is defined"
    context::C

    # common to all processes
    "mesh on which the topology is constructed"
    mesh::M
    "`elemorder[gidx]` should give the `e`th element"
    elemorder::EO
    "the inverse of `elemorder`: `orderindex[elemorder[e]] == e`"
    orderindex::OI
    "`elempid[gidx]` gives the process id which owns the `e`th element"
    elempid::Vector{Int}

    # specific to this process
    "the global indices that are local to this process"
    local_elem_gidx::UnitRange{Int}
    "process ids of neighboring processes"
    neighbor_pids::Vector{Int}
    "local indices of elements to be copied to send buffer"
    send_elem_lidx::Vector{Int}
    "number of elems to send to each neighbor process"
    send_elem_lengths::Vector{Int}
    "global indices of elements being received"
    recv_elem_gidx::Vector{Int}
    "number of elems to send to each neighbor process"
    recv_elem_lengths::Vector{Int}

    "a vector of all unique interior faces, (e, face, o, oface, reversed)"
    interior_faces::IF
    "a vector of all ghost faces, (e, face, o, oface, reversed)"
    ghost_faces::GF
    "the collection of `(lidx, vert)`` pairs of vertices of local elements which touch a ghost element"
    local_vertices::LV
    "the index in `local_vertices` of the first tuple for each unique vertex"
    local_vertex_offset::LVO
    """
    The collection of `(isghost, idx, vert)` tuples of vertices of local elements which touch a ghost element.
    If `isghost` is false, then `idx` is the `lidx`, otherwise it is the `ridx`
    """
    ghost_vertices::GV
    "the index in `ghost_vertices` of the first tuple for each unique vertex"
    ghost_vertex_offset::GVO

    "a vector of the lidx of neighboring local elements of each element"
    local_neighbor_elem::VI
    "the index into `local_neighbor_elem` for the start of each element"
    local_neighbor_elem_offset::VI
    "a vector of the ridx of neighboring ghost elements of each element"
    ghost_neighbor_elem::Vector{Int}
    "the index into `ghost_neighbor_elem` for the start of each element"
    ghost_neighbor_elem_offset::Vector{Int}

    "a NamedTuple of vectors of `(e,face)` "
    boundaries::BF
    "internal local elements (lidx)"
    internal_elems::Vector{Int}
    "local elements (lidx) located on process boundary"
    perimeter_elems::Vector{Int}
    "total number of unique non-boundary vertices"
    nglobalvertices::Int
    "total number of unique non-boundary faces"
    nglobalfaces::Int
    "global connection idx for each unique ghost vertex"
    ghost_vertex_gcidx::Vector{Int}
    "global connection idx for each unique ghost face"
    ghost_face_gcidx::Vector{Int}
    "number of unique vertices to be communicated to neighboring processes
    (`comm_vertex_lengths[i]` = number of unique vertices to be communicated to `neighbor_pids[i]`)"
    comm_vertex_lengths::Vector{Int}
    "number of unique faces to be communicated to each neighboring process
    (`comm_face_lengths[i]` = number of unique faces to be communicated to `neighbor_pids[i]`)"
    comm_face_lengths::Vector{Int}
    "neighbor process locations in neighbor_pids for each of the ghost vertices"
    ghost_vertex_neighbor_loc::Vector{Int}
    "offset array for `ghost_vertex_neighbor_loc`
    a ragged array representation: for i = 1:length(ghost_vertex_gcidx), we have that
    for j = ghost_vertex_comm_idx_offset[i]:ghost_vertex_comm_idx_offset[i+1]-1
    the ghost vertex ghost_vertex_gcidx[i] should get sent to
    neighbor_pids[ghost_vertex_neighbor_loc[j]]"
    ghost_vertex_comm_idx_offset::Vector{Int}
    "representative local ghost vertex (idx, vert) for each unique ghost vertex"
    repr_ghost_vertex::RGV #Vector{Tuple{Int, Int}}
    "neighbor process location in neighbor_pids for each ghost face"
    ghost_face_neighbor_loc::Vector{Int}
end

ClimaComms.device(topology::Topology2D) = ClimaComms.device(topology.context)
ClimaComms.array_type(topology::Topology2D) =
    ClimaComms.array_type(topology.context.device)

function Base.show(io::IO, topology::Topology2D)
    indent = get(io, :indent, 0)
    println(io, nameof(typeof(topology)))
    print(io, " "^(indent + 2), "context: ")
    print_context(io, topology.context)
    println(io)
    println(io, " "^(indent + 2), "mesh: ", topology.mesh)
    print(io, " "^(indent + 2), "elemorder: ")
    if topology.elemorder isa CartesianIndices
        print(io, topology.elemorder)
    else
        summary(io, topology.elemorder)
    end
end

"""
    spacefillingcurve(mesh::Meshes.AbstractCubedSphere)

Generate element ordering, `elemorder`, based on a space filling curve
for a `CubedSphere` mesh.

"""
function spacefillingcurve(mesh::Meshes.AbstractCubedSphere)
    ne = mesh.ne
    majordim = [1, 2, 1, 2, 1, 2]
    elemorder = CartesianIndex{3}[]
    for panel in 1:6
        push!(
            elemorder,
            [
                CartesianIndex(cartidx, panel) for
                cartidx in gilbertindices((ne, ne); majdim = majordim[panel])
            ]...,
        )
    end
    return elemorder
end

"""
    spacefillingcurve(mesh::Meshes.RectilinearMesh)

Generate element ordering, `elemorder`, based on a space filling curve
for a `Rectilinear` mesh.

"""
spacefillingcurve(mesh::Meshes.RectilinearMesh) = gilbertindices((
    Meshes.nelements(mesh.intervalmesh1),
    Meshes.nelements(mesh.intervalmesh2),
))

# returns partition[1..nelems] where partition[e] = pid of the owning process
function simple_partition(nelems::Int, npart::Int)
    partition = zeros(Int, nelems)
    ranges = [0:0 for _ in 1:nelems]
    quot, rem = divrem(nelems, npart)

    offset = 0
    for i in 1:npart
        part_len = i <= rem ? quot + 1 : quot
        part_range = (offset + 1):(offset + part_len)
        partition[part_range] .= i
        ranges[i] = part_range
        offset += part_len
    end
    return partition, ranges
end

function Topology2D(
    context::ClimaComms.AbstractCommsContext,
    mesh::Meshes.AbstractMesh{2},
    elemorder = Meshes.elements(mesh),
    elempid = nothing, # array of same size as elemorder, containing owning pid for each element (it should be sorted)
    orderindex = Meshes.linearindices(elemorder), # inverse mapping of elemorder (e.g. map CartesianIndex => Int)
)
    get!(
        Cache.OBJECT_CACHE,
        (Topology2D, context, mesh, elemorder, elempid, orderindex),
    ) do
        _Topology2D(context, mesh, elemorder, elempid, orderindex)
    end
end

function _Topology2D(
    context::ClimaComms.AbstractCommsContext,
    mesh::Meshes.AbstractMesh{2},
    elemorder,
    elempid,
    orderindex,
)
    pid = ClimaComms.mypid(context)
    nprocs = ClimaComms.nprocs(context)
    DA = ClimaComms.array_type(context.device)

    # To make IO easier, we enforce the partitioning to be sequential in the element ordering.
    @assert elempid === nothing
    elempid, ranges = simple_partition(length(elemorder), nprocs)
    @assert issorted(elempid)

    local_elem_gidx = ranges[pid] # gidx = local_elem_gidx[lidx]
    global_elem_lidx = Dict{Int, Int}() # inverse of local_elem_gidx: lidx = global_elem_lidx[gidx]
    for (lidx, gidx) in enumerate(local_elem_gidx)
        global_elem_lidx[gidx] = lidx
    end
    # build global vertex and face idx
    global_vertex_gidx = zeros(Int, 4, Meshes.nelements(mesh))
    global_face_gidx = zeros(Int, 4, Meshes.nelements(mesh))
    gvid, gfid = 1, 1
    for elem in Meshes.elements(mesh)
        ielem = orderindex[elem]
        for (face, vert) in zip(1:4, 1:4)
            if global_vertex_gidx[vert, ielem] == 0
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    ivelem = orderindex[velem]
                    global_vertex_gidx[vvert, ivelem] = gvid
                end
                gvid += 1
            end
            if !Meshes.is_boundary_face(mesh, elem, face)
                if global_face_gidx[face, ielem] == 0
                    opelem, opface, reversed =
                        Meshes.opposing_face(mesh, elem, face)
                    iopelem = orderindex[opelem]
                    global_face_gidx[face, ielem] = gfid
                    global_face_gidx[opface, iopelem] = gfid
                    gfid += 1
                end
            end
        end
    end
    nglobalvertices = gvid - 1
    nglobalfaces = gfid - 1
    # convention
    #  - elem: an element of the Mesh (often a CartesianIndex object)
    #  - gidx (global element index): this is a unique numbering of all elements
    #    - elemorder[gidx] = elem
    #    - orderindex[elem] = gidx
    #  - lidx (local element index): this is a unique numbering of the local elements
    #    - local_elem_gidx[lidx] = gidx
    #    - global_elem_lidx[gidx] = lidx  (local elements only)
    #  - sidx (index in sendbuffer)
    #    - send_elem_lidx[sidx] = lidx (lidx may appear multiple times)
    #  - ridx (index in recvbuffer)
    #    - global_elem_ridx[gidx] = ridx  (ghost elements only)
    send_elem_set = SortedSet{Tuple{Int, Int}}() # (pid to send to, lidx to send)
    recv_elem_set = SortedSet{Tuple{Int, Int}}() # (pid to receive from, gidx of element to receive)
    local_vertices = Tuple{Int, Int}[]
    local_vertex_offset = Int[1]
    ghost_vertices = Tuple{Bool, Int, Int}[]
    ghost_vertex_offset = Int[1]
    temp_vertlist = Tuple{Int, Int, Int}[]

    temp_local_neighbor_elemset = SortedSet{Int}()
    temp_ghost_neighbor_elemset = SortedSet{Int}()
    local_neighbor_elem = Int[]
    local_neighbor_elem_offset = Int[1]
    ghost_neighbor_elem = Int[]
    ghost_neighbor_elem_offset = Int[1]

    # 1) iterate over the vertices of local elements to determeind the vertex connectivity
    #    since we don't yet know the ridx of halo elements, we instead store the gidx.
    for (lidx, gidx) in enumerate(local_elem_gidx)
        elem = elemorder[gidx]
        empty!(temp_local_neighbor_elemset)
        empty!(temp_ghost_neighbor_elemset)
        # Determine whether a vertex is an interior or a ghost vertex. For
        # ghost vertices, we record the elements to be sent, the elements
        # to be received (ghost elements) and other information to ease DSS.
        for vert in 1:4
            seen = false
            isghostvertex = false
            empty!(temp_vertlist)
            for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                vgidx = orderindex[velem]
                vpid = elempid[vgidx]
                if vpid == pid
                    vlidx = global_elem_lidx[vgidx]
                    if vlidx != lidx
                        push!(temp_local_neighbor_elemset, vlidx)
                    end
                    if (vgidx, vvert) < (gidx, vert)
                        # we've already seen this vertex
                        seen = true
                    end
                    push!(temp_vertlist, (0, vlidx, vvert))
                else
                    isghostvertex = true
                    # use gidx until we can determine ridx ridx
                    push!(recv_elem_set, (vpid, vgidx))
                    push!(temp_ghost_neighbor_elemset, vgidx)
                    push!(temp_vertlist, (vpid, vgidx, vvert))
                end
            end
            if !seen
                # append to either local_vertices or ghost_vertices
                if isghostvertex
                    for (vpid, vidx, vvert) in temp_vertlist
                        # vidx will be lidx if local, gidx if ghost
                        # replace the gidx with ridx in step 4.
                        push!(ghost_vertices, (vpid != 0, vidx, vvert))
                        # if it's a ghost_vertex, push every (dest pid, lidx) to the send_elem_set
                        if vpid != 0
                            for (xpid, xidx, _) in temp_vertlist
                                if xpid == 0
                                    push!(send_elem_set, (vpid, xidx))
                                end
                            end
                        end
                    end
                    push!(ghost_vertex_offset, length(ghost_vertices) + 1)
                else
                    for (_, vlidx, vvert) in temp_vertlist
                        push!(local_vertices, (vlidx, vvert))
                    end
                    push!(local_vertex_offset, length(local_vertices) + 1)
                end
            end
        end

        # build neighbors
        append!(local_neighbor_elem, temp_local_neighbor_elemset)
        push!(local_neighbor_elem_offset, length(local_neighbor_elem) + 1)
        append!(ghost_neighbor_elem, temp_ghost_neighbor_elemset)
        push!(ghost_neighbor_elem_offset, length(ghost_neighbor_elem) + 1)
    end

    # 2) compute send_elem information
    send_elem_lidx = Int[] # lidx to copy to send buffer
    send_elem_pids = Int[] # list of pids (should be sorted)
    send_elem_lengths = Int[] # list of number of elements
    curpid = 0
    curlength = 0
    for (pid, lidx) in send_elem_set
        push!(send_elem_lidx, lidx)
        if pid != curpid
            if curpid != 0
                push!(send_elem_pids, curpid)
                push!(send_elem_lengths, curlength)
            end
            curpid = pid
            curlength = 0
        end
        curlength += 1
    end
    if curpid != 0
        push!(send_elem_pids, curpid)
        push!(send_elem_lengths, curlength)
    end
    @assert issorted(send_elem_pids)
    @assert sum(send_elem_lengths) == length(send_elem_lidx)


    # 3) compute recv_elem information
    recv_elem_pids = Int[] # list of pids (should be sorted)
    recv_elem_lengths = Int[] # list of number of elements
    recv_elem_gidx = Int[] # gidx of elements to be received
    global_elem_ridx = Dict{Int, Int}() # map of gidx to ridx (inverse of recv_elem_gidx)
    curpid = 0
    curlength = 0
    for (ridx, (pid, gidx)) in enumerate(recv_elem_set)
        push!(recv_elem_gidx, gidx)
        global_elem_ridx[gidx] = ridx
        if pid != curpid
            if curpid != 0
                push!(recv_elem_pids, curpid)
                push!(recv_elem_lengths, curlength)
            end
            curpid = pid
            curlength = 0
        end
        curlength += 1
    end
    if curpid != 0
        push!(recv_elem_pids, curpid)
        push!(recv_elem_lengths, curlength)
    end
    @assert recv_elem_pids == send_elem_pids
    @assert sum(recv_elem_lengths) == length(global_elem_ridx)

    # 4) update ghost_vertices with ridx
    for (i, (isghost, idx, vert)) in enumerate(ghost_vertices)
        if isghost
            ridx = global_elem_ridx[idx]
            ghost_vertices[i] = (isghost, ridx, vert)
        end
    end
    # update ghost_neighbor_elem with ridx, and order by ridx
    for i in eachindex(ghost_neighbor_elem)
        gidx = ghost_neighbor_elem[i]
        ridx = global_elem_ridx[gidx]
        ghost_neighbor_elem[i] = ridx
    end

    # 5) faces
    boundaries = NamedTuple(
        boundary_name => Tuple{Int, Int}[] for
        boundary_name in Meshes.boundary_names(mesh)
    )
    interior_faces = Tuple{Int, Int, Int, Int, Bool}[]
    ghost_faces = Tuple{Int, Int, Int, Int, Bool}[]
    for (lidx, gidx) in enumerate(local_elem_gidx)
        elem = elemorder[gidx]
        for face in 1:4
            if Meshes.is_boundary_face(mesh, elem, face)
                boundary_name = Meshes.boundary_face_name(mesh, elem, face)
                push!(boundaries[boundary_name], (lidx, face))
            else
                oelem, oface, reversed = Meshes.opposing_face(mesh, elem, face)
                ogidx = orderindex[oelem]
                opid = elempid[ogidx]
                if opid == pid
                    if (ogidx, oface) < (gidx, face)
                        # we've already seen this face
                        continue
                    end
                    olidx = global_elem_lidx[ogidx]
                    push!(interior_faces, (lidx, face, olidx, oface, reversed))
                else
                    ridx = global_elem_ridx[ogidx]
                    push!(ghost_faces, (lidx, face, ridx, oface, reversed))
                end
            end
        end
    end
    # 6). 
    comm_vertex_lengths = zeros(Int, length(send_elem_pids))
    ghost_vertex_neighbor_loc = Int[]
    ghost_vertex_comm_idx_offset = ones(Int, length(ghost_vertex_offset))
    ghost_vertex_gcidx = zeros(Int, length(ghost_vertex_offset) - 1)
    repr_ghost_vertex = Tuple{Int, Int}[] # representative local ghost vertex (idx, vert) for each unique ghost vertex
    perimeter_elems = Int[]
    for (i, vertex) in
        enumerate(VertexIterator(ghost_vertices, ghost_vertex_offset))
        procs = Int[]
        repr_assigned = false
        for (isghost, idx, lvert) in vertex
            if isghost
                push!(procs, elempid[recv_elem_gidx[idx]])
            else
                if !repr_assigned
                    push!(repr_ghost_vertex, (idx, lvert))
                    repr_assigned = true
                end
                push!(perimeter_elems, idx) # mark perimeter elements
            end
            if ghost_vertex_gcidx[i] == 0
                ghost_vertex_gcidx[i] =
                    isghost ? global_vertex_gidx[lvert, recv_elem_gidx[idx]] :
                    global_vertex_gidx[lvert, local_elem_gidx[idx]]
            end
        end
        unique!(procs)
        ghost_vertex_comm_idx_offset[i + 1] =
            ghost_vertex_comm_idx_offset[i] + length(procs)
        for pr in procs
            loc = findfirst(send_elem_pids .== pr)
            push!(ghost_vertex_neighbor_loc, loc)
            comm_vertex_lengths[loc] += 1
        end
    end
    unique!(perimeter_elems)
    internal_elems = setdiff(1:length(local_elem_gidx), perimeter_elems)
    # 7). 
    comm_face_lengths = zeros(Int, length(send_elem_pids))
    ghost_face_neighbor_loc = Vector{Int}(undef, length(ghost_faces))
    ghost_face_gcidx = zeros(Int, length(ghost_faces))
    for (i, (e, face, o, oface, reversed)) in enumerate(ghost_faces)
        ghostelem = recv_elem_gidx[o]
        loc = findfirst(send_elem_pids .== elempid[ghostelem])
        comm_face_lengths[loc] += 1
        ghost_face_neighbor_loc[i] = loc
        ghost_face_gcidx[i] = global_face_gidx[face, local_elem_gidx[e]]
    end

    return Topology2D(
        context,
        mesh,
        elemorder,
        orderindex,
        elempid,
        local_elem_gidx,
        send_elem_pids,
        send_elem_lidx,
        send_elem_lengths,
        recv_elem_gidx,
        recv_elem_lengths,
        DA(interior_faces),
        ghost_faces,
        DA(local_vertices),
        DA(local_vertex_offset),
        DA(ghost_vertices),
        DA(ghost_vertex_offset),
        DA(local_neighbor_elem),
        DA(local_neighbor_elem_offset),
        ghost_neighbor_elem,
        ghost_neighbor_elem_offset,
        boundaries,
        internal_elems,
        perimeter_elems,
        nglobalvertices,
        nglobalfaces,
        ghost_vertex_gcidx,
        ghost_face_gcidx,
        comm_vertex_lengths,
        comm_face_lengths,
        ghost_vertex_neighbor_loc,
        ghost_vertex_comm_idx_offset,
        DA(repr_ghost_vertex),
        ghost_face_neighbor_loc,
    )
end

perimeter_vertex_node_index(v) = v
perimeter_face_indices(f, nfacedof, reversed = false) =
    !reversed ? ((4 + (f - 1) * nfacedof + 1):(4 + f * nfacedof)) :
    ((4 + f * nfacedof):-1:(4 + (f - 1) * nfacedof + 1))
perimeter_face_indices_cuda(f, nfacedof, reversed = false) =
    !reversed ? ((4 + (f - 1) * nfacedof + 1), 1, (4 + f * nfacedof)) :
    ((4 + f * nfacedof), -1, (4 + (f - 1) * nfacedof + 1))



function compute_ghost_send_recv_idx(topology::Topology2D, Nq)
    DA = ClimaComms.array_type(topology)
    (;
        context,
        neighbor_pids,
        comm_vertex_lengths,
        comm_face_lengths,
        ghost_vertex_gcidx,
        ghost_face_gcidx,
        ghost_vertex_neighbor_loc,
        ghost_vertex_comm_idx_offset,
        ghost_faces,
        ghost_face_neighbor_loc,
        nglobalvertices,
        nglobalfaces,
    ) = topology
    ghost_vertices = Array(topology.ghost_vertices)
    ghost_vertex_offset = Array(topology.ghost_vertices)
    repr_ghost_vertex = Array(topology.repr_ghost_vertex)
    nfacedof = Nq - 2
    comm_lengths = comm_vertex_lengths .+ (comm_face_lengths .* nfacedof)
    ghost_face_ugidx = ghost_face_gcidx .+ nglobalvertices # unique id for both vertices and faces
    send_data = Array{Int}(undef, sum(comm_lengths))
    recv_data = similar(send_data)

    graph_context = ClimaComms.graph_context(
        context,
        send_data,
        comm_lengths,
        neighbor_pids,
        recv_data,
        comm_lengths,
        neighbor_pids,
    )
    send_offset = Int[1]
    append!(send_offset, cumsum(comm_lengths) .+ 1)
    send_idx = deepcopy(send_offset)
    send_buf_idx = zeros(Int, sum(comm_lengths), 2)
    # load send buffer with global vertex idx
    for (i, gidx) in enumerate(ghost_vertex_gcidx)
        st, en = ghost_vertex_comm_idx_offset[i:(i + 1)]
        for loc in ghost_vertex_neighbor_loc[st:(en - 1)]
            send_data[send_idx[loc]] = gidx
            (lidx, lvert) = repr_ghost_vertex[i]
            perimeterloc = perimeter_vertex_node_index(lvert)
            send_buf_idx[send_idx[loc], :] .= [lidx, perimeterloc]
            send_idx[loc] += 1
        end
    end
    # load send buffer with global face idx (shifted)
    for (i, gidx) in enumerate(ghost_face_ugidx)
        loc = ghost_face_neighbor_loc[i]
        (e, face, _, _, _) = ghost_faces[i]
        prange = perimeter_face_indices(face, nfacedof)
        send_data[send_idx[loc]] = gidx
        send_buf_idx[send_idx[loc], :] .= [e, prange[1]]
        send_idx[loc] += 1
        for j in 2:nfacedof
            send_data[send_idx[loc]] = 0
            send_buf_idx[send_idx[loc], :] .= [e, prange[j]]
            send_idx[loc] += 1
        end
    end
    ClimaComms.start(graph_context)
    ClimaComms.progress(graph_context)
    ClimaComms.finish(graph_context)
    # setup send and recv indexes for ghost vertices
    recv_buf_idx = zeros(Int, sum(comm_lengths), 2)
    ghost_vertex_send_idx = similar(ghost_vertex_neighbor_loc)
    ghost_vertex_recv_idx = similar(ghost_vertex_neighbor_loc)

    for (i, gidx) in enumerate(ghost_vertex_gcidx)
        st, en = ghost_vertex_comm_idx_offset[i:(i + 1)]
        for (ctr, loc) in enumerate(ghost_vertex_neighbor_loc[st:(en - 1)])
            offset = send_offset[loc]
            ghost_vertex_send_idx[st + ctr - 1] =
                offset + findfirst(send_data[offset:end] .== gidx) - 1
            ghost_vertex_recv_idx[st + ctr - 1] =
                offset + findfirst(recv_data[offset:end] .== gidx) - 1
            (lidx, lvert) = repr_ghost_vertex[i]
            perimeterloc = perimeter_vertex_node_index(lvert)
            recv_buf_idx[ghost_vertex_recv_idx[st + ctr - 1], :] .=
                [lidx, perimeterloc]
        end
    end
    # setup send and recv indexes for ghost vertices
    ghost_face_send_idx = similar(ghost_face_neighbor_loc)
    ghost_face_recv_idx = similar(ghost_face_neighbor_loc)

    for (i, gidx) in enumerate(ghost_face_ugidx)
        loc = ghost_face_neighbor_loc[i]
        offset = send_offset[loc]
        (e, face, _, _, _) = ghost_faces[i]
        prange = perimeter_face_indices(face, nfacedof, true) # reverse face data (double check)
        ghost_face_send_idx[i] =
            offset + findfirst(send_data[offset:end] .== gidx) - 1
        ghost_face_recv_idx[i] =
            offset + findfirst(recv_data[offset:end] .== gidx) - 1
        for j in 1:nfacedof
            recv_buf_idx[ghost_face_recv_idx[i] + j - 1, :] = [e, prange[j]]
        end
    end
    return DA(send_buf_idx), DA(recv_buf_idx)
end

domain(topology::Topology2D) = domain(topology.mesh)
nelems(topology::Topology2D) = length(topology.elemorder)
nlocalelems(topology::Topology2D) = length(topology.local_elem_gidx)
function localelems(topology::Topology2D)
    topology.elemorder[topology.local_elem_gidx]
end
nghostelems(topology::Topology2D) = length(topology.recv_elem_gidx)
function ghostelems(topology::Topology2D)
    topology.elemorder[topology.recv_elem_gidx]
end

nneighbors(topology::Topology2D) = length(topology.neighbor_pids)



nsendelems(topology::Topology2D) = length(topology.send_elem_lidx)
nrecvelems(topology::Topology2D) = length(topology.recv_elem_gidx)

function localelemindex(topology::Topology2D, elem)
    idx = topology.localorderindex[topology.orderindex[elem]]
    if idx == 0
        error("requested local element index for non-local element $(elem)")
    end
    return idx
end

function vertex_coordinates(topology::Topology2D, e::Int)
    ntuple(4) do vert
        Meshes.coordinates(topology.mesh, topology.elemorder[e], vert)
    end
end

coordinates(
    topology::Topology2D{<:ClimaComms.SingletonCommsContext},
    e::Int,
    arg,
) = coordinates(topology.mesh, topology.elemorder[e], arg)

function opposing_face(topology::Topology2D, e::Int, face::Int)
    mesh = topology.mesh
    elem = topology.elemorder[e]
    if Meshes.is_boundary_face(mesh, elem, face)
        boundary_name = Meshes.boundary_face_name(mesh, elem, face)
        b = findfirst(==(boundary_name), keys(topology.boundaries))
        return (0, b, false)
    end
    opelem, opface, reversed = Meshes.opposing_face(mesh, elem, face)
    return (topology.orderindex[opelem], opface, reversed)
end
interior_faces(topology::Topology2D) = topology.interior_faces
ghost_faces(topology::Topology2D) = topology.ghost_faces

local_vertices(topology::Topology2D) =
    VertexIterator(topology.local_vertices, topology.local_vertex_offset)
ghost_vertices(topology::Topology2D) =
    VertexIterator(topology.ghost_vertices, topology.ghost_vertex_offset)

function local_neighboring_elements(topology::Topology2D, elem)
    return view(
        topology.local_neighbor_elem,
        topology.local_neighbor_elem_offset[elem]:(topology.local_neighbor_elem_offset[elem + 1] - 1),
    )
end
function ghost_neighboring_elements(topology::Topology2D, elem)
    return view(
        topology.ghost_neighbor_elem,
        topology.ghost_neighbor_elem_offset[elem]:(topology.ghost_neighbor_elem_offset[elem + 1] - 1),
    )
end


neighbors(topology::Topology2D) = topology.neighbor_pids
boundary_names(topology::Topology2D) = keys(topology.boundaries)
boundary_tags(topology::Topology2D) = NamedTuple{boundary_names(topology)}(
    ntuple(i -> i, length(topology.boundaries)),
)
boundary_tag(topology::Topology2D, boundary_name::Symbol) =
    findfirst(==(boundary_name), boundary_names(topology))

boundary_faces(topology::Topology2D, boundary) = topology.boundaries[boundary]




abstract type AbstractPerimeter end

struct Perimeter2D{Nq} <: AbstractPerimeter end

"""
    Perimeter2D(Nq)

Construct a perimeter iterator for a 2D spectral element with `Nq` nodes per
dimension (i.e. polynomial degree `Nq-1`).
"""
Perimeter2D(Nq) = Perimeter2D{Nq}()

Adapt.adapt_structure(to, x::Perimeter2D) = x

Base.IteratorEltype(::Type{Perimeter2D{Nq}}) where {Nq} = Base.HasEltype()
Base.eltype(::Type{Perimeter2D{Nq}}) where {Nq} = Tuple{Int, Int}

Base.IteratorSize(::Type{Perimeter2D{Nq}}) where {Nq} = Base.HasLength()
Base.length(::Perimeter2D{Nq}) where {Nq} = 4 + (Nq - 2) * 4

function Base.iterate(perimeter::Perimeter2D{Nq}, loc = 1) where {Nq}
    if loc <= 4
        return (vertex_node_index(loc, Nq), loc + 1)
    elseif loc ≤ length(perimeter)
        f = cld(loc - 4, Nq - 2)
        n = mod(loc - 4, Nq - 2) == 0 ? (Nq - 2) : mod(loc - 4, Nq - 2)
        return (face_node_index(f, Nq, 1 + n), loc + 1)
    else
        return nothing
    end
end

function Base.getindex(perimeter::Perimeter2D{Nq}, loc = 1) where {Nq}
    if loc < 1 || loc > length(perimeter)
        return (-1, -1)
    elseif loc <= 4
        return vertex_node_index(loc, Nq)
    else
        f = cld(loc - 4, Nq - 2)
        n = mod(loc - 4, Nq - 2) == 0 ? (Nq - 2) : mod(loc - 4, Nq - 2)
        return face_node_index(f, Nq, 1 + n)
    end
end


## aliases
const RectilinearTopology2D =
    Topology2D{<:ClimaComms.AbstractCommsContext, <:Meshes.RectilinearMesh}
const CubedSphereTopology2D =
    Topology2D{<:ClimaComms.AbstractCommsContext, <:Meshes.AbstractCubedSphere}
