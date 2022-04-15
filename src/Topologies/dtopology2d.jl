using ClimaComms, DataStructures

"""
    DistributedTopology2D(mesh::AbstractMesh2D, elemorder=Mesh.elements(mesh))

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
struct DistributedTopology2D{
    C <: ClimaComms.AbstractCommsContext,
    M <: Meshes.AbstractMesh{2},
    EO,
    OI,
    BF,
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
    local_elem_gidx::Vector{Int}
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
    interior_faces::Vector{Tuple{Int, Int, Int, Int, Bool}}
    "a vector of all ghost faces, (e, face, o, oface, reversed)"
    ghost_faces::Vector{Tuple{Int, Int, Int, Int, Bool}}
    "the collection of `(lidx, vert)`` pairs of vertices of local elements which touch a ghost element"
    local_vertices::Vector{Tuple{Int, Int}}
    "the index in `local_vertices` of the first tuple for each unique vertex"
    local_vertex_offset::Vector{Int}
    """
    The collection of `(isghost, idx, vert)` tuples of vertices of local elements which touch a ghost element.
    If `isghost` is false, then `idx` is the `lidx`, otherwise it is the `ridx`
    """
    ghost_vertices::Vector{Tuple{Bool, Int, Int}}
    "the index in `ghost_vertices` of the first tuple for each unique vertex"
    ghost_vertex_offset::Vector{Int}
    "a NamedTuple of vectors of `(e,face)` "
    boundaries::BF
end

# returns partition[1..nelems] where partition[e] = pid of the owning process
function simple_partition(elemorder, npart::Int)
    nelems = length(elemorder)
    partition = zeros(Int, nelems)
    quot, rem = divrem(nelems, npart)
    part_len = ones(Int, nelems) .* quot
    if rem ≠ 0
        part_len[1:rem] .+= 1
    end
    ctr = 1
    for i in 1:npart
        for j in 1:part_len[i]
            partition[ctr] = i
            ctr += 1
        end
    end
    return partition
end


function DistributedTopology2D(
    context::ClimaComms.AbstractCommsContext,
    mesh::Meshes.AbstractMesh{2},
    elemorder = Meshes.elements(mesh), # an iterator of mesh elements (typically a map Int => CartesianIndex)
    elempid = nothing, # array of same size as elemorder, containing owning pid for each element (it should be sorted)
    orderindex = Meshes.linearindices(elemorder), # inverse mapping of elemorder (e.g. map CartesianIndex => Int)
)
    pid = ClimaComms.mypid(context)
    nprocs = ClimaComms.nprocs(context)
    nelements = length(elemorder)
    if isnothing(elempid)
        elempid = simple_partition(elemorder, nprocs)
    end
    @assert issorted(elempid)
    EO = eltype(elemorder)

    local_elem_gidx = Int[] # gidx = local_elem_gidx[lidx]
    global_elem_lidx = Dict{Int, Int}() # inverse of local_elem_gidx: lidx = global_elem_lidx[gidx]
    nlocalelems = 0
    for (gidx, owner_pid) in enumerate(elempid)
        if owner_pid == pid
            nlocalelems += 1
            push!(local_elem_gidx, gidx)
            global_elem_lidx[gidx] = nlocalelems
        end
    end


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
    temp_elem = Tuple{Int, Int, Int}[]

    # 1) iterate over the vertices of local elements to determeind the vertex connectivity
    #    since we don't yet know the ridx of halo elements, we instead store the gidx.
    for gidx in local_elem_gidx
        elem = elemorder[gidx]
        # Determine whether a vertex is an interior or a ghost vertex. For
        # ghost vertices, we record the elements to be sent, the elements
        # to be received (ghost elements) and other information to ease DSS.
        for vert in 1:4
            isghostvertex = false
            empty!(temp_elem)
            for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                vgidx = orderindex[velem]
                vpid = elempid[vgidx]
                if vpid == pid
                    if (vgidx, vvert) < (gidx, vert)
                        # we've already seen this vertex
                        @goto skip
                    end
                    vlidx = global_elem_lidx[vgidx]
                    push!(temp_elem, (0, vlidx, vvert))
                else
                    isghostvertex = true
                    push!(recv_elem_set, (vpid, vgidx))
                    push!(temp_elem, (vpid, vgidx, vvert))
                end
            end
            # append to either local_vertices or ghost_vertices
            if isghostvertex
                for (vpid, vidx, vvert) in temp_elem
                    # vidx will be lidx if local, gidx if ghost
                    # replace the gidx with ridx in step 4.
                    push!(ghost_vertices, (vpid != 0, vidx, vvert))
                    # if it's a ghost_vertex, push every (dest pid, lidx) to the send_elem_set
                    if vpid != 0
                        for (xpid, xidx, _) in temp_elem
                            if xpid == 0
                                push!(send_elem_set, (vpid, xidx))
                            end
                        end
                    end
                end
                push!(ghost_vertex_offset, length(ghost_vertices) + 1)
            else
                for (_, vlidx, vvert) in temp_elem
                    push!(local_vertices, (vlidx, vvert))
                end
                push!(local_vertex_offset, length(local_vertices) + 1)
            end
            @label skip
        end
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

    return DistributedTopology2D(
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
        interior_faces,
        ghost_faces,
        local_vertices,
        local_vertex_offset,
        ghost_vertices,
        ghost_vertex_offset,
        boundaries,
    )
end

domain(topology::DistributedTopology2D) = domain(topology.mesh)
nelems(topology::DistributedTopology2D) = length(topology.elemorder)
nlocalelems(topology::DistributedTopology2D) = length(topology.local_elem_gidx)
function localelems(topology::DistributedTopology2D)
    topology.elemorder[topology.local_elem_gidx]
end
nghostelems(topology::DistributedTopology2D) = length(topology.recv_elem_gidx)
function ghostelems(topology::DistributedTopology2D)
    topology.elemorder[topology.recv_elem_gidx]
end

nneighbors(topology::DistributedTopology2D) = length(topology.neighbor_pids)



nsendelems(topology::DistributedTopology2D) = length(topology.send_elem_lidx)
nrecvelems(topology::DistributedTopology2D) = length(topology.recv_elem_gidx)

function localelemindex(topology::DistributedTopology2D, elem)
    idx = topology.localorderindex[topology.orderindex[elem]]
    if idx == 0
        error("requested local element index for non-local element $(elem)")
    end
    return idx
end

function vertex_coordinates(topology::DistributedTopology2D, e::Int)
    ntuple(4) do vert
        Meshes.coordinates(topology.mesh, topology.elemorder[e], vert)
    end
end

function opposing_face(topology::DistributedTopology2D, e::Int, face::Int)
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
interior_faces(topology::DistributedTopology2D) = topology.interior_faces
ghost_faces(topology::DistributedTopology2D) = topology.ghost_faces

local_vertices(topology::DistributedTopology2D) =
    VertexIterator(topology.local_vertices, topology.local_vertex_offset)
ghost_vertices(topology::DistributedTopology2D) =
    VertexIterator(topology.ghost_vertices, topology.ghost_vertex_offset)

neighbors(topology::DistributedTopology2D) = topology.neighbor_pids
boundary_names(topology::DistributedTopology2D) = keys(topology.boundaries)
boundary_tags(topology::DistributedTopology2D) =
    NamedTuple{boundary_names(topology)}(
        ntuple(i -> i, length(topology.boundaries)),
    )
boundary_tag(topology::DistributedTopology2D, boundary_name::Symbol) =
    findfirst(==(boundary_name), boundary_names(topology))

boundary_faces(topology::DistributedTopology2D, boundary) =
    topology.boundaries[boundary]
