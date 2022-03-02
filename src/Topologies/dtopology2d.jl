using ClimaComms

"""
    DistributedTopology2D(mesh::AbstractMesh2D, elemorder=Mesh.elements(mesh))

This is a distributed topology for 2D meshes. `elemorder` is a vector or
other linear ordering of the `Mesh.elements(mesh)`. `elempid` is a sorted
vector of the same length as `elemorder`, each element of which contains
the `pid` of the owning process.
"""
struct DistributedTopology2D{
    M <: Meshes.AbstractMesh{2},
    EO,
    OI,
    EP,
    EL,
    LO,
    NE,
    BF,
} <: AbstractDistributedTopology
    "mesh on which the topology is constructed"
    mesh::M
    "`elemorder[e]` should give the `e`th element"
    elemorder::EO
    "the inverse of `elemorder`: `orderindex[elemorder[e]] == e`"
    orderindex::OI
    "`elempid[e]` gives the PID that owns the `e`th element"
    elempid::EP
    "the elements that are local to this process"
    local_elems::EL
    "the inverse of `local_elems`: `localorderindex[local_elems[e]] == e`"
    localorderindex::LO
    "the PIDs of neighboring processes"
    neighbor_pids::Vector{Int}
    "local idx of neighboring processes"
    neighbor_pid_idx::Dict{Int, Int}
    "the elements that must be sent to neighboring processes"
    send_elems::NE
    "ghost elements, i.e. those received from neighboring processes"
    ghost_elems::NE
    "a vector of all unique interior faces, (e, face, o, oface, reversed)"
    interior_faces::Vector{Tuple{Int, Int, Int, Int, Bool}}
    "a vector of all ghost faces, (e, face, o, oface, reversed)"
    ghost_faces::Vector{Tuple{Int, Int, Int, Int, Bool}}
    "the collection of local (e,vert) pairs"
    interior_vertices::Vector{Vector{Tuple{Int, Int}}}
    "the collection of ghost (e,vert) pairs"
    ghost_vertices::Vector{Vector{Tuple{Int, Int}}}
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


# naming decisions:
#  - elemorder and orderindex
#  - realelems / localelems
#  - interior faces / internal faces

function DistributedTopology2D(
    mesh::Meshes.AbstractMesh{2},
    Context::Type{<:ClimaComms.AbstractCommsContext},
    elemorder = Meshes.elements(mesh), # an iterator of mesh elements (typically a map Int => CartesianIndex)
    elempid = nothing, # array of same size as elemorder, containing owning pid for each element (it should be sorted)
    orderindex = Meshes.linearindices(elemorder), # inverse mapping of elemorder (e.g. map CartesianIndex => Int)
)
    pid = ClimaComms.mypid(Context)
    nprocs = ClimaComms.nprocs(Context)
    nelements = length(elemorder)
    if isnothing(elempid)
        elempid = simple_partition(elemorder, nprocs)
    end
    @assert issorted(elempid)
    EO = eltype(elemorder)

    # put elements belonging to the local process into `local_elems`
    local_elems = EO[]
    # XXX: is there a better way to do this? `localorderindex` can get pretty big
    # contains
    # - local element number (in 1:nlocalelems) if local
    # - 0 if ghost element
    localorderindex = Vector{Int}(undef, nelements)
    for (i, owner_pid) in enumerate(elempid)
        if owner_pid == pid
            push!(local_elems, elemorder[i])
            localorderindex[i] = length(local_elems)
        else
            localorderindex[i] = 0
        end
    end

    neighbor_pids = Int[]
    # find index in neighbor_pids
    find_neighbor(pid) = findfirst(npid -> npid == pid, neighbor_pids)
    send_elems = Set{EO}[]
    ghost_elems = Set{EO}[]
    interior_faces = Set{Tuple{Int, Int, Int, Int, Bool}}() # faces which both sides are local
    ghost_faces = Set{Tuple{Int, Int, Int, Int, Bool}}() # faces in which one side is local, one side is ghost
    interior_vertices = Vector{Tuple{Int, Int}}[]
    ghost_vertices = Vector{Tuple{Int, Int}}[]
    boundaries = NamedTuple(
        boundary_name => Tuple{Int, Int}[] for
        boundary_name in unique(Meshes.boundary_names(mesh))
    )

    iface_seen = BitArray(undef, (nelements, 4))
    fill!(iface_seen, 0)
    gface_seen = BitArray(undef, (nelements, 4))
    fill!(gface_seen, 0)
    ivert_seen = BitArray(undef, (nelements, 4))
    fill!(ivert_seen, 0)
    gvert_seen = BitArray(undef, (nelements, 4))
    fill!(gvert_seen, 0)

    for (e, elem) in enumerate(local_elems)
        oe = orderindex[elem]

        # Determine whether a face is a boundary face, an interior face
        # (the opposing face is of a local element), or a ghost face (the
        # opposing face is of a ghost element). However, we do not
        # determine the set of ghost elements (or the set of elements to
        # send) here, but when dealing with vertices (below).
        for face in 1:4
            if Meshes.is_boundary_face(mesh, elem, face)
                boundary_name = Meshes.boundary_face_name(mesh, elem, face)
                push!(boundaries[boundary_name], (e, face))
            else
                opelem, opface, reversed =
                    Meshes.opposing_face(mesh, elem, face)
                oi = orderindex[opelem]
                if elempid[oi] == pid
                    iface_seen[oe, face] && continue
                    iface_seen[oe, face] = iface_seen[oi, opface] = true
                    push!(interior_faces, (oe, face, oi, opface, reversed))
                else
                    gface_seen[oe, face] && continue
                    gface_seen[oe, face] = gface_seen[oi, opface] = true
                    push!(ghost_faces, (oe, face, oi, opface, reversed))
                end
            end
        end

        # Determine whether a vertex is an interior or a ghost vertex. For
        # ghost vertices, we record the elements to be sent, the elements
        # to be received (ghost elements) and other information to ease DSS.
        for vert in 1:4
            isghostvertex =
                any(Meshes.SharedVertices(mesh, elem, vert)) do (velem, vvert)
                    oi = orderindex[velem]
                    elempid[oi] != pid
                end
            if isghostvertex
                # one or more elements sharing this vertex are ghost elements
                gvert_seen[oe, vert] && continue
                gvert_seen[oe, vert] = true

                # record all the shared vertices as ghost vertices and record all
                # the neighbor PIDs so that all the elements sharing this vertex
                # can later be added to `send_elems` or `ghost_elems`
                gvert_group = Tuple{Int, Int}[]
                nbr_idxs = Set{Int}()
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    oi = orderindex[velem]
                    push!(gvert_group, (oi, vvert))
                    gvert_seen[oi, vvert] = true

                    # identify the owner of this element
                    opid = elempid[oi]
                    if opid != pid
                        # find the index into neighbor_pids for the owner and
                        # add it if not present
                        nbr_idx = find_neighbor(opid)
                        if nbr_idx === nothing
                            push!(neighbor_pids, opid)
                            # contains the elements to send to the neighbor
                            push!(send_elems, Set{EO}())
                            # contains the elements recieved by neighbor
                            push!(ghost_elems, Set{EO}())
                            nbr_idx = length(neighbor_pids)
                        end
                        push!(nbr_idxs, nbr_idx)
                    end
                end
                # push the neighbors of this vertex element
                push!(ghost_vertices, sort(gvert_group))

                # record the `send_elems` and `ghost_elems`; this must be
                # done in a second pass to avoid missing elements
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    oi = orderindex[velem]
                    opid = elempid[oi]
                    if opid == pid
                        # this is a local element; it has to be sent to all the
                        # neighbors that share this vertex
                        for nbr_idx in nbr_idxs
                            push!(send_elems[nbr_idx], velem)
                        end
                    else
                        # this is a ghost element; it has to be received from the
                        # owning neighbor
                        nbr_idx = find_neighbor(opid)
                        push!(ghost_elems[nbr_idx], velem)
                    end
                end
                empty!(nbr_idxs)
            else
                # all elements sharing this vertex are local
                ivert_seen[oe, vert] && continue
                ivert_seen[oe, vert] = true
                ivert_group = Tuple{Int, Int}[]
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    oi = orderindex[velem]
                    push!(ivert_group, (oi, vvert))
                    ivert_seen[oi, vvert] = true
                end
                push!(interior_vertices, sort(ivert_group))
            end
        end
    end

    send_elems_ra = [sort(collect(elemset)) for elemset in send_elems]
    ghost_elems_ra = [sort(collect(elemset)) for elemset in ghost_elems]

    neighbor_pid_idx = Dict{Int, Int}()
    for (i, neighbor_pid) in enumerate(neighbor_pids)
        neighbor_pid_idx[neighbor_pid] = i
    end

    return DistributedTopology2D(
        mesh,
        elemorder,
        orderindex,
        elempid,
        local_elems,
        localorderindex,
        neighbor_pids,
        neighbor_pid_idx,
        send_elems_ra,
        ghost_elems_ra,
        collect(interior_faces),
        collect(ghost_faces),
        collect(interior_vertices),
        collect(ghost_vertices),
        boundaries,
    )
end

domain(topology::DistributedTopology2D) = domain(topology.mesh)
nelems(topology::DistributedTopology2D) = length(topology.elemorder)
nlocalelems(topology::DistributedTopology2D) = length(topology.local_elems)
nneighbors(topology::DistributedTopology2D) = length(topology.neighbor_pids)
nsendelems(topology::DistributedTopology2D) =
    sum(map(length, topology.send_elems))
nsendelems(topology::DistributedTopology2D, idx) =
    length(topology.send_elems[idx])
nghostelems(topology::DistributedTopology2D) =
    sum(map(length, topology.ghost_elems))
nghostelems(topology::DistributedTopology2D, idx) =
    length(topology.ghost_elems[idx])

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

interior_vertices(topology::DistributedTopology2D) = topology.interior_vertices
ghost_vertices(topology::DistributedTopology2D) = topology.ghost_vertices

# XXX: Should the non-specific `VertexIterator` be removed? Made it refer
# to interior vertices for now.
Base.length(vertiter::VertexIterator{<:DistributedTopology2D}) =
    length(vertiter.topology.interior_vertices)
Base.eltype(::VertexIterator{T}) where {T <: DistributedTopology2D} =
    Vertex{T, Int}
function Base.iterate(
    vertiter::VertexIterator{<:DistributedTopology2D},
    uvert = 1,
)
    topology = vertiter.topology
    if uvert >= length(topology.interior_vertices)
        return nothing
    end
    return Vertex(topology, uvert), uvert + 1
end

Base.length(vertiter::InteriorVertexIterator{<:DistributedTopology2D}) =
    length(vertiter.topology.interior_vertices)
Base.eltype(::InteriorVertexIterator{T}) where {T <: DistributedTopology2D} =
    Vertex{T, Int}
function Base.iterate(
    vertiter::InteriorVertexIterator{<:DistributedTopology2D},
    uvert = 1,
)
    topology = vertiter.topology
    if uvert >= length(topology.interior_vertices)
        return nothing
    end
    return Vertex(topology, uvert), uvert + 1
end

Base.length(vertiter::GhostVertexIterator{<:DistributedTopology2D}) =
    length(vertiter.topology.ghost_vertices)
Base.eltype(::GhostVertexIterator{T}) where {T <: DistributedTopology2D} =
    Vertex{T, Int}
function Base.iterate(
    vertiter::GhostVertexIterator{<:DistributedTopology2D},
    uvert = 1,
)
    topology = vertiter.topology
    if uvert >= length(topology.interior_vertices)
        return nothing
    end
    return Vertex(topology, uvert), uvert + 1
end

Base.length(vertex::Vertex{<:DistributedTopology2D}) =
    length(vertex.topology.interior_vertices[vertex.num])
Base.eltype(vertex::Vertex{<:DistributedTopology2D}) =
    eltype(eltype(vertex.topology.interior_vertices))
function Base.iterate(vertex::Vertex{<:DistributedTopology2D}, idx = 1)
    if idx > length(vertex.topology.interior_vertices[vertex.num])
        return nothing
    end
    return vertex.topology.interior_vertices[vertex.num][idx], idx + 1
end

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
