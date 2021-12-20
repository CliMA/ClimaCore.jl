using ClimaComms

using OrderedCollections

"""
    DistributedTopology2D(mesh::AbstractMesh2D, elemorder=Mesh.elements(mesh))

This is a distributed topology for 2D meshes. `elemorder` is a vector or
other linear ordering of the `Mesh.elements(mesh)`. `elempid` is a sorted
vector of the same length as `elemorder`, each element of which contains
the `pid` of the owning process.
"""
struct DistributedTopology2D{M <: Meshes.AbstractMesh{2}, EO, OI, EL, PE, BF} <:
       AbstractTopology
    "mesh on which the topology is constructed"
    mesh::M
    "`elemorder[e]` should give the `e`th element "
    elemorder::EO
    "the inverse of `elemorder`: `orderindex[elemorder[e]] == e`"
    orderindex::OI
    "the elements that are local to this process"
    real_elems::EL
    "the elements that must be sent to neighboring processes"
    send_elems::PE
    "ghost elements, i.e. those received from neighboring processes"
    ghost_elems::PE
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


function DistributedTopology2D(
    mesh::Meshes.AbstractMesh{2},
    Context::Type{<:ClimaComms.AbstractCommsContext},
    elemorder = Meshes.elements(mesh),
    elempid = nothing, # array of same size as elemorder, containing owning pid for each element (it should be sorted)
    orderindex = Meshes.linearindices(elemorder),
)
    pid, nprocs = ClimaComms.init(Context)
    if isnothing(elempid)
        elempid = simple_partition(elemorder, nprocs)
    end
    @assert issorted(elempid)
    EO = eltype(elemorder)

    # put elements belonging to the local process into `real_elems`
    real_elems = EO[]
    for (i, owner_pid) in enumerate(elempid)
        if owner_pid == pid
            push!(real_elems, elemorder[i])
        end
    end
    send_elems = Dict{Int, OrderedSet{EO}}() # pid => set of local elements to be sent to pid
    ghost_elems = Dict{Int, OrderedSet{EO}}() # pid => set of elements to be received from pid
    interior_faces = OrderedSet{Tuple{Int, Int, Int, Int, Bool}}() # faces which both sides are local
    ghost_faces = OrderedSet{Tuple{Int, Int, Int, Int, Bool}}() # faces in which one side is local, one side is ghost
    interior_vertices = Vector{Tuple{Int, Int}}[]
    ghost_vertices = Vector{Tuple{Int, Int}}[]
    boundaries = NamedTuple(
        boundary_name => Tuple{Int, Int}[] for
        boundary_name in unique(Meshes.boundary_names(mesh))
    )

    iface_seen = BitArray(undef, (length(elemorder), 4))
    fill!(iface_seen, 0)
    gface_seen = BitArray(undef, (length(elemorder), 4))
    fill!(gface_seen, 0)
    ivert_seen = BitArray(undef, (length(elemorder), 4))
    fill!(ivert_seen, 0)
    gvert_seen = BitArray(undef, (length(elemorder), 4))
    fill!(gvert_seen, 0)

    for (e, elem) in enumerate(real_elems)
        oe = orderindex[elem]

        # Determine whether a face is a boundary face, an interior face
        # (the opposing face is of a real element), or a ghost face (the
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
                if opelem ∈ real_elems # this is doing a linear search through elems: TODO could we improve this?
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
        # ghost vertices, we record the elements to be sent as well as the
        # ghost elements to be received (together with the pid of the
        # destination/owner process).
        for vert in 1:4
            if any(Meshes.SharedVertices(mesh, elem, vert)) do (velem, vvert)
                oi = orderindex[velem]
                elempid[oi] != pid
            end
                # one or more elements sharing this vertex are ghost elements
                gvert_seen[oe, vert] && continue
                gvert_seen[oe, vert] = true
                gvert_group = Tuple{Int, Int}[]
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    oi = orderindex[velem]
                    push!(gvert_group, (oi, vvert))
                    gvert_seen[oi, vvert] = true
                    opid = elempid[oi]
                    if opid != pid
                        seset = get!(send_elems, opid, OrderedSet{EO}())
                        push!(seset, elem)
                        geset = get!(ghost_elems, opid, OrderedSet{EO}())
                        push!(geset, velem)
                    end
                    # TODO: We need to group ghost_vertices by their unique shared vertex
                end
                push!(ghost_vertices, sort(gvert_group))
            else
                # all elements sharing this vertex are real
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
    println("$pid> iv: $interior_vertices")

    # build ragged arrays out of the dictionaries
    send_elems_ra = [(k, collect(v)) for (k, v) in send_elems]
    ghost_elems_ra = [(k, collect(v)) for (k, v) in ghost_elems]

    return DistributedTopology2D(
        mesh,
        elemorder,
        orderindex,
        real_elems,
        send_elems_ra,
        ghost_elems_ra,
        collect(interior_faces),
        collect(ghost_faces),
        collect(interior_vertices),
        collect(ghost_vertices),
        boundaries,
    )
end

# XXX: added `nelems` for real+ghost; added nrealelems, nghostelems;
# make a choice about naming: local <-> real?
domain(topology::DistributedTopology2D) = domain(topology.mesh)
nelems(topology::DistributedTopology2D) = length(topology.elemorder)
nlocalelems(topology::DistributedTopology2D) = length(topology.real_elems)
nrealelems(topology::DistributedTopology2D) = length(topology.real_elems)
nghostelems(topology::DistributedTopology2D) = length(topology.ghost_elems)

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

# XXX: we need `length()`, `eltype()` and `iterate()` for faces?

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

# XXX: added `InteriorVertexIterator`
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

# XXX: added `GhostVertexIterator`
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

boundary_names(topology::DistributedTopology2D) = keys(topology.boundaries)
boundary_tags(topology::DistributedTopology2D) =
    NamedTuple{boundary_names(topology)}(
        ntuple(i -> i, length(topology.boundaries)),
    )
boundary_tag(topology::DistributedTopology2D, boundary_name::Symbol) =
    findfirst(==(boundary_name), boundary_names(topology))

boundary_faces(topology::DistributedTopology2D, boundary) =
    topology.boundaries[boundary]
