using DocStringExtensions
using UnPack
using LinearAlgebra

#export AbstractGraph,
#    Graph #, build_face_graph, build_vertex_graph, laplacian_x_v!, fiedler_vector, split_graph

abstract type AbstractGraph end
struct Lanczos end
struct PowerIt end
struct Direct end
"""
    Graph{I,FT,IA1D} <: AbstractGraph
Connectivity graph for a mesh.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct Graph{I <: Int, FT <: AbstractFloat, IA1D <: AbstractVector{I}} <:
       AbstractGraph
    "number of graph vertices"
    nverts::I
    "edge data for each vertex"
    edge_data::IA1D
    "edge data offsets for each vertex"
    edge_offset::IA1D
    "vertex numbers"
    vertices::IA1D
    "spectral radius estimate/bound of graph laplacian matrix"
    spectral_radius::FT
end

# constructor for an empty graph
Graph(::Type{I}, ::Type{FT}) where {I, FT} =
    Graph{I, FT, Vector{I}}(0, I[], I[], I[], FT(0))

"""
    build_vertex_graph(topology::AbstractTopology)

This function builds a graph, with adjacency based on mesh elements connected by a vertex.
"""
function build_vertex_graph(topology::AbstractTopology)
    I = Int
    FT = eltype(Topologies.coordinate_type(topology))
    nelems = nlocalelems(topology)
    connect = [zeros(I, i) for i in zeros(I, nelems)]
    gsize = 0
    spectral_radius = FT(0)
    for vertex in Topologies.vertices(topology)
        elems = I[]
        for (elem, vertex_num) in vertex
            push!(elems, elem)
        end
        for el in elems
            for el_neigh in elems
                if el_neigh ≠ el &&
                   findfirst(connect[el] .== el_neigh) == nothing
                    push!(connect[el], el_neigh)
                    gsize += 1
                end
            end
        end
    end
    # construct the graph object
    g_nverts = nelems
    g_edge_offset = Array{I}(undef, g_nverts + 1)
    g_edge_offset[1] = 1
    g_edge_data = Array{I}(undef, gsize)

    for el in 1:nelems
        sort!(connect[el])
        st = g_edge_offset[el]
        en = st + length(connect[el])
        g_edge_offset[el + 1] = en
        g_edge_data[st:(en - 1)] .= connect[el]
        spectral_radius = max(spectral_radius, FT(en - st))
    end
    spectral_radius *= 2
    return Graph(
        g_nverts,
        g_edge_data,
        g_edge_offset,
        Array(1:g_nverts),
        spectral_radius,
    )
end

"""
    build_face_graph(topology::AbstractTopology)

This function builds a graph, with adjacency based on mesh elements connected by a face.
"""
function build_face_graph(topology::AbstractTopology)
    I = Int
    FT = eltype(Topologies.coordinate_type(topology))
    nelems = nlocalelems(topology)
    connect = [zeros(I, i) for i in zeros(I, nelems)]
    gsize = 0
    spectral_radius = FT(0)
    for (elem1, face1, elem2, face2) in Topologies.interior_faces(topology)
        if findfirst(connect[elem1] .== elem2) == nothing
            push!(connect[elem1], elem2)
            gsize += 1
        end
        if findfirst(connect[elem2] .== elem1) == nothing
            push!(connect[elem2], elem1)
            gsize += 1
        end
    end

    # construct the graph object
    g_nverts = nelems
    g_edge_offset = Array{I}(undef, g_nverts + 1)
    g_edge_offset[1] = 1
    g_edge_data = Array{I}(undef, gsize)

    for el in 1:nelems
        sort!(connect[el])
        st = g_edge_offset[el]
        en = st + length(connect[el])
        g_edge_offset[el + 1] = en
        g_edge_data[st:(en - 1)] .= connect[el]
        spectral_radius = max(spectral_radius, FT(en - st))
    end
    spectral_radius *= 2
    return Graph(
        g_nverts,
        g_edge_data,
        g_edge_offset,
        Array(1:g_nverts),
        spectral_radius,
    )
end

"""
    laplacian_x_v!(vout::FTA1D, vin::FTA1D, grph::Graph)
This function computed the product of graph Laplacian matrix with vector `vin`.
"""
function laplacian_x_v!(
    vout::FTA1D,
    vin::FTA1D,
    grph::Graph,
) where {FT <: AbstractFloat, FTA1D <: Vector{FT}}
    @unpack nverts, edge_data, edge_offset = grph
    @assert length(vout) == length(vin) == nverts

    @inbounds for i in 1:nverts
        vout[i] = FT(0)
        @inbounds for j in edge_offset[i]:(edge_offset[i + 1] - 1)
            vout[i] += vin[i] - vin[edge_data[j]]
        end
    end
    return nothing
end
"""
    fiedler_vector(graph::Graph{<:Any,FT}, ::Lanczos)
This function estimates the Fiedler vector using Lanczos method.
Reference:
Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a Random Start
Kuczynski, Jacek; Wozniakowski, Henryk
https://academiccommons.columbia.edu/doi/10.7916/D8B56SVV
"""
function fiedler_vector(graph::Graph{<:Any, FT}, ::Lanczos) where {FT} # using Lanczos
    nv = graph.nverts
    sr = graph.spectral_radius # convervative estimate of spectral radius of graph laplacian
    rel_tol = FT(1e-2) #FT(1e-3)
    maxit = min(
        Int(ceil((2.575 / rel_tol)^0.5 * log(nv))), # Kuczynski et. al.
        nv,
        1024,
    )

    ev = Vector{FT}(undef, nv)
    start = similar(ev)
    q0 = similar(ev)
    q1 = similar(ev)
    qn = similar(ev)
    H = zeros(FT, maxit, maxit)

    [start[i] = i^(-1.0) for i in 1:nv] # initial guess
    # orthogonalize wrt constant vector
    mean = sum(start) / nv
    start .-= mean
    start ./= norm(start) # normalize
    q1 .= start

    for i in 1:maxit
        laplacian_x_v!(qn, q1, graph)
        for j in 1:nv
            qn[j] = sr * q1[j] - qn[j]
        end
        # orthogonalizing wrt cnst vector
        mean = sum(qn) / nv
        qn .-= mean

        #fac = sqrt(sum(q1.*qn))
        fac = sum(q1 .* qn)
        H[i, i] = fac

        for j in 1:nv
            qn[j] -= H[i, i] * q1[j]
        end

        if i > 1
            for j in 1:nv
                qn[j] = qn[j] - H[i - 1, i] * q0[j]
            end
        end

        fac = sqrt(sum(qn .* qn))

        if i < maxit
            H[i + 1, i] = fac
            H[i, i + 1] = fac
        end

        qn ./= fac
        q0 .= q1
        q1 .= qn
    end

    il = iu = maxit
    vl = vu = FT(0) # not used
    abstol = 1.0E-14

    eigval, eigvec =
        LinearAlgebra.LAPACK.syevr!('V', 'I', 'L', H, vl, vu, il, iu, abstol)
    lam = sr - eigval[1]
    # computing the largest eigen vector
    ev .= FT(0)
    q1 .= start

    for i in 1:maxit
        for j in 1:nv
            ev[j] += q1[j] * eigvec[i]
        end
        laplacian_x_v!(qn, q1, graph)
        for j in 1:nv
            qn[j] = sr * q1[j] - qn[j]
        end
        # orthogonalizing wrt constant vector
        mean = sum(qn) / nv
        qn .-= mean

        fac = sum(q1 .* qn)
        H[i, i] = fac

        for j in 1:nv
            qn[j] -= H[i, i] * q1[j]
        end

        if i > 1
            for j in 1:nv
                qn[j] = qn[j] - H[i - 1, i] * q0[j]
            end
        end

        fac = sum(qn .* qn)^0.5

        if i < maxit
            H[i + 1, i] = fac
            H[i, i + 1] = fac
        end

        qn ./= fac
        q0 .= q1
        q1 .= qn
    end
    ev ./= norm(ev)
    return ev
end
"""
    fiedler_vector(graph::Graph{<:Any,FT}, ::PowerIt)
This function estimates the Fiedler vector using power iteration.
"""
function fiedler_vector(graph::Graph{<:Any, FT}, ::PowerIt) where {FT} # using power_iteration
    maxit = 100
    toler = 1.0E-14
    nv = graph.nverts
    sr = graph.spectral_radius # convervative estimate of spectral radius of graph laplacian
    # initial guess for fiedler vector
    split = cld(nv, 2)
    ev = Vector{FT}(undef, nv)
    [ev[i] = -1.0 for i in 1:split]
    [ev[i] = 1.0 for i in (split + 1):nv]

    # orthogonlizing wrt constant (first) eigen vector
    shift = sum(ev) / FT(nv)
    ev .-= shift
    ev ./= norm(ev) # normalize

    ev_old = deepcopy(ev)
    ev_temp = similar(ev)

    for i in 1:(maxit - 1)
        # compute (I*sr - lmat) * ev
        laplacian_x_v!(ev_temp, ev, graph)
        for j in 1:nv
            ev[j] = ev[j] * sr - ev_temp[j]
        end
        ev ./= norm(ev) # normalize

        # orthogonlizing wrt constant (first) eigen vector
        shift = sum(ev) / FT(nv)
        ev .-= shift
        ev ./= norm(ev) # normalize

        diff = FT(1) - abs(sum(ev .* ev_old))
        if diff ≤ toler
            break
        else
            ev_old .= ev
        end
    end
    return ev
end

"""
    fiedler_vector(graph::Graph{<:Any,FT}, ::Direct)
This function computes the Fiedler vector using Julia/LAPACK subroutines.
This is intended for testing purposes only. Lanczos method is the 
recommended default.
"""
function fiedler_vector(graph::Graph{<:Any, FT}, ::Direct) where {FT}
    lmat = laplacian_matrix_full(graph)
    vl, vu = FT(1), FT(2) # dummies, not used
    il, iu = 2, 2 # second eigen vector (Fiedler vector)
    abstol = FT(1e-14)
    _, z =
        LinearAlgebra.LAPACK.syevr!('V', 'I', 'L', lmat, vl, vu, il, iu, abstol)
    return z[:]
end

"""
    laplacian_matrix_full(graph::Graph{<:Any,FT})
Computed the full Graph laplacian matrix explicitly. This is 
recommended for testing purposes only.
"""
function laplacian_matrix_full(graph::Graph{<:Any, FT}) where {FT}
    @unpack nverts, edge_data, edge_offset = graph
    lmat = zeros(nverts, nverts)
    for i in 1:nverts
        st, en = edge_offset[i:(i + 1)]
        for j in st:(en - 1)
            lmat[edge_data[j], i] = -1
        end
        lmat[i, i] = en - st
    end
    return lmat
end
"""
    split_graph(graph::Graph{I,FT}, n1)
This function partitions the graph such that the first partition
has `n1` vertices and second partition has `graph.nverts - n1`
vertices.
"""
function split_graph(graph::Graph{I, FT}, n1) where {I, FT}
    @unpack nverts, vertices, edge_offset, edge_data = graph
    if n1 == nverts
        return graph, Graph(I, FT)
    end

    lvertno = deepcopy(vertices)
    order = sortperm(fiedler_vector(graph, Lanczos()))

    n2 = nverts - n1
    sr1 = sr2 = FT(0)
    sort_ord1 = sort(order[1:n1])
    sort_ord2 = sort(order[(n1 + 1):nverts])
    vertices1 = vertices[sort_ord1]
    vertices2 = vertices[sort_ord2]

    lvertno[sort_ord1] .= 1:n1    # local vertex numbers
    lvertno[sort_ord2] .= -(1:n2) # for split graphs

    edge_offset1 = zeros(I, n1 + 1)
    edge_offset2 = zeros(I, n2 + 1)

    ed1 = [zeros(I, i) for i in zeros(I, n1)]
    ed2 = [zeros(I, i) for i in zeros(I, n2)]

    for i in 1:nverts
        v1 = lvertno[i]
        for j in edge_offset[i]:(edge_offset[i + 1] - 1)
            v2 = lvertno[edge_data[j]]
            if v1 > 0 && v2 > 0
                edge_offset1[v1 + 1] += 1
                push!(ed1[v1], v2)
            elseif v1 < 0 && v2 < 0
                edge_offset2[-v1 + 1] += 1
                push!(ed2[-v1], -v2)
            end
        end
    end
    edge_offset1[1], edge_offset2[1] = 1, 1

    [edge_offset1[i + 1] += edge_offset1[i] for i in 1:n1]
    [edge_offset2[i + 1] += edge_offset2[i] for i in 1:n2]

    edge_data1 = Array{I}(undef, edge_offset1[end] - 1)
    edge_data2 = Array{I}(undef, edge_offset2[end] - 1)

    for i in 1:n1
        edge_data1[edge_offset1[i]:(edge_offset1[i + 1] - 1)] .= ed1[i]
        sr1 = max(sr1, length(ed1[i]))
    end
    sr1 *= 2
    for i in 1:n2
        edge_data2[edge_offset2[i]:(edge_offset2[i + 1] - 1)] .= ed2[i]
        sr2 = max(sr2, length(ed2[i]))
    end
    sr2 *= 2

    return Graph(n1, edge_data1, edge_offset1, vertices1, sr1),
    Graph(n2, edge_data2, edge_offset2, vertices2, sr2)
end
