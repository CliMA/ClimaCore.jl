
"""
    rsb_partition(topology::AbstractTopology, npart::I)

This function partitions the mesh into `npart` partitions using
Recursive Spectral Bisection.
"""
function rsb_partition(
    topology::AbstractTopology,
    npart::I,
    connect_type::Symbol,
) where {I}
    FT = eltype(Topologies.coordinate_type(topology))
    nelems = nlocalelems(topology)
    @assert npart < nelems
    @assert connect_type ∈ (:vertex, :face)

    partition = zeros(I, nelems)
    if npart == 1
        return partition
    end

    # 1D mesh is special
    # Assumes elememts are monotonically ordered
    # This can be removed if topology object supports 1D meshes
    if topology.mesh.domain isa Domains.IntervalDomain
        part_len = ones(I, nelems) .* div(nelems, npart)
        rem = nelems % npart
        if rem ≠ 0
            part_len[1:rem] .+= I(1)
        end
        ctr = 1
        for i in 1:npart
            for j in 1:part_len[i]
                partition[ctr] = i - 1
                ctr += 1
            end
        end
        return partition
    end
    # for 2D / 3D meshes
    graph =
        connect_type == :face ? build_face_graph(topology) :
        build_vertex_graph(topology)

    nlev = I(trunc(log(FT(npart)) / log(FT(2))))     # number of bisection levels
    nlev = 2^nlev < npart ? nlev + 1 : nlev
    part_len = ones(I, npart) * div(nelems, npart) # # of elements per partition
    rem = nelems % npart
    part_len[1:rem] .+= 1
    offset = [zeros(I, i) for i in zeros(I, nlev)]

    for i in 1:nlev
        offset[i] = zeros(I, 2^i + 1)
        if i == 1
            offset[i][1] = 1
            offset[i][2] = 1 + I(cld(npart, 2))
            offset[i][3] = 1 + npart
        else
            offset[i][1] = 1
            for j in 1:(2^(i - 1))
                nlocal = offset[i - 1][j + 1] - offset[i - 1][j]
                offset[i][1 + (j - 1) * 2 + 1] =
                    offset[i][1 + (j - 1) * 2] + I(cld(nlocal, 2))
                offset[i][1 + (j - 1) * 2 + 2] = offset[i - 1][j + 1]
            end
        end
    end

    # level 1
    n1 = sum(part_len[offset[1][1]:(offset[1][2] - 1)])
    garray_upper = split_graph(graph, n1)

    for i in 2:nlev # levels 2 through nlev
        garray = []
        for j in 1:(2^(i - 1))
            n1 = sum(
                part_len[offset[i][(j - 1) * 2 + 1]:(offset[i][(j - 1) * 2 + 2] - 1)],
            )
            push!(garray, split_graph(garray_upper[j], n1)...)
        end
        garray_upper = deepcopy(garray)
    end

    part = 0
    for grph in garray_upper
        if grph.nverts > 0
            partition[grph.vertices] .= part
            part += 1
        end
    end

    return partition
end
