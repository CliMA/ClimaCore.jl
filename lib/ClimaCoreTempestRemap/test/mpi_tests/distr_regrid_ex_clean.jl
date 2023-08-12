# Simple concrete example of distributed regridding - modular approach

import ClimaCore
import ClimaCoreTempestRemap as CCTR
using ClimaComms
using ClimaCore:
    Geometry, Meshes, Domains, Topologies, Spaces, Fields, Operators
using ClimaCore.Spaces: Quadratures

using IntervalSets
using MPI
using SparseArrays

FT = Float64

# Construct a space using the input information
function make_space(
    domain::Domains.RectangleDomain,
    nq,
    nxelems = 1,
    nyelems = 1,
    comms_ctx = ClimaComms.SingletonCommsContext(),
)
    nq == 1 ? (quad = Quadratures.GL{1}()) : (quad = Quadratures.GLL{nq}())
    mesh = Meshes.RectilinearMesh(domain, nxelems, nyelems)
    topology = Topologies.Topology2D(comms_ctx, mesh)
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

# Given a distributed space, return a serial space using the same mesh and quadrature
function distr_to_serial_space(distr_space)
    # set up serial comms context
    comms_ctx_serial = ClimaComms.SingletonCommsContext()

    # extract info from distributed space
    mesh = distr_space.topology.mesh
    quad = distr_space.quadrature_style

    # construct serial objects
    topology = Topologies.Topology2D(comms_ctx_serial, mesh)
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

# Given two distr spaces, generate a weight matrix mapping between
#  the two associated serial spaces
# Note: this function should only be called by the root process
function gen_weights(
    source_space,
    target_space,
)
    # construct serial spaces from distributed space info
    source_space_serial = distr_to_serial_space(source_space)
    target_space_serial = distr_to_serial_space(target_space)

    # generate weights on serial spaces
    weights = Operators.overlap(target_space_serial, source_space_serial)
end

# TODO combine this with col_offsets function?
# Calculate the number of nonzero values (weights) on each pid
function calc_counts(weights, source_space, nprocs)
    # create map from source linear ind to pid
    s_inds = collect(Spaces.all_nodes(source_space))
    s_lind_to_pid = zeros(Int, weights.n)

    # extract element number for each linear index, use this to get pid
    for (ind_lin, ind_ije) in enumerate(s_inds)
        elem = ind_ije[2]
        pid = source_space.topology.elempid[elem]
        s_lind_to_pid[ind_lin] = pid
    end

    # get number of nonzero weights on each process
    nzval_counts = zeros(Int, nprocs)
    for ind_lin in LinearIndices(s_inds)
        pid = s_lind_to_pid[ind_lin]
        nzval_counts[pid] += weights.colptr[ind_lin + 1] - weights.colptr[ind_lin]
    end

    return nzval_counts
end

# Calculate the number of columns containing weights on each process
function calc_col_offsets(weights, source_space, nprocs)
    s_inds = collect(Spaces.all_nodes(source_space))
    # create map from source column ind to pid
    s_col_to_pid = zeros(Int, weights.n)

    # extract element number for each linear index, use this to get pid
    for (ind_lin, ind_ije) in enumerate(s_inds)
        col = ind_ije[1][2]

        # if col is unseen, continue
        if s_col_to_pid[col] == 0
            elem = ind_ije[2]
            pid = source_space.topology.elempid[elem]

            s_col_to_pid[col] = pid
        end
    end

    # get number of columns on each process
    num_cols = zeros(Int, nprocs)
    for p in 1:nprocs
        num_cols[p] = count(c -> c == p, s_col_to_pid)
    end

    # go through columns and accumulate offset for each pid
    col_offsets = zeros(Int, nprocs)
    offset = Int(1)
    for p in 1:nprocs
        col_offsets[p] = offset
        offset += num_cols[p]
    end

    return col_offsets
end

# Scatter data where each process receives one value
#  Use for counts and column offest
function scatter_length1(data, comms_ctx, inplace::Bool)
    # if ClimaComms.iamroot(comms_ctx)
    #     # set up buffer to send 1 value to each pid
    #     sendbuf = MPI.UBuffer(data, Int(1))
    #     # if inplace, don't modify root recvbuf
    #     #  when exchanging counts, want root to retain info about all processes
    #     recvbuf = inplace ? MPI.IN_PLACE : MPI.Buffer(zeros(Int, 1))
    # else
    #     # send nothing on non-root processes
    #     sendbuf = nothing
    #     # create receive buffer of length 1 on each process
    #     recvbuf = MPI.Buffer(zeros(Int, 1))
    # end

    # # scatter data to all processes
    # MPI.Scatter!(sendbuf, recvbuf, comms_ctx.mpicomm)
    if ClimaComms.iamroot(comms_ctx)
        # set up buffer to send 1 value to each pid
        sendbuf = MPI.UBuffer(data, Int(1))
        MPI.Scatter!(sendbuf, MPI.IN_PLACE, comms_ctx.mpicomm)
        return data
    else
        recvbuf = MPI.Buffer(zeros(Int, 1))
        MPI.Scatter!(nothing, recvbuf, comms_ctx.mpicomm)
        return recvbuf.data
    end

    # return recvbuf.data
end

# Scatter data where each process receives multiple values
function scatterv_exchange(data, comms_ctx, send_counts, recv_length)
    if ClimaComms.iamroot(comms_ctx)
        # set up buffer to send `length` values to each pid
        sendbuf = MPI.VBuffer(data, send_counts)
    else
        # send nothing on non-root processes
        sendbuf = nothing
    end
    # create receive buffer of specified length on each process
    recvbuf = MPI.Buffer(zeros(Int, recv_length))

    # scatter data to all processes
    MPI.Scatterv!(sendbuf, recvbuf, comms_ctx.mpicomm)
    return recvbuf.data
end

# Take weight matrix mapping between serial spaces, distribute among processes
function distr_weights(weights, source_space, target_space, comms_ctx, nprocs)
    comm = comms_ctx.mpicomm
    # calculate number of nonzero vals (weights) on each pid to fill counts
    if ClimaComms.iamroot(comms_ctx)
        counts = calc_counts(weights, source_space, nprocs)
        col_offsets = calc_col_offsets(weights, source_space, nprocs)
        nzvals = weights.nzval
        row_vals = weights.rowval
    else
        counts = nothing
        col_offsets = nothing
        nzvals = nothing
        row_vals = nothing
    end
    # scatter to each process (counts will be used to create receive buffer lengths)
    #  do this in place so root retains info about all pids' counts
    n_weights_mypid = scatter_length1(counts, comms_ctx, true)[1]
    col_offset = scatter_length1(col_offsets, comms_ctx, false)[1]

    # scatter weights
    weights = scatterv_exchange(nzvals, comms_ctx, counts, n_weights_mypid)

    # scatter row indices
    row_inds = scatterv_exchange(row_vals, comms_ctx, counts, n_weights_mypid)1

    @show weights
    @show row_inds
    @show col_offsets
    return weights, row_inds, col_offsets
end


# set up MPI info
comms_ctx = ClimaComms.MPICommsContext()
pid, nprocs = ClimaComms.init(comms_ctx)
comm = comms_ctx.mpicomm
rank = MPI.Comm_rank(comm)
root_pid = 0

# construct domain
domain = Domains.RectangleDomain(
    Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
    Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
    x1boundary = (:bottom, :top),
    x2boundary = (:left, :right),
)

# construct distributed source space
source_nq = 3
source_nex = 1
source_ney = 2
source_space = make_space(domain, source_nq, source_nex, source_ney, comms_ctx)

# construct distributed target space
target_nq = 3
target_nex = 1
target_ney = 3
target_space = make_space(domain, target_nq, target_nex, target_ney, comms_ctx)

# generate weights matrix on root process
if ClimaComms.iamroot(comms_ctx)
    weights = gen_weights(source_space, target_space)
else
    weights = nothing
end

# distribute (scatter) weights to all processes
weights, row_inds, col_offsets = distr_weights(weights, source_space, target_space, comms_ctx, nprocs)

@show weights
@show row_inds
@show col_offsets
