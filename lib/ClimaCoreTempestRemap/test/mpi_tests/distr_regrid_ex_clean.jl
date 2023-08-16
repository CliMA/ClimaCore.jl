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
function gen_weights(source_space, target_space)
    # construct serial spaces from distributed space info
    source_space_serial = distr_to_serial_space(source_space)
    target_space_serial = distr_to_serial_space(target_space)

    # generate weights on serial spaces
    weights = Operators.overlap(target_space_serial, source_space_serial)
    return weights
end

# Calculate the total number of nodes on each pid for this space
function node_counts_by_pid(space::Spaces.SpectralElementSpace2D)
    nprocs = ClimaComms.nprocs(space.topology.context)
    # count how many elements each process is responsible for
    elem_counts = zeros(Int, nprocs + 1)
    elempid = space.topology.elempid
    # counts for all pids are offset by 1 to match sparse matrix `colptr` setup
    for pid in elempid
        elem_counts[pid + 1] += 1
    end

    # calculate number of nodes per element
    nq = Spaces.Quadratures.degrees_of_freedom(space.quadrature_style)

    # number of nodes = number of elements * nq ^ 2
    node_counts = elem_counts .* (nq^2)
    # 1st value is 1 to match sparse matrix `colptr` setup
    node_counts[1] = 1
    # return cumulative sum of node counts
    return cumsum(node_counts)
end

# Scatter data where each process receives multiple values
function scatterv_exchange(data, send_counts, recv_length, comms_ctx, data_type)
    if ClimaComms.iamroot(comms_ctx)
        # set up buffer to send `length` values to each pid
        sendbuf = MPI.VBuffer(data, send_counts)
    else
        # send nothing on non-root processes
        sendbuf = nothing
    end
    # create receive buffer of specified length on each process
    recvbuf = MPI.Buffer(zeros(data_type, recv_length))

    # scatter data to all processes
    MPI.Scatterv!(sendbuf, recvbuf, comms_ctx.mpicomm)
    return recvbuf.data
end

# Calculate the number of nonzero weights on each process
function n_weights_by_pid(node_counts, colptrs, nprocs)
    n_weights = zeros(Int, nprocs)

    for p in 1:nprocs
        n_weights[p] = colptrs[node_counts[p + 1]] - colptrs[node_counts[p]]
    end

    return n_weights
end

# Return range of colptrs for weights on this process
#  This contains all column pointers for this process, and
#  one additional bound, as in the CSC sparse matrix representation
function colptrs_my_pid(node_counts, colptrs, nprocs)
    colptrs_pid = zeros(Int, nprocs)
    # `node_counts` contains cumulative sum, so can be used for range
    for p in 1:nprocs
        colptrs_pid = colptrs[node_counts[pid]:node_counts[pid + 1]]
    end
    return colptrs_pid
end

# Take weight matrix mapping between serial spaces, distribute among processes
function distr_weights(weights, source_space, target_space, comms_ctx, nprocs)
    # calculate number of source space nodes on each pid
    node_counts = node_counts_by_pid(source_space)

    if ClimaComms.iamroot(comms_ctx)
        # extract weight matrix fields
        colptrs = weights.colptr
        nzvals = weights.nzval
        rowvals = weights.rowval
    else
        colptrs = nothing
        nzvals = nothing
        rowvals = nothing
    end

    # broadcast weight column pointers to all processes
    colptrs = MPI.bcast(colptrs, comms_ctx.mpicomm)

    # extract only the column pointers needed on this process
    colptrs_pid = colptrs_my_pid(node_counts, colptrs, nprocs)

    # get number of nonzero weights on each process - use for send and receive buffer lengths
    n_weights = n_weights_by_pid(node_counts, colptrs, nprocs)

    # scatter weights and row indices
    send_counts = n_weights
    recv_length = n_weights[pid]
    weight_vals =
        scatterv_exchange(nzvals, send_counts, recv_length, comms_ctx, FT)
    row_inds =
        scatterv_exchange(rowvals, send_counts, recv_length, comms_ctx, Int)

    return weight_vals, row_inds, colptrs_pid
end

# Given nonzero values, their row indices, and column pointers, construct a sparsematrix
function to_sparse(nzval, rowval, colptr)
    # convert colptr to column indices
    len = colptr[end] - 1
    colval = zeros(Int, len)
    col = 1
    for i in 1:len
        if i == colptr[col + 1]
            col += 1
        end
        colval[i] = col
    end

    return spase(rowval, colval, nzval)
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

# STEP 1: generate weights matrix on root process
if ClimaComms.iamroot(comms_ctx)
    weights = gen_weights(source_space, target_space)
else
    weights = nothing
end

# STEP 2: distribute (scatter) weights to all processes
weights, row_inds, col_offsets =
    distr_weights(weights, source_space, target_space, comms_ctx, nprocs)

@show weights
@show row_inds
@show col_offsets



# TODO STEP 3: reconstruct weight matrix on each process (SparseMatrixCSC)
# TODO should we just return column inds from distr_weights so we don't have to reconstruct here?
weights = to_sparse(weights, row_inds, col_offsets)

# Note now we have weight matrix divided by columns into sub matrices
#  now we have to to separate weight matrix by rows for receive side
#  any row with nonzero value needs to be sent

# v1: assume each chunk of rows all has to be sent (no rows all zero - dense)
# need target_ind_to_pid (map of row ind to pid)
# need number of rows to receive on each process (should be in unclean example)
# need number of rows to send to each process
# with dense assumption, array containing number of values to send to each process is same on each process

# TODO STEP 4: construct send/recv buffers for source data
