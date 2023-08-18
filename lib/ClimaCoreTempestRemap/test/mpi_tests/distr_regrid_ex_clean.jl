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
    nq::Int,
    nxelems::Int = 1,
    nyelems::Int = 1,
    comms_ctx = ClimaComms.SingletonCommsContext(),
)
    nq == 1 ? (quad = Quadratures.GL{1}()) : (quad = Quadratures.GLL{nq}())
    mesh = Meshes.RectilinearMesh(domain, nxelems, nyelems)
    topology = Topologies.Topology2D(comms_ctx, mesh)
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

# Given a distributed space, return a serial space using the same mesh and quadrature
function distr_to_serial_space(distr_space::Spaces.AbstractSpace)
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
    source_space::Spaces.AbstractSpace,
    target_space::Spaces.AbstractSpace,
)
    # construct serial spaces from distributed space info
    source_space_serial = distr_to_serial_space(source_space)
    target_space_serial = distr_to_serial_space(target_space)

    # generate weights on serial spaces
    weights = Operators.overlap(target_space_serial, source_space_serial)
    return weights
end

# Scatter data from root where each process receives multiple values
function scatterv_exchange(
    data::Union{Array{T}, Nothing},
    send_lengths::Array{Int},
    recv_length::Int,
    comms_ctx::ClimaComms.AbstractCommsContext,
    data_type::Type{T},
) where {T}
    if ClimaComms.iamroot(comms_ctx)
        # set up buffer to send `send_lengths` values to each pid
        sendbuf = MPI.VBuffer(data, send_lengths)
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

# Exchange data from all processes to all processes
function all_to_allv_exchange(
    data::Array{T},
    send_lengths::Array{Int},
    recv_lengths::Array{Int},
    comms_ctx::ClimaComms.AbstractCommsContext,
) where {T}
    # set up buffer to send `send_lengths` values to each pid
    @show data
    @show send_lengths
    @show recv_lengths
    sendbuf = MPI.VBuffer(data, send_lengths)

    # set up receive buffer of specified length on each process
    recvbuf = MPI.VBuffer(data, recv_lengths)

    # perform information exchange
    MPI.Alltoallv!(sendbuf, recvbuf, comms_ctx.mpicomm)
    return recvbuf.data
end

# Calculate the total number of nodes on each pid for this space
function node_counts_by_pid(
    space::Spaces.SpectralElementSpace2D;
    is_cumul::Bool,
)
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
    if is_cumul
        # 1st value is 1 to match sparse matrix `colptr` setup
        node_counts[1] = 1
        # return cumulative sum of node counts
        return cumsum(node_counts)
    else
        return node_counts[2:end]
    end
end

# Calculate the number of nonzero weights on each process
function n_weights_by_pid(
    node_counts_src::Array{Int},
    colptrs::Array{Int},
    nprocs::Int,
)
    n_weights = zeros(Int, nprocs)

    for p in 1:nprocs
        n_weights[p] =
            colptrs[node_counts_src[p + 1]] - colptrs[node_counts_src[p]]
    end

    return n_weights
end

# Return range of colptrs for weights on this process
#  This contains all column pointers for this process, and
#  one additional bound, as in the CSC sparse matrix representation
function colptrs_my_pid(
    node_counts_src::Array{Int},
    colptrs::Array{Int},
    nprocs::Int,
)
    colptrs_pid = zeros(Int, nprocs)
    # `node_counts_src` contains cumulative sum, so can be used for range
    colptrs_pid = colptrs[node_counts_src[pid]:node_counts_src[pid + 1]]
    return colptrs_pid
end

# Take weight matrix mapping between serial spaces, distribute among processes
function distr_weights(
    weights::Union{SparseMatrixCSC, Nothing},
    source_space::Spaces.AbstractSpace,
    target_space::Spaces.AbstractSpace,
    comms_ctx::ClimaComms.AbstractCommsContext,
    nprocs::Int,
) where {T}
    # calculate number of source space nodes on each pid
    node_counts_src = node_counts_by_pid(source_space, is_cumul = true)

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
    colptrs_pid = colptrs_my_pid(node_counts_src, colptrs, nprocs)

    # get number of nonzero weights on each process - use for send and receive buffer lengths
    n_weights = n_weights_by_pid(node_counts_src, colptrs, nprocs)

    # scatter weights and row indices
    send_lengths = n_weights
    recv_length = n_weights[pid]
    weight_vals =
        scatterv_exchange(nzvals, send_lengths, recv_length, comms_ctx, FT)
    row_inds =
        scatterv_exchange(rowvals, send_lengths, recv_length, comms_ctx, Int)

    return weight_vals, row_inds, colptrs_pid
end

# Given nonzero values, their row indices, and column pointers, construct a sparsematrix
function to_sparse(
    nzval::Array{T},
    rowval::Array{Int},
    colptr::Array{Int},
) where {T}
    # convert colptr to column indices
    len = length(rowval)
    colval = zeros(Int, len)
    col = 1
    for i in 1:len
        if i == colptr[col + 1]
            col += 1
        end
        colval[i] = col
    end

    return sparse(rowval, colval, nzval)
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

# @show weights
# @show row_inds
# @show col_offsets



# TODO STEP 3: reconstruct weight matrix on each process (SparseMatrixCSC)
# TODO should we just return column inds from distr_weights so we don't have to reconstruct here?
weights = to_sparse(weights, row_inds, col_offsets)

# Note now we have weight matrix divided by columns into sub matrices
#  now we have to to separate weight matrix by rows for receive side
#  any row with nonzero value needs to be sent

# v1: assume each chunk of rows all has to be sent (no rows all zero - dense)
# need target_ind_to_pid (map of row ind to pid)
# need number of rows to receive on each process (should be in unclean example)
# need number of rows to send to each process (find by exchanging receive lengths?)
# with dense assumption, array containing number of values to send to each process is same on each process

# TODO STEP 4: construct send/recv buffers for source data
# Generate source data on source space
source_data = Fields.ones(source_space)

# Calculate the number of nodes on the target space for each pid
# this gives the total number of nodes each process needs, but we need to subtract number of nodes it already has (?)
node_counts_tgt = node_counts_by_pid(target_space, is_cumul = false)

# Receive buffer length is the number of target space nodes for this pid
# TODO this will expect total number of nodes needed on pid from every other pid
#  seems like we should divide by nprocs, but we need to consider how many elements on each pid (may not divide evenly)
# src_data_recv_lengths = [(node_counts_tgt[pid] - node_counts_src[pid]) for i in 1:nprocs]
src_data_recv_lengths = [node_counts_tgt[pid] for i in 1:nprocs]
# Send buffer lengths is number of target space nodes for each pid
# src_data_send_lengths = [(node_counts_tgt[i] - node_counts_src[i]) for i in 1:nprocs]
src_data_send_lengths = [node_counts_tgt[i] for i in 1:nprocs]

# TODO is it okay to use parent here? what will happen to ordering?
source_data_exchanged = all_to_allv_exchange(
    parent(source_data),
    src_data_send_lengths,
    src_data_recv_lengths,
    comms_ctx,
)
