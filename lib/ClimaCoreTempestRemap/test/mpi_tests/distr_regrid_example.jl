# Simple concrete example of distributed regridding

import ClimaCore
import ClimaCoreTempestRemap as CCTR
using ClimaComms
using MPI
using ClimaCore:
    Geometry, Meshes, Domains, Topologies, Spaces, Fields, Operators
using IntervalSets

using ClimaCore.Spaces: Quadratures
using SparseArrays

FT = Float64

# copied from test/Operators/remapping.jl
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

comms_ctx = ClimaComms.MPICommsContext()
pid, nprocs = ClimaComms.init(comms_ctx)
comm = comms_ctx.mpicomm
rank = MPI.Comm_rank(comm)
root_pid = 0

comms_ctx_serial = ClimaComms.SingletonCommsContext()

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
source_space_serial = make_space(domain, source_nq, source_nex, source_ney, comms_ctx_serial)

# construct distributed target space
target_nq = 3
target_nex = 1
target_ney = 3
target_space = make_space(domain, target_nq, target_nex, target_ney, comms_ctx)
target_space_serial = make_space(domain, target_nq, target_nex, target_ney, comms_ctx_serial)

# generate weights (no remapping in x direction, so we really only need y_weights)
# TODO
if ClimaComms.iamroot(comms_ctx)
    # x_weights = Operators.x_overlap(target_space, source_space) # TODO doesn't work distributedly
    # y_weights = Operators.y_overlap(target_space, source_space)
    # weights = kron(x_weights, y_weights)
    weights = Operators.overlap(target_space, source_space)
    arr = [1, 2, 3]
else
    weights = nothing # TODO I think weights should be initialized as a SparseMatrixCSC but not sure how since we don't have the row/col lengths
    arr = nothing
end
ClimaComms.bcast(comms_ctx, weights)
ClimaComms.bcast(comms_ctx, arr)
ClimaComms.barrier(comms_ctx)
# MPI.Bcast!(weights, root_pid, comm)
# MPI.Barrier(comm)
@show arr
@show weights

# TODO reorder weights produced by kronecker product - or manually create weights from x_weights, y_weights

# map global redundant indices to pid (for both source and target)
#  - maybe like this https://github.com/CliMA/ClimaCore.jl/blob/main/lib/ClimaCoreTempestRemap/src/onlineremap.jl#L213
# source_global_inds = collect(Spaces.all_nodes(source_space))
# source_ind_to_pid = Dict{Int, Int}()

# # ind is of the form ((i, j), e) i.e. ((target/row, source/col), element), where i and j are nodal indices
# # source_linear_inds gives the column index into weights
# source_linear_inds = LinearIndices(source_global_inds)
# for (n, ind) in enumerate(source_global_inds)
#     elem = ind[2]
#     source_ind_to_pid[source_linear_inds[n]] = source_space.topology.elempid[elem]
# end

target_global_inds = collect(Spaces.all_nodes(target_space))
target_ind_to_pid = Dict{Int, Int}()

# target_linear_inds gives the row index into weights
target_global_linds = LinearIndices(target_global_inds)
for (n, ind) in enumerate(target_global_inds)
    elem = ind[2]
    target_ind_to_pid[target_global_linds[n]] =
        target_space.topology.elempid[elem]
end

# This code is for using Scatterv to distribute weights - using bcast for now
# if ClimaComms.iamroot(comms_ctx)
#     # set up send buffer for weights - gather weights data and length for each process
#     wt_data = zeros(FT, 1, length(weights.nzval))
#     #  - maybe wt_data needs to be structured differently (i.e. one vector for each process)
#     counts = zeros(Int, 1, nprocs)
#     for (n, wt) in enumerate(weights.nzval)
#         # get column (source) index of w in sparse array
#         source_ind = SparseArrays.findnz(weights)[2][n]
#         # get pid of source index of wt so we send to correct pid
#         pid = source_ind_to_pid[source_ind]

#         # add this weight to our data array and update the count for this pid
#         # TODO this only works if the weights for each process are contiguous in the sparse array
#         #  this might work for this example but we should fix it -
#         #  maybe we need to construct an array for each process then combine them all afterwards
#         wt_data[n] = wt
#         counts[pid] += 1

#         # TODO for sending weights, we need to send entire sparse matrix rep - just broadcast for now

#         # construct a MPI.VBuffer containing the weights data and length for each process
#         # this will automatically break up wt_data into correctly-sized chunks for each process
#         wt_send_buf = MPI.VBuffer(wt_data, vec(counts))
#     end
# else
#     wt_send_buf = nothing
# end

# # use MPI.scatterv to send the weights to each process
# # allocate space in the recv buffer for the incoming weights to this process (counts[rank+1])
# wt_recv_buf = Array{FT}(undef, counts[rank + 1])
# MPI.Scatterv!(wt_send_buf, wt_recv_buf, comm)
# # TODO merge local weights with received weights when using scatter

# generate source data on source space
source_data = Fields.ones(source_space)
source_array = dropdims(parent(source_data), dims = 3)
source_2d = reshape(
    source_array,
    (size(source_array)[1], size(source_array)[2] * size(source_array)[3]),
)
Nf = size(source_array, 3) # Nf is number of fields being remapped

# loop over rows in weight matrix, compute dot prod for each row
# TODO convert these to sparse representations later on
send_row_sums = FT.(zeros(weights.m))

@show target_ind_to_pid
@show weights.m

local_rows = findall(j -> target_ind_to_pid[j] == pid, collect(1:(weights.m)))
n_local_rows = length(local_rows)
local_row_sums = FT.(zeros(n_local_rows))
send_counts = Dict{typeof(pid), FT}()
# counts = FT.(zeros(nprocs))
for j in 1:(weights.m)
    target_pid = target_ind_to_pid[j]
    if target_pid != pid
        if haskey(send_counts, target_pid)
            send_counts[target_pid] += 1
        else
            send_counts[target_pid] = 1
        end
    end
    # loop over weights in this row
    for n in findall(x -> x == j, weights.rowval)
        # n lets us index into weights.nzval
        wt = weights.nzval[n]

        # find column (source) index of this weight
        source_ind = SparseArrays.findnz(weights)[2][n]
        source_val = vec(parent(source_data))[source_ind]

        # add to local array if target index is local to this process
        if target_pid == pid
            local_row_sums[j] += wt * source_val
        else
            send_row_sums[j] += wt * source_val
        end
    end
end

MPI.barrier(comms_ctx.mpicomm)
@show local_row_sums
@show send_row_sums
MPI.barrier(comms_ctx.mpicomm)

# set up send buffer for dot products - gather dot product data and length for each process
# counts[i] gives the number of row (target) inds process i is responsible for
#  (i.e. length of send buffer it will receive from each other process)
# TODO does this have to be on root?
# counts = [count(==(i), target_ind_to_pid) for i in unique(target_ind_to_pid)]
@assert send_counts[pid] == target_ney * target_nq^2 # TODO is this right


# set up graph_context for information exchange
#  - see graph_context code here https://github.com/CliMA/ClimaComms.jl/blob/main/src/mpi.jl#L120
send_array = send_row_sums
send_lengths = send_counts
send_pids = findall(p -> p != pid, collect(1:nprocs))
recv_array = FT.(zeros(weights.m))
recv_lengths = FT.(n_local_rows * ones(length(send_pids))) # from each process, we receive number of rows this process responsible for
recv_pids = send_pids

graph_context = ClimaComms.graph_context(
    comms_ctx,
    send_array,
    send_lengths,
    send_pids,
    recv_array,
    recv_lengths,
    recv_pids,
)

# from dss_transform: - maybe use topology fields instead?
# graph_context = ClimaComms.graph_context(
#     topology.context,
#     parent(send_data),
#     k .* topology.send_elem_lengths,
#     topology.neighbor_pids,
#     parent(recv_data),
#     k .* topology.recv_elem_lengths,
#     topology.neighbor_pids,
# )

# information exchange between processes
ClimaComms.start(graph_context)
ClimaComms.finish(graph_context)


# combine received data with local data
# loop over received rows, add received sums to local sum for each row
for j in local_rows
    # TODO how to index into recv_array correctly? - this assumes it's ordered
    # recv array will contain multiplied weight*source_data pairs, so I just have to add to my local sum
    # n * n_local_rows factor loops over received data from all other processes
    for n in collect(0:length(recv_pids))
        local_row_sums[j] += recv_array[j + n * n_local_rows]
    end
end


MPI.barrier(comms_ctx.mpicomm)
@show local_row_sums
MPI.barrier(comms_ctx.mpicomm)

# return local_row_sums
