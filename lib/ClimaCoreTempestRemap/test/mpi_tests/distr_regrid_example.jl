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

comms_ctx_serial = ClimaComms.SingletonCommsContext()
if isinteractive()
    comms_ctx = ClimaComms.SingletonCommsContext()
    pid, nprocs = ClimaComms.init(comms_ctx)
else
    comms_ctx = ClimaComms.MPICommsContext()
    pid, nprocs = ClimaComms.init(comms_ctx)
    comm = comms_ctx.mpicomm
    rank = MPI.Comm_rank(comm)
end
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
# TODO does this have to be on root?
source_space_serial =
    make_space(domain, source_nq, source_nex, source_ney, comms_ctx_serial)

# construct distributed target space
target_nq = 3
target_nex = 1
target_ney = 3
target_space = make_space(domain, target_nq, target_nex, target_ney, comms_ctx)
target_space_serial =
    make_space(domain, target_nq, target_nex, target_ney, comms_ctx_serial)

# generate source data on source space
source_data = Fields.ones(source_space)
source_data_serial = Fields.ones(source_space_serial)


# generate weights (no remapping in x direction, so we really only need y_weights)
if ClimaComms.iamroot(comms_ctx)
    weights = Operators.overlap(target_space, source_space)
    weights_serial = Operators.overlap(target_space_serial, source_space_serial)
else
    weights = nothing
    weights_serial = nothing
end
weights = ClimaComms.bcast(comms_ctx, weights)
weights_serial = ClimaComms.bcast(comms_ctx, weights_serial)
if !isinteractive()
    ClimaComms.barrier(comms_ctx)
end
# TODO reorder weights produced by kronecker product? - or manually create weights from x_weights, y_weights

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


# we need to use the serial space here to maintain information about which pid each elem is on
# if we use elempid on the distributed space, elem is in range [1...m_pid] for each pid,
#  this results in elempid of 1 for all elems since counting restarts
# we need target_ind_to_pid instead of elempid to get nodal gidx -> pid
if ClimaComms.iamroot(comms_ctx)
    target_global_inds = collect(Spaces.all_nodes(target_space_serial))
    target_ind_to_pid = Dict{Int, Int}()

    # target_global_inds gives the row index into weights
    # this is used to determine if a row is local or remote for each pid
    target_global_linds = LinearIndices(target_global_inds)
    for (n, ind) in enumerate(target_global_inds)
        elem = ind[2]
        target_ind_to_pid[target_global_linds[n]] =
            target_space.topology.elempid[elem]
    end
else
    target_ind_to_pid = nothing
end
target_ind_to_pid = ClimaComms.bcast(comms_ctx, target_ind_to_pid)
@show target_ind_to_pid
# # filter dictionary to get local map on each pid
# target_ind_to_pid = filter(kv -> kv.second == pid, target_ind_to_pid)
# @show target_ind_to_pid
# # subtract one less than min value to retrieve local index values
# local_min_row = minimum(collect(keys(target_ind_to_pid)))
# rows = collect(keys(target_ind_to_pid))
# pids = collect(values(target_ind_to_pid))
# rows .-= (local_min_row - 1)
# target_ind_to_pid = Dict(zip(rows, pids))
# @show target_ind_to_pid


# # create map from global to local source indices
# # START source index map
# source_global_inds = LinearIndices(collect(Spaces.all_nodes(source_space_serial)))
# source_local_inds = LinearIndices(collect(Spaces.all_nodes(source_space)))

# n_source_local_inds = length(source_local_inds)

# # TODO take global, local indices and construct map, then fill send_buf with this map

# # construct graph context to exchange information
# send_buf = [n_source_local_inds]
# send_pids = findall(p -> p != pid, collect(1:nprocs))
# send_lengths = Int.(ones(length(send_pids)))
# # this exchanges the number of local indices on each process
# # each pid then needs to use this to index from global to local
# recv_buf = FT.(zeros(nprocs))
# recv_lengths = Int.(ones(length(send_pids)))
# recv_pids = send_pids
# # TODO do I need to append my value to recv_buf? how do I know where?

# gc_source_inds = ClimaComms.graph_context(
#     comms_ctx,
#     send_buf,
#     send_lengths,
#     send_pids,
#     recv_buf,
#     recv_lengths,
#     recv_pids,
# )
# @show send_buf
# @show recv_buf
# # exchange information between processes (like `allgather`)
# ClimaComms.start(gc_source_inds)
# ClimaComms.finish(gc_source_inds)
# @show recv_buf

# pid_n_source_inds = recv_buf
# @show pid_n_source_inds

# # count all source indices in processes before this one
# prev_source_inds = 0
# for p in 1:pid - 1
#     global prev_source_inds += pid_n_source_inds[p]
# end

# # map global to local source indices
# source_global_to_local = Dict{typeof(pid), FT}()
# for i in 1:length(source_global_inds)
#     source_global_to_local[prev_source_inds + i] = source_local_inds[i]
# end
# @show source_global_to_local
# # END source index map


# loop over rows in weight matrix, compute dot prod for each row
# TODO convert these to sparse representations later on
send_row_sums = FT.(zeros(weights_serial.m))

# target_ind_to_pid maps nodal index to pid, where elempid maps element index to pid
local_rows = filter(j -> target_ind_to_pid[j] == pid, 1:(weights_serial.m))
# local_rows = collect(keys(target_ind_to_pid))
n_local_rows = length(local_rows)
local_row_sums = FT.(zeros(weights_serial.m))
send_counts = zeros(Int, nprocs)
# counts = FT.(zeros(nprocs))
@show local_rows

# loop over all rows (target indices) of weight matrix
for j in 1:(weights_serial.m)
    target_pid = target_ind_to_pid[j]
    # accumulate number of non-local rows (collecting to send)
    if target_pid != pid
        send_counts[target_pid] += 1
    end
    # loop over indices of weights in this row
    for n in findall(x -> x == j, weights_serial.rowval)
        # n lets us index into weights.nzval
        wt = weights_serial.nzval[n]

        # find local column (source) index of this weight
        # index 2 gives weights.colptr
        # ISSUE: we need to use weights_serial, but this gives the global source index, not local
        #  and we need to index into source_data using local index since it's generated on the distributed space

        # TODO use distributed source data
        #  we either need a global to local source index map, or a way to index into serial source data
        source_gidx = SparseArrays.findnz(weights_serial)[2][n]
        source_val = vec(parent(source_data_serial))[source_gidx]

        # add to local array if target index is local to this process
        if target_pid == pid
            local_row_sums[j] += wt * source_val
        else
            send_row_sums[j] += wt * source_val
        end
    end
end

@show local_row_sums
@show send_row_sums

# set up send buffer for dot products - gather dot product data and length for each process
# send_counts[i] gives the number of row (target) inds process i is responsible for
#  (i.e. length of send buffer it will receive from each other process)
# Note: send_counts will be empty for non-distributed runs

# set up graph_context for information exchange
#  - see graph_context code here https://github.com/CliMA/ClimaComms.jl/blob/main/src/mpi.jl#L120
send_array = send_row_sums
send_pids = filter(p -> send_counts[p] > 0, 1:nprocs)
send_lengths = send_counts[send_pids]
@assert length(send_lengths) == nprocs - 1


recv_array = FT.(zeros(weights_serial.m))
# from each process, we receive number of rows this process responsible for
recv_lengths = Int.(n_local_rows * ones(length(send_pids)))
recv_pids = send_pids

@show comms_ctx
@show send_array
@show send_lengths
@show send_pids
@show recv_array
@show recv_lengths
@show recv_pids

gc_row_sums = ClimaComms.graph_context(
    comms_ctx,
    send_array,
    send_lengths,
    send_pids,
    recv_array,
    recv_lengths,
    recv_pids,
)

# exchange information between processes (like `allgather`)
ClimaComms.start(gc_row_sums)
ClimaComms.finish(gc_row_sums)

# TODO one pid has recv_array all 0 after exchange
@show local_row_sums
@show recv_array

# # combine received data with local data
# # loop over received rows, add received sums to local sum for each row
# for j in local_rows
#     # TODO how to index into recv_array correctly? - this assumes it's ordered
#     # `recv_array` contains multiplied `weight*source_data` vals, so just add to local sum for dot product
#     # `n * n_local_rows` factor loops over received data from all other processes, skipping local rows
#     for n in collect(0:length(recv_pids))
#         @show j
#         @show n
#         @show j + n * n_local_rows
#         local_row_sums[j] += recv_array[j + n * n_local_rows]
#     end
# end

# loop over all rows (target indices) of weight matrix
for j in 1:(weights_serial.m)
    target_pid = target_ind_to_pid[j]

    # if row is non-local on target, add received value to row sums
    if !(target_pid == pid)
        local_row_sums[j] += recv_array[j]
    end
end


@show local_row_sums

# TODO the length of local_row_sums = number of redundant nodes we have
#  but still need to convert this vector to CC convention
remapped_field = Fields.ones(target_space)
remapped_field .= local_row_sums
