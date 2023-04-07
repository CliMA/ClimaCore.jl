using ClimaCore.DataLayouts
using ClimaComms
using ClimaCommsMPI
using MPI


"""
    LinearMap{S, T, W, I, V, M}

stores information on the TempestRemap map and the source and target data:

where:
 - `source_space` and `target_space` are ClimaCore's 2D spaces.
 - `weights` is a vector of remapping weights. (length = number of overlap-mesh nodes)
 - `source_idxs` a 3-element Tuple with 3 index vectors, representing (i,j,elem) indices on the source mesh. (length of each index vector = number of overlap-mesh nodes)
 - `target_idxs` is the same as `source_idxs` but for the target mesh.
 - `col_indices` are the source column indices from TempestRemap. (length = number of overlap-mesh nodes)
 - `row_indices` are the target row indices from TempestRemap. (length = number of overlap-mesh nodes)
 - `out_type` string that defines the output type
 - `source_global_elem_lidx` is a mapping from global to local indices on the source space
 - `target_global_elem_lidx` is a mapping from global to local indices on the target space

"""
struct LinearMap{S, T, W, I, V, M} # make consistent with / move to regridding.jl
    source_space::S
    target_space::T
    weights::W
    source_idxs::I
    target_idxs::I
    col_indices::V
    row_indices::V
    out_type::String
    source_global_elem_lidx::M
    target_global_elem_lidx::M
end

function remap!(
    target::IJFH{S, Nqt},
    R::LinearMap,
    source::IJFH{S, Nqs},
) where {S, Nqt, Nqs}
    source_array = parent(source)
    target_array = parent(target)

    fill!(target_array, zero(eltype(target_array)))
    Nf = size(target_array, 3)

    # ideally we would use the tempestremap dgll (redundant node) representation
    # unfortunately, this doesn't appear to work quite as well (for out_type = dgll) as the cgll
    for (n, wt) in enumerate(R.weights)
        is, js, es = (
            view(R.source_idxs[1], n),
            view(R.source_idxs[2], n),
            view(R.source_idxs[3], n),
        )
        it, jt, et = (
            view(R.target_idxs[1], n),
            view(R.target_idxs[2], n),
            view(R.target_idxs[3], n),
        )
        for f in 1:Nf
            target_array[it, jt, f, et] += wt * source_array[is, js, f, es]
        end
    end

    # use unweighted dss to broadcast the so-far unpopulated (redundant) nodes from their unique node counterparts
    # (we could get rid of this if we added the redundant nodes to the matrix)
    # using out_type == "cgll"
    if R.out_type == "cgll"
        topology = Spaces.topology(R.target_space)
        hspace = Spaces.horizontal_space(R.target_space)
        quadrature_style = hspace.quadrature_style
        Spaces.dss2!(target, topology, quadrature_style)
    end
    return target
end

function remap!(target::Fields.Field, R::LinearMap, source::Fields.Field)
    # Serial remapping case
    if Spaces.topology(axes(target)).context isa
       ClimaComms.SingletonCommsContext &&
       Spaces.topology(axes(source)).context isa
       ClimaComms.SingletonCommsContext
        @assert axes(source) == R.source_space
        @assert axes(target) == R.target_space
        # we use the tempestremap cgll representation
        # it will set the redundant nodes to zero
        remap!(Fields.field_values(target), R, Fields.field_values(source))
        return target
        # Mixed serial/distributed case - error
    elseif Spaces.topology(axes(target)).context isa
           ClimaComms.SingletonCommsContext ||
           Spaces.topology(axes(source)).context isa
           ClimaComms.SingletonCommsContext
        error(
            "Remapping is only possible between two serial spaces or two distributed spaces.",
        )
        # Distributed remapping case
    else
        @assert !(
            Spaces.topology(axes(source)).context isa
            ClimaComms.SingletonCommsContext
        )
        @assert !(
            Spaces.topology(axes(target)).context isa
            ClimaComms.SingletonCommsContext
        )

        target_array = parent(target)
        source_array = parent(source)

        fill!(target_array, zero(eltype(target_array)))
        Nf = size(target_array, 3)

        # ideally we would use the tempestremap dgll (redundant node) representation
        # unfortunately, this doesn't appear to work quite as well (for out_type = dgll) as the cgll
        for (n, wt) in enumerate(R.weights)
            # choose all global source indices
            #  (for simple distr. remapping with broadcasted source data, no halo exchange)
            # TODO check: when sending all source data to all processes, this should give same result as below selection
            # is, js, es = map(
            #     x -> x[1],
            #     (
            #         view(R.source_idxs[1], n),
            #         view(R.source_idxs[2], n),
            #         view(R.source_idxs[3], n),
            #     ),
            # )

            # choose only the source inds local to this process (skip inds from other processes)
            source_elem_gidx = view(R.source_idxs[3], n)[1]
            if !(source_elem_gidx in keys(R.source_global_elem_lidx))
                continue
            else
                # get global i, j inds but local elem index e
                is = view(R.source_idxs[1], n)[1]
                js = view(R.source_idxs[2], n)[1]
                es = R.source_global_elem_lidx[source_elem_gidx]

                # choose only the target inds local to this process (skip inds from other processes)
                target_elem_gidx = view(R.target_idxs[3], n)[1]
                if !(target_elem_gidx in keys(R.target_global_elem_lidx))
                    continue
                else
                    # get global i, j inds but local elem index e
                    it = view(R.target_idxs[1], n)[1]
                    jt = view(R.target_idxs[2], n)[1]
                    et = R.target_global_elem_lidx[target_elem_gidx]

                    for f in 1:Nf
                        target_array[it, jt, f, et] +=
                            wt * source_array[is, js, f, es]
                    end
                end
            end

        end
        # use unweighted dss to broadcast the so-far unpopulated (redundant) nodes from their unique node counterparts
        # (we could get rid of this if we added the redundant nodes to the matrix)
        # using out_type == "cgll"
        if R.out_type == "cgll"
            topology = Spaces.topology(axes(target))
            hspace = Spaces.horizontal_space(axes(target))
            quadrature_style = hspace.quadrature_style
            Spaces.dss2!(
                Fields.field_values(target),
                topology,
                quadrature_style,
            )
        end
        return target
    end
end

"""
    remap(R::LinearMap, source::Field)

Applies the remapping `R` to a `source` Field, allocating a new field in the output.
"""
function remap(R::LinearMap, source::Fields.Field)
    remap!(Fields.Field(eltype(source), R.target_space), R, source)
end


"""
    generate_map(comms_ctx, target_space, source_space; in_type="cgll", out_type="cgll")

Generate the remapping weights from TempestRemap, returning a `LinearMap` object. This should only be called once.
"""
# TODO change order of target, source args
function generate_map(
    comms_ctx::ClimaCommsMPI.MPICommsContext,
    target_space::Spaces.SpectralElementSpace2D,
    source_space::Spaces.SpectralElementSpace2D;
    target_space_distr = nothing,
    source_space_distr = nothing,
    meshfile_source = tempname(),
    meshfile_target = tempname(),
    meshfile_overlap = tempname(),
    weightfile = tempname(),
    in_type = "cgll",
    out_type = "cgll",
)
    # TODO change all target_space, source_space uses to _distr so we can remove serial spaces
    @assert target_space_distr.topology.context == comms_ctx
    @assert source_space_distr.topology.context == comms_ctx

    if ClimaComms.iamroot(comms_ctx)
        # write meshes and generate weights on root process (using global indices)
        write_exodus(meshfile_source, source_space_distr.topology)
        write_exodus(meshfile_target, target_space_distr.topology)
        overlap_mesh(meshfile_overlap, meshfile_source, meshfile_target)
        remap_weights(
            weightfile,
            meshfile_source,
            meshfile_target,
            meshfile_overlap;
            in_type = in_type,
            in_np = Spaces.Quadratures.degrees_of_freedom(
                source_space_distr.quadrature_style,
            ),
            out_type = out_type,
            out_np = Spaces.Quadratures.degrees_of_freedom(
                target_space_distr.quadrature_style,
            ),
        )

        # read weight data
        weights, col_indices, row_indices = NCDataset(weightfile, "r") do ds_wt
            (Array(ds_wt["S"]), Array(ds_wt["col"]), Array(ds_wt["row"]))
        end
        # TempestRemap exports in CSR format (i.e. row_indices is sorted)

        # TODO: add in extra rows to avoid DSS step
        #  - for each unique node, we would add extra rows for all the duplicates, with the same column
        #  - ideally keep in CSR format
        #  e.g. iterate by row (i.e. target node):
        #   - if new node, then append it
        #   - if not new, copy previous entries
        #  - requires a mechanism to query whether find the first common node of a given node


        # we need to be able to look up the indices of unique nodes
        # TODO extend unique_nodes for distributed spaces
        source_unique_idxs =
            in_type == "cgll" ? collect(Spaces.unique_nodes(source_space)) :
            collect(Spaces.all_nodes(source_space_distr))
        target_unique_idxs =
            out_type == "cgll" ? collect(Spaces.unique_nodes(target_space)) :
            collect(Spaces.all_nodes(target_space_distr))

        # re-order to avoid unnecessary allocations
        source_unique_idxs_i =
            map(col -> source_unique_idxs[col][1][1], col_indices)
        source_unique_idxs_j =
            map(col -> source_unique_idxs[col][1][2], col_indices)
        source_unique_idxs_e =
            map(col -> source_unique_idxs[col][2], col_indices)
        target_unique_idxs_i =
            map(row -> target_unique_idxs[row][1][1], row_indices)
        target_unique_idxs_j =
            map(row -> target_unique_idxs[row][1][2], row_indices)
        target_unique_idxs_e =
            map(row -> target_unique_idxs[row][2], row_indices)

        source_unique_idxs =
            (source_unique_idxs_i, source_unique_idxs_j, source_unique_idxs_e)
        target_unique_idxs =
            (target_unique_idxs_i, target_unique_idxs_j, target_unique_idxs_e)
    else
        weights = nothing
        source_unique_idxs = nothing
        target_unique_idxs = nothing
        col_indices = nothing
        row_indices = nothing
    end

    if !(comms_ctx isa ClimaComms.SingletonCommsContext)
        root_pid = 0
        weights = MPI.bcast(weights, root_pid, comms_ctx.mpicomm)
        source_unique_idxs =
            MPI.bcast(source_unique_idxs, root_pid, comms_ctx.mpicomm)
        target_unique_idxs =
            MPI.bcast(target_unique_idxs, root_pid, comms_ctx.mpicomm)
        col_indices = MPI.bcast(col_indices, root_pid, comms_ctx.mpicomm)
        row_indices = MPI.bcast(row_indices, root_pid, comms_ctx.mpicomm)
    end
    ClimaComms.barrier(comms_ctx)

    # Create mappings from global to local element indices (for distributed remapping)
    if (source_space_distr != nothing)
        source_local_elem_gidx = source_space_distr.topology.local_elem_gidx # gidx = local_elem_gidx[lidx]
        source_global_elem_lidx = Dict{Int, Int}() # inverse of local_elem_gidx: lidx = global_elem_lidx[gidx]
        for (lidx, gidx) in enumerate(source_local_elem_gidx)
            source_global_elem_lidx[gidx] = lidx
        end
    else
        source_global_elem_lidx = nothing
    end

    if (target_space_distr != nothing)
        target_local_elem_gidx = target_space_distr.topology.local_elem_gidx # gidx = local_elem_gidx[lidx]
        target_global_elem_lidx = Dict{Int, Int}() # inverse of local_elem_gidx: lidx = global_elem_lidx[gidx]
        for (lidx, gidx) in enumerate(target_local_elem_gidx)
            target_global_elem_lidx[gidx] = lidx
        end
    else
        target_global_elem_lidx = nothing
    end

    return LinearMap(
        source_space_distr,
        target_space_distr,
        weights,
        source_unique_idxs,
        target_unique_idxs,
        col_indices,
        row_indices,
        out_type,
        source_global_elem_lidx,
        target_global_elem_lidx,
    )
end

"""
    generate_map(comms_ctx, target_space, source_space; in_type="cgll", out_type="cgll")

Generate the remapping weights from TempestRemap, returning a `LinearMap` object. This should only be called once.
"""
function generate_map(
    comms_ctx::ClimaComms.SingletonCommsContext,
    target_space::Spaces.SpectralElementSpace2D,
    source_space::Spaces.SpectralElementSpace2D;
    meshfile_source = tempname(),
    meshfile_target = tempname(),
    meshfile_overlap = tempname(),
    weightfile = tempname(),
    in_type = "cgll",
    out_type = "cgll",
)
    # write meshes and generate weights on root process (using global indices)
    write_exodus(meshfile_source, source_space.topology)
    write_exodus(meshfile_target, target_space.topology)
    overlap_mesh(meshfile_overlap, meshfile_source, meshfile_target)
    remap_weights(
        weightfile,
        meshfile_source,
        meshfile_target,
        meshfile_overlap;
        in_type = in_type,
        in_np = Spaces.Quadratures.degrees_of_freedom(
            source_space.quadrature_style,
        ),
        out_type = out_type,
        out_np = Spaces.Quadratures.degrees_of_freedom(
            target_space.quadrature_style,
        ),
    )

    # read weight data
    weights, col_indices, row_indices = NCDataset(weightfile, "r") do ds_wt
        (Array(ds_wt["S"]), Array(ds_wt["col"]), Array(ds_wt["row"]))
    end
    # TempestRemap exports in CSR format (i.e. row_indices is sorted)

    # TODO: add in extra rows to avoid DSS step
    #  - for each unique node, we would add extra rows for all the duplicates, with the same column
    #  - ideally keep in CSR format
    #  e.g. iterate by row (i.e. target node):
    #   - if new node, then append it
    #   - if not new, copy previous entries
    #  - requires a mechanism to query whether find the first common node of a given node

    # we need to be able to look up the indices of unique nodes
    source_unique_idxs =
        in_type == "cgll" ? collect(Spaces.unique_nodes(source_space)) :
        collect(Spaces.all_nodes(source_space))
    target_unique_idxs =
        out_type == "cgll" ? collect(Spaces.unique_nodes(target_space)) :
        collect(Spaces.all_nodes(target_space))

    # re-order to avoid unnecessary allocations
    source_unique_idxs_i =
        map(col -> source_unique_idxs[col][1][1], col_indices)
    source_unique_idxs_j =
        map(col -> source_unique_idxs[col][1][2], col_indices)
    source_unique_idxs_e = map(col -> source_unique_idxs[col][2], col_indices)
    target_unique_idxs_i =
        map(row -> target_unique_idxs[row][1][1], row_indices)
    target_unique_idxs_j =
        map(row -> target_unique_idxs[row][1][2], row_indices)
    target_unique_idxs_e = map(row -> target_unique_idxs[row][2], row_indices)

    source_unique_idxs =
        (source_unique_idxs_i, source_unique_idxs_j, source_unique_idxs_e)
    target_unique_idxs =
        (target_unique_idxs_i, target_unique_idxs_j, target_unique_idxs_e)

    source_global_elem_lidx = nothing
    target_global_elem_lidx = nothing

    return LinearMap(
        source_space,
        target_space,
        weights,
        source_unique_idxs,
        target_unique_idxs,
        col_indices,
        row_indices,
        out_type,
        source_global_elem_lidx,
        target_global_elem_lidx,
    )
end
