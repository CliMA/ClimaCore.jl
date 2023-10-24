using ClimaCore.DataLayouts
using ClimaComms


"""
    LinearMap{S, T, W, I, V}

stores information on the TempestRemap map and the source and target data:

where:
 - `source_space` and `target_space` are ClimaCore's 2D spaces.
 - `weights` is a vector of remapping weights. (length = number of overlap-mesh nodes).
 - `source_local_idxs` a 3-element Tuple with 3 index vectors, representing local (i,j,elem) indices on the source mesh. (length of each index vector = number of overlap-mesh nodes)
 - `target_local_idxs` is the same as `source_local_idxs` but for the target mesh.
 - `row_indices` are the target row indices from TempestRemap. (length = number of overlap-mesh nodes)
 - `out_type` string that defines the output type.

"""
struct LinearMap{S, T, W, I, V} # make consistent with / move to regridding.jl
    source_space::S
    target_space::T
    weights::W
    source_local_idxs::I
    target_local_idxs::I
    row_indices::V
    out_type::String
end


"""
    remap!(target::IJFH{S, Nqt}, R::LinearMap, source::IJFH{S, Nqs})
    remap!(target::Fields.Field, R::LinearMap, source::Fields.Field)

Applies the remapping `R` to a `source`
Field and stores the result in `target`.
"""
function remap! end

# This version of this function is used for serial remapping
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
            view(R.source_local_idxs[1], n)[1],
            view(R.source_local_idxs[2], n)[1],
            view(R.source_local_idxs[3], n)[1],
        )
        it, jt, et = (
            view(R.target_local_idxs[1], n)[1],
            view(R.target_local_idxs[2], n)[1],
            view(R.target_local_idxs[3], n)[1],
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

# This version of this function is used for distributed remapping
function remap!(target::Fields.Field, R::LinearMap, source::Fields.Field)
    if Spaces.topology(axes(target)).context isa
       ClimaComms.SingletonCommsContext
        @assert axes(source) == R.source_space
        @assert axes(target) == R.target_space
        # we use the tempestremap cgll representation
        # it will set the redundant nodes to zero
        remap!(Fields.field_values(target), R, Fields.field_values(source))
        return target
    else
        # For now, the source data must be on a non-distributed space
        @assert Spaces.topology(axes(source)).context isa
                ClimaComms.SingletonCommsContext

        target_array = parent(target)
        source_array = parent(source)

        fill!(target_array, zero(eltype(target_array)))
        Nf = size(target_array, 3)

        # ideally we would use the tempestremap dgll (redundant node) representation
        # unfortunately, this doesn't appear to work quite as well (for out_type = dgll) as the cgll
        for (n, wt) in enumerate(R.weights)
            # extract local source indices
            is, js, es = (
                view(R.source_local_idxs[1], n)[1],
                view(R.source_local_idxs[2], n)[1],
                view(R.source_local_idxs[3], n)[1],
            )

            # extract local target indices
            it, jt, et = (
                view(R.target_local_idxs[1], n)[1],
                view(R.target_local_idxs[2], n)[1],
                view(R.target_local_idxs[3], n)[1],
            )

            # multiply source data by weights to get target data
            # only use local weights - i.e. et, es != 0
            if (et != 0)
                for f in 1:Nf
                    target_array[it, jt, f, et] +=
                        wt * source_array[is, js, f, es]
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
    generate_map(target_space, source_space; in_type="cgll", out_type="cgll")

Generate the remapping weights from TempestRemap, returning a `LinearMap` object. This should only be called once.
"""
function generate_map(
    target_space::Spaces.SpectralElementSpace2D,
    source_space::Spaces.SpectralElementSpace2D;
    target_space_distr = nothing,
    meshfile_source = tempname(),
    meshfile_target = tempname(),
    meshfile_overlap = tempname(),
    weightfile = tempname(),
    in_type = "cgll",
    out_type = "cgll",
)
    if (target_space_distr != nothing)
        comms_ctx = ClimaComms.context(target_space_distr)
    else
        comms_ctx = ClimaComms.context(target_space)
    end

    if ClimaComms.iamroot(comms_ctx)
        # write meshes and generate weights on root process (using global indices)
        write_exodus(meshfile_source, Spaces.topology(source_space))
        write_exodus(meshfile_target, Spaces.topology(target_space))
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

        # re-order our inds to TR's ordering of col/row inds and weights to avoid unnecessary allocations
        # extract i, j, e components of TR indexing
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
        row_indices = nothing
        source_unique_idxs = nothing
        target_unique_idxs = nothing
    end

    weights = ClimaComms.bcast(comms_ctx, weights)
    row_indices = ClimaComms.bcast(comms_ctx, row_indices)
    source_unique_idxs = ClimaComms.bcast(comms_ctx, source_unique_idxs)
    target_unique_idxs = ClimaComms.bcast(comms_ctx, target_unique_idxs)
    ClimaComms.barrier(comms_ctx)

    if target_space_distr != nothing
        # Create map from unique (TempestRemap convention) to local element indices
        target_local_elem_gidx = target_space_distr.topology.local_elem_gidx # gidx = local_elem_gidx[lidx]
        target_global_elem_lidx = Dict{Int, Int}() # inverse of local_elem_gidx: lidx = global_elem_lidx[gidx]
        for (lidx, gidx) in enumerate(target_local_elem_gidx)
            target_global_elem_lidx[gidx] = lidx
        end

        # store only the inds local to this process (set inds from other processes to 0)
        # TODO this is not a great implementation, but we needs the lengths of each array to be = |weights|
        target_local_idxs = map(vec -> similar(vec), target_unique_idxs)
        for (n, wt) in enumerate(weights)
            target_elem_gidx = view(target_unique_idxs[3], n)[1]
            if !(target_elem_gidx in keys(target_global_elem_lidx))
                it = 0
                jt = 0
                et = 0
            else
                # get global i, j inds but local elem index e
                it = view(target_unique_idxs[1], n)[1]
                jt = view(target_unique_idxs[2], n)[1]
                et = target_global_elem_lidx[target_elem_gidx]
            end
            target_local_idxs[1][n] = it
            target_local_idxs[2][n] = jt
            target_local_idxs[3][n] = et
        end
    else
        target_local_idxs = target_unique_idxs
    end

    source_local_idxs = source_unique_idxs

    return LinearMap(
        source_space,
        target_space,
        weights,
        source_local_idxs,
        target_local_idxs,
        row_indices,
        out_type,
    )
end
