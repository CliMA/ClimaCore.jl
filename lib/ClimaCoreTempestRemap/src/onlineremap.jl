import ClimaCore.DataLayouts: IJFH


"""
    LinearTempestRemap{T, S, M, C, V}

stores information on the TempestRemap map and the source and target data:

where:
 - `source_space` and `target_space` are ClimaCore's 2D spaces.
 - `weights` is a vector of remapping weights. (length = number of overlap-mesh nodes)
 - `source_idxs` a 3-element Tuple with 3 index vectors, representing (i,j,elem) indices on the source mesh. (length of each index vector = number of overlap-mesh nodes)
 - `target_idxs` is the same as `source_idxs` but for the target mesh.
 - `col_indices` are the source column indices from TempestRemap. (length = number of overlap-mesh nodes)
 - `row_indices` are the target row indices from TempestRemap. (length = number of overlap-mesh nodes)

"""
struct LinearMap{S, T, W, I, V} # make consistent with / move to regridding.jl
    source_space::S
    target_space::T
    weights::W
    source_idxs::I
    target_idxs::I
    col_indices::V
    row_indices::V
    out_type::String
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
        Spaces.dss_interior_faces!(topology, target)
        Spaces.dss_local_vertices!(topology, target)
    end
    return target
end

"""
    remap!(target::Field, R::LinearMap, source::Field)

Applies the remapping `R` to a `source` Field, storing the result in `target`.
"""
function remap!(target::Fields.Field, R::LinearMap, source::Fields.Field)
    @assert axes(source) == R.source_space
    @assert axes(target) == R.target_space
    # we use the tempestremap cgll representation
    # it will set the redundant nodes to zero
    remap!(Fields.field_values(target), R, Fields.field_values(source))
    return target
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
    meshfile_source = tempname(),
    meshfile_target = tempname(),
    meshfile_overlap = tempname(),
    weightfile = tempname(),
    in_type = "cgll",
    out_type = "cgll",
)

    # write meshes and generate weights
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

    return LinearMap(
        source_space,
        target_space,
        weights,
        source_unique_idxs,
        target_unique_idxs,
        col_indices,
        row_indices,
        out_type,
    )
end
