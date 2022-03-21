import ClimaCore.DataLayouts: IJFH


"""
    LinearTempestRemap{T, S, M, C, V}

stores info on the TempestRemap map and the source and target data
"""
struct LinearMap{S, T, W, I, V} # make consistent with / move to regridding.jl
    source_space::S
    target_space::T
    weights::W # remapping weights
    source_idxs_i::I # source indices
    source_idxs_j::I # source indices
    source_idxs_e::I # source indices
    target_idxs_i::I # target indices
    target_idxs_j::I # target indices
    target_idxs_e::I # target indices
    col::V
    row::V
end

"""
    remap!(target, R, source)

applies the remapping of a `source` Field to a `target` Field using a sparse matrix multiply
"""
function remap!(
    target::IJFH{S, Nqt},
    source::IJFH{S, Nqs},
    R::LinearMap,
) where {S, Nqt, Nqs}
    source_array = parent(source)
    target_array = parent(target)

    fill!(target_array, zero(eltype(target_array)))
    Nf = size(target_array, 3)

    # ideally we would use the tempestremap dgll (redundant node) representation
    # unfortunately, this doesn't appear to work quite as well as the cgll
    
    # for (source_idx, target_idx, wt) in
    #     zip(R.source_idxs, R.target_idxs, R.weights)
    for n in collect(1:length(R.weights))
        wt = view(R.weights,n)
        is, js, es = (view(R.source_idxs_i,n), view(R.source_idxs_j,n), view(R.source_idxs_e,n))
        it, jt, et = (view(R.target_idxs_i,n), view(R.target_idxs_j,n), view(R.target_idxs_e,n))
        for f in 1:Nf
            target_array[it, jt, f, et] += wt * view(source_array, is, js, f, es)[1]
        end
    end
    return target
end
function remap!(target::Fields.Field, source::Fields.Field, R::LinearMap)
    @assert axes(source) == R.source_space
    @assert axes(target) == R.target_space
    # we use the tempestremap cgll representation
    # it will set the redundant nodes to zero
    remap!(Fields.field_values(target), Fields.field_values(source), R)
    # use unweighted dss to broadcast the so far unpopulated (redundant) nodes from their unique node counterparts
    # (we could get rid of this if we added the redundant nodes to the matrix)
    Spaces.horizontal_dss!(target)
    return target
end


"""
    generate_map

offline generation of remapping weights using TempestRemap
"""
function generate_map(
    target_space::Spaces.SpectralElementSpace2D,
    source_space::Spaces.SpectralElementSpace2D;
    meshfile_source = tempname(),
    meshfile_target = tempname(),
    meshfile_overlap = tempname(),
    weightfile = tempname(),
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
        in_type = "cgll",
        in_np = Spaces.Quadratures.degrees_of_freedom(
            source_space.quadrature_style,
        ),
        out_type = "cgll",
        out_np = Spaces.Quadratures.degrees_of_freedom(
            target_space.quadrature_style,
        ),
    )

    # read weight data
    weights, col, row = NCDataset(weightfile, "r") do ds_wt
        (Array(ds_wt["S"]), Array(ds_wt["col"]), Array(ds_wt["row"]))
    end

    # we need to be able to look up the indices of unique nodes

    source_unique_idxs = collect(Spaces.unique_nodes(source_space))[col]
    target_unique_idxs = collect(Spaces.unique_nodes(target_space))[row]

    source_unique_idxs_i = map(x -> source_unique_idxs[x][1][1],collect(1:length(source_unique_idxs)))
    source_unique_idxs_j = map(x -> source_unique_idxs[x][1][2],collect(1:length(source_unique_idxs)))
    source_unique_idxs_e = map(x -> source_unique_idxs[x][2],collect(1:length(source_unique_idxs)))
    target_unique_idxs_i = map(x -> target_unique_idxs[x][1][1],collect(1:length(target_unique_idxs)))
    target_unique_idxs_j = map(x -> target_unique_idxs[x][1][2],collect(1:length(target_unique_idxs)))
    target_unique_idxs_e = map(x -> target_unique_idxs[x][2],collect(1:length(target_unique_idxs)))

    return LinearMap(
        source_space,
        target_space,
        weights,
        source_unique_idxs_i,
        source_unique_idxs_j,
        source_unique_idxs_e,
        target_unique_idxs_i,
        target_unique_idxs_j,
        target_unique_idxs_e,
        col,
        row,
    )
end
