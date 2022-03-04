import ClimaCore.DataLayouts: IJFH


"""
    LinearTempestRemap{T, S, M, C}

stores info on the TempestRemap map and the source and target data
"""
struct LinearMap{S,T,W,I} # make consistent with / move to regridding.jl
    source_space::S
    target_space::T
    weights::W # remapping weights
    source_idxs::I # source indices
    target_idxs::I # target indices
end

"""
    remap!(target, R, source)

applies the remapping
"""
function remap!(target::IJFH{S,Nqt}, source::IJFH{S,Nqs}, R::LinearMap) where {S,Nqt,Nqs}
    source_array = parent(source)
    target_array = parent(target)

    fill!(target_array, zero(eltype(target_array)))
    Nf = size(target_array,3)

    # ideally we would use the tempestremap dgll (redundant node) representation
    # unfortunately, this doesn't appear to work quite as well as the cgll
    f = 1
    for (source_idx, target_idx, wt) = zip(R.source_idxs, R.target_idxs, R.weights)
        (is,js), es = source_idx
        (it,jt), et = target_idx
        for f = 1:Nf
            target_array[it,jt,f,et] += wt * source_array[is,js,f,es]
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
    # we can make the redundant nodes the same by applying the unweighted dss
    # we could get rid of this if we added the redundant nodes to the matrix
    Spaces.horizontal_dss!(target)
    return target
end


"""
    generate_map

offline generation of remapping weights using TempestRemap
"""
function generate_map(
    target_space::Spaces.SpectralElementSpace2D,
    source_space::Spaces.SpectralElementSpace2D
    ;
    meshfile_source = tempname(),
    meshfile_target = tempname(),
    meshfile_overlap = tempname(),
    weightfile = tempname()
    )

    # write meshes
    write_exodus(meshfile_source, source_space.topology)
    write_exodus(meshfile_target, target_space.topology)
    overlap_mesh(meshfile_overlap, meshfile_source, meshfile_target)
    remap_weights(
        weightfile,
        meshfile_source,
        meshfile_target,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Spaces.Quadratures.degrees_of_freedom(source_space.quadrature_style),
        out_type = "cgll",
        out_np = Spaces.Quadratures.degrees_of_freedom(target_space.quadrature_style),
    )

    # read weight data
    weights, col, row = NCDataset(weightfile,"r") do ds_wt
        (Array(ds_wt["S"]), Array(ds_wt["col"]), Array(ds_wt["row"]))
    end

    # we need to be able to look up the index
    source_unique_idxs = collect(Spaces.unique_nodes(source_space))
    target_unique_idxs = collect(Spaces.unique_nodes(target_space))

    return LinearMap(source_space, target_space, weights, source_unique_idxs[col], target_unique_idxs[row])
end