# Backwards-compatibility shims for the pre-rewrite DataLayouts API.
#
# These are not deprecated with warnings — they're plain aliases — so old code
# continues to type-check without noise. Remove this file when all downstream
# consumers have migrated to the new names.

export AbstractData, IJFH, IJHF

# Old: AbstractData{T}
# New: DataLayout{T, N, F, S, A}
const AbstractData{T} = DataLayout{T}

# Old: IJFH/IJHF layouts with 4-D parent arrays
# New: VIJFH/VIJHF layouts with Nv = 1 and 5-D parent arrays
# Used by ClimaCoreTempestRemap's remap!(target::IJFH{T, Nq}, ...).
const IJFH{T, Nij} = VIJFH{T, 1, Nij, Nij, nothing}
const IJHF{T, Nij} = VIJHF{T, 1, Nij, Nij, nothing}

# Old: getindex/setindex! with a 5-D CartesianIndex(i, j, _, v, h)
# New: getindex/setindex! with a 4-D CartesianIndex(v, i, j, h)
# Used by ClimaCoreTempestRemap in slab(data, h)[CartesianIndex(i, j, 1, 1, 1)]
# and similar functions. This overrides the generic AbstractArray interpretation
# of a CartesianIndex{5}; until it is removed, new code should not add a
# trailing singleton coordinate when indexing into VIJFH/VIJHF layouts.
@propagate_inbounds Base.getindex(data::VIJHWithF, I::CartesianIndex{5}) =
    getindex(data, CartesianIndex(I[4], I[1], I[2], I[5]))
@propagate_inbounds Base.setindex!(data::VIJHWithF, value, I::CartesianIndex{5}) =
    setindex!(data, value, CartesianIndex(I[4], I[1], I[2], I[5]))

# Old: single-point layouts ignore every component of a 5-D CartesianIndex
# New: single-point layouts without any index
@propagate_inbounds Base.getindex(data::DataF, I::CartesianIndex{5}) = data[]
@propagate_inbounds Base.setindex!(data::DataF, value, I::CartesianIndex{5}) =
    setindex!(data, value)
