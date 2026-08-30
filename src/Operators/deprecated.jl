# Backwards-compatibility aliases for the weak-form spectral element
# operators. The weak operators are the WeakForm variants of their strong-form
# counterparts (e.g., Divergence{WeakForm}); the aliases below keep the
# WeakDivergence/WeakGradient/WeakCurl names working. Nothing in this repo uses
# them — src, ext, tests, examples, tutorials, and benchmarks all call the
# parameterized names, and the docs page (docs/src/operators.md) documents
# those — so they are kept only for downstream packages. As with
# src/DataLayouts/deprecated.jl, these are plain aliases without deprecation
# warnings. Remove them once downstream consumers have migrated to the
# parameterized names (type-alias cleanup, Phase 4 of
# https://github.com/CliMA/ClimaCore.jl/issues/2554).
# get_node at the end of this file is live API used by the Remapping module.

# Weak-form documentation lives on the parameterized operators
# (`Divergence`/`Gradient`/`Curl`, under their "Weak form" headings); these
# aliases are undocumented backward-compatibility bindings.
const WeakDivergence = Divergence{WeakForm}
const WeakGradient = Gradient{WeakForm}
const WeakCurl = Curl{WeakForm}

# The finite-difference biased interpolation operators are named for the
# vertical direction they lean toward: `BottomBiasedC2F`/`BottomBiasedF2C` take
# the value below a node, `TopBiasedC2F`/`TopBiasedF2C` the value above. The
# `LeftBiased`/`RightBiased` spellings are retained for downstream packages; the
# `false` third argument leaves the old names unexported (the default would
# export them), while accessing them still emits a deprecation warning.
Base.@deprecate_binding LeftBiasedC2F BottomBiasedC2F false
Base.@deprecate_binding LeftBiasedF2C BottomBiasedF2C false
Base.@deprecate_binding RightBiasedC2F TopBiasedC2F false
Base.@deprecate_binding RightBiasedF2C TopBiasedF2C false

# Nodal getter for Fields. Not deprecated: the Remapping module and its CUDA
# extension call it, so it must move elsewhere before this file is deleted.
# A 1-dimensional index is a column of a 1D space, so its second index is 1.
Base.@propagate_inbounds function get_node(
    parent_space, field::Fields.Field, ij::CartesianIndex{N}, slabidx,
) where {N}
    space = reconstruct_placeholder_space(axes(field), parent_space)
    _v =
        if space isa Spaces.FaceExtrudedFiniteDifferenceSpace ||
           space isa Spaces.FaceFiniteDifferenceSpace
            slabidx.v + half
        elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
               space isa Spaces.AbstractSpectralElementSpace ||
               space isa Spaces.CenterFiniteDifferenceSpace
            slabidx.v
        else
            error("invalid space")
        end
    (i, j) = N == 1 ? (ij[1], 1) : Tuple(ij)
    return Fields.field_values(field)[isnothing(_v) ? 1 : _v, i, j, slabidx.h]
end
