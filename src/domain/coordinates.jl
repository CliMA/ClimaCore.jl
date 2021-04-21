"""
    interpolate(coords::NTuple{4}, ξ1, ξ2)

Interpolate between `coords` by parameters `ξ1, ξ2` in the interval `[-1,1]`.

The type of interpolation is determined by the element type of `coords`:
- `SVector`: use bilinear interpolation
"""
function interpolate(coords::NTuple{4, V}, ξ1, ξ2) where {V <: SVector}
    ((1 - ξ1) * (1 - ξ2)) / 4 .* coords[1] .+
    ((1 + ξ1) * (1 - ξ2)) / 4 .* coords[2] .+
    ((1 - ξ1) * (1 + ξ2)) / 4 .* coords[3] .+
    ((1 + ξ1) * (1 + ξ2)) / 4 .* coords[4]
end
