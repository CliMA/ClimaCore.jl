#=
There are 4 possible types of general bidiagonal matrices:
1 2 . . .          1 . . . .
. 1 2 . .    or    2 1 . . .
. . 1 2 .          . 2 1 . .
or
1 2 .          1 . .
. 1 2          2 1 .
. . 1    or    . 2 1
. . .          . . 2
. . .          . . .
=#
struct GeneralBidiagonal{T,AT<:AbstractVector{T}} <: AbstractMatrix{T}
  d::AT
  d2::AT
  isUpper::Bool
  nrows::Int
  ncols::Int
end
function GeneralBidiagonal(
  ::Type{AT},
  isUpper::Bool,
  nrows::Int,
  ncols::Int,
) where {AT}
  nd = min(nrows, ncols)
  nd2 = (isUpper ? ncols : nrows) > nd ? nd : nd - 1
  @assert nd2 > 0
  d = AT(undef, nd)
  d2 = AT(undef, nd2)
  return GeneralBidiagonal{eltype(d), typeof(d)}(d, d2, isUpper, nrows, ncols)
end

import Base: size, getindex, setindex!
size(A::GeneralBidiagonal) = (A.nrows, A.ncols)
function getindex(A::GeneralBidiagonal, i::Int, j::Int)
  @boundscheck 1 <= i <= A.nrows && 1 <= j <= A.ncols
  if i == j
      return A.d[i]
  elseif A.isUpper && j == i + 1
      return A.d2[i]
  elseif !A.isUpper && i == j + 1
      return A.d2[j]
  else
      return zero(eltype(A))
  end
end
function setindex!(A::GeneralBidiagonal, v, i::Int, j::Int)
  @boundscheck 1 <= i <= A.nrows && 1 <= j <= A.ncols
  if i == j
      A.d[i] = v
  elseif A.isUpper && j == i + 1
      A.d2[i] = v
  elseif !A.isUpper && i == j + 1
      A.d2[j] = v
  elseif !iszero(v)
      throw(ArgumentError(
          "Setting A[$i, $j] to $v will make A no longer be GeneralBidiagonal"
      ))
  end
end

import LinearAlgebra: mul!
function mul!(
  C::AbstractVector,
  A::GeneralBidiagonal,
  B::AbstractVector,
  α::Number,
  β::Number,
)
  if A.nrows != length(C)
      throw(DimensionMismatch(
          "A has $(A.nrows) rows, but C has length $(length(C))"
      ))
  end
  if A.ncols != length(B)
      throw(DimensionMismatch(
          "A has $(A.ncols) columns, but B has length $(length(B))"
      ))
  end
  if iszero(α)
      return LinearAlgebra._rmul_or_fill!(C, β)
  end
  nd = length(A.d)
  nd2 = length(A.d2)
  @inbounds if A.isUpper
      if nd2 == nd
          @views @. C = α * (A.d * B[1:nd] + A.d2 * B[2:nd + 1]) + β * C
      else
          @views @. C[1:nd - 1] =
              α * (A.d[1:nd - 1] * B[1:nd - 1] + A.d2 * B[2:nd]) +
              β * C[1:nd - 1]
          C[nd] = α * A.d[nd] * B[nd] + β * C[nd]
      end
  else
      C[1] = α * A.d[1] * B[1] + β * C[1]
      @views @. C[2:nd] =
          α * (A.d[2:nd] * B[2:nd] + A.d2[1:nd - 1] * B[1:nd - 1]) +
          β * C[2:nd]
      if nd2 == nd
          C[nd + 1] = α * A.d2[nd] * B[nd] + β * C[nd + 1]
      end
  end
  C[nd2 + 2:end] .= zero(eltype(C))
  return C
end
function mul!(
  C::Tridiagonal,
  A::GeneralBidiagonal,
  B::GeneralBidiagonal,
  α::Number,
  β::Number,
)
  if A.nrows != B.ncols || A.nrows != size(C, 1)
      throw(DimensionMismatch(string(
          "A has $(A.nrows) rows, B has $(B.ncols) columns, and C has ",
          "$(size(C, 1)) rows/columns, but all three must match"
      )))
  end
  if A.ncols != B.nrows
      throw(DimensionMismatch(
          "A has $(A.ncols) columns, but B has $(B.rows) rows"
      ))
  end
  if A.isUpper && B.isUpper
      throw(ArgumentError(
          "A and B are both upper bidiagonal, so C is not tridiagonal"
      ))
  end
  if !A.isUpper && !B.isUpper
      throw(ArgumentError(
          "A and B are both lower bidiagonal, so C is not tridiagonal"
      ))
  end
  if iszero(α)
      return LinearAlgebra._rmul_or_fill!(C, β)
  end
  nd = length(A.d) # == length(B.d)
  nd2 = length(A.d2) # == length(B.d2)
  @inbounds if A.isUpper # && !B.isUpper
      if nd2 == nd
          #                   3 . .
          # 1 2 . . .         4 3 .         13+24 23    .
          # . 1 2 . .    *    . 4 3    =    14    13+24 23
          # . . 1 2 .         . . 4         .     14    13+24
          #                   . . .
          @. C.d = α * (A.d * B.d + A.d2 * B.d2) + β * C.d
      else
          # 1 2 .                           13+24 23    .     .     .
          # . 1 2         3 . . . .         14    13+24 23    .     .
          # . . 1    *    4 3 . . .    =    .     14    13    .     .
          # . . .         . 4 3 . .         .     .     .     .     .
          # . . .                           .     .     .     .     .
          @views @. C.d[1:nd - 1] =
              α * (A.d[1:nd - 1] * B.d[1:nd - 1] + A.d2 * B.d2) +
              β * C.d[1:nd - 1]
          C.d[nd] = α * A.d[nd] * B.d[nd] + β * C.d[nd]
      end
      @views @. C.du[1:nd - 1] =
          α * A.d2[1:nd - 1] * B.d[2:nd] + β * C.du[1:nd - 1]
      @views @. C.dl[1:nd - 1] =
          α * A.d[2:nd] * B.d2[1:nd - 1] + β * C.dl[1:nd - 1]
  else # !A.isUpper && B.isUpper
      C.d[1] = α * A.d[1] * B.d[1] + β * C.d[1]
      @views @. C.d[2:nd] =
          α * (A.d[2:nd] * B.d[2:nd] + A.d2[1:nd - 1] * B.d2[1:nd - 1]) +
          β * C.d[2:nd]
      if nd2 == nd
          # 1 . .                           13    14    .     .     .
          # 2 1 .         3 4 . . .         23    13+24 14    .     .
          # . 2 1    *    . 3 4 . .    =    .     23    13+24 14    .
          # . . 2         . . 3 4 .         .     .     23    24    .
          # . . .                           .     .     .     .     .
          C.d[nd + 1] = α * A.d2[nd] * B.d2[nd] + β * C.d[nd + 1]
      # else
          #                   3 4 .
          # 1 . . . .         . 3 4         13    14    .
          # 2 1 . . .    *    . . 3    =    23    13+24 14
          # . 2 1 . .         . . .         .     23    13+24
          #                   . . .
      end
      @views @. C.du[1:nd2] =
          α * A.d[1:nd2] * B.d2 + β * C.du[1:nd2]
      @views @. C.dl[1:nd2] =
          α * A.d2 * B.d[1:nd2] + β * C.dl[1:nd2]
  end
  C.d[nd2 + 2:end] .= zero(eltype(C))
  C.du[nd2 + 1:end] .= zero(eltype(C))
  C.dl[nd2 + 1:end] .= zero(eltype(C))
  return C
end
#=
Other possible GeneralBidiagonal multiplications (not implemented):
U * U:
1 2 .         3 4 .         13    14+23 24
. 1 2    *    . 3 4    =    .     13    14+23
. . 1         . . 3         .     .     13
                3 4 .
1 2 . . .         . 3 4         13    14+23 24
. 1 2 . .    *    . . 3    =    .     13    14+23
. . 1 2 .         . . .         .     .     13
                . . .
1 2 .                           13    14+23 24    .     .
. 1 2         3 4 . . .         .     13    14+23 24    .
. . 1    *    . 3 4 . .    =    .     .     13    14    .
. . .         . . 3 4 .         .     .     .     .     .
. . .                           .     .     .     .     .
L * L:
1 . .         3 . .         13    .     .
2 1 .    *    4 3 .    =    14+23 13    .
. 2 1         . 4 3         24    14+23 13
                3 . .
1 . . . .         4 3 .         13    .     .
2 1 . . .    *    . 4 3    =    14+23 13    .
. 2 1 . .         . . 4         24    14+23 13
                . . .
1 . .                           13    .     .     .     .
2 1 .         3 . . . .         14+23 13    .     .     .
. 2 1    *    4 3 . . .    =    24    14+23 13    .     .
. . 2         . 4 3 . .         .     24    23    .     .
. . .                           .     .     .     .     .
=#

#=
A = [-I     0      α J13;
   0      -I     α J23;
   α J31  α J32  -I   ] =
  [-I   0    A13;
   0    -I   A23;
   A31  A32  -I ]
b = [b1; b2; b3]
x = [x1; x2; x3]
Solving A x = b:
  -x1 + A13 x3 = b1 ==> x1 = -b1 + A13 x3  (1)
  -x2 + A23 x3 = b2 ==> x2 = -b2 + A23 x3  (2)
  A31 x1 + A32 x2 - x3 = b3  (3)
Substitute (1) and (2) into (3):
  A31 (-b1 + A13 x3) + A32 (-b2 + A23 x3) - x3 = b3 ==>
  (-I + A31 A13 + A32 A23) x3 = b3 + A31 b1 + A32 b2 ==>
  x3 = (-I + A31 A13 + A32 A23) \ (b3 + A31 b1 + A32 b2)
Finally, use (1) and (2) to get x1 and x2.
Note: The tridiagonal matrix S = -I + A31 A13 + A32 A23 is the
    "Schur complement" of [-I 0; 0 -I] (the top-left 4 blocks) in A.
=#
function schur_solve!(
  x1::AbstractVector,
  x2::AbstractVector,
  x3::AbstractVector,
  J13::GeneralBidiagonal,
  J23::GeneralBidiagonal,
  J31::GeneralBidiagonal,
  J32::GeneralBidiagonal,
  b1::AbstractVector,
  b2::AbstractVector,
  b3::AbstractVector,
  α::Number,
  S::Tridiagonal,
)
  # LHS = -I + α^2 J31 J13 + α^2 J32 J23
  S.dl .= 0
  S.d .= -1
  S.du .= 0
  mul!(S, J31, J13, α^2, 1)
  mul!(S, J32, J23, α^2, 1)

  # RHS = b3 + α J31 b1 + α J32 b2
  x3 .= b3
  mul!(x3, J31, b1, α, 1)
  mul!(x3, J32, b2, α, 1)

  # x3 = LHS \ RHS
  # TODO: LinearAlgebra will compute lu! and then ldiv! in seperate steps.
  #       The Thomas algorithm can do this in one step:
  #       https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm.
  ldiv!(lu!(S), x3)

  # x1 = -b1 + α J13 x3
  x1 .= b1
  mul!(x1, J13, x3, α, -1)

  # x2 = -b2 + α J23 x3
  x2 .= b2
  mul!(x2, J23, x3, α, -1)
end