#####
##### Column mesh
#####

export n_cells, n_faces
export FiniteDifferenceSpace, coords
export AbstractDataLocation, CellCent, CellFace
export All, Ghost, Boundary, Interior
export ColumnMinMax, ColumnMin, ColumnMax
export n_hat, binary

#####
##### Data locations
#####

"""
    AbstractDataLocation

Subtypes are used for dispatching
on the data location
"""
abstract type Staggering end

""" cell center location """
struct CellCent <: Staggering end

""" cell face location """
struct CellFace <: Staggering end

dual_type(::CellCent) = CellFace
dual_type(::CellFace) = CellCent

struct CollocatedCoordinates{DL <: Staggering, A}
    h::A
    Δh::A
end

Base.length(c::CollocatedCoordinates) = length(c.h)

function Base.show(io::IO, c::CollocatedCoordinates)
    print(io, "\n")
    println(io, "  size(h) = ", length(c))
    println(io, "  h       = ", c.h)
    println(io, "  Δh      = ", c.Δh)
end

#####
##### Local stencils
#####

"""
    Stencil

A local stencil.
"""
struct Stencil{DL <: Staggering, hi, lo, A}
    array::A
end
array(stencil::Stencil) = stencil.array

abstract type AbstractOperationType end
struct Collocated <: AbstractOperationType end
struct FaceToCent <: AbstractOperationType end
struct CentToFace <: AbstractOperationType end
struct FaceToFace <: AbstractOperationType end
struct CentToCent <: AbstractOperationType end

"""
    LocalOperator

A local operator, used for interpolations or
finite-differences. This is effectively a
single row of the global operator.
"""
struct LocalOperator{OpType <: AbstractOperationType, hi, lo, A}
    array::A
end
Base.parent(stencil::LocalOperator) = stencil.array
Base.length(stencil::LocalOperator) = length(parent(stencil))
undertype(operator::LocalOperator) = eltype(operator.array)
operation_type(::Type{T}) where {OpType, T <: LocalOperator{OpType}} = OpType

# Likely satisfies needs for local collocated operations:
Base.:(*)(
    operator::LocalOperator{Collocated},
    stencil::Stencil{DataLoc},
) where {DataLoc} = dot(array(A) .* array(B))

#=
function Base.:*(stencil::LocalOperator{FaceToCent}, v::SVector) # v ∈ face 
    return LocalOperator{dual_type()}(dot(array(stencil), v))
end


 

(∇ α ∇).(T[1:3])

β = (∇).(T[1:3])

∇((∇).(T[1:3])[1:2])[1]
d
# ?
function Base.:(*)(
        op_lo::LocalOperator{FaceToCent, 1, 0},
        op_hi::LocalOperator{FaceToCent, 1, 0},
        stencil::LocalStencil{CellFace, 1, 1}
    )
    ∇stencil_lo = dot(array(op_lo), array(stencil)[1:2])
    ∇stencil_hi = dot(array(op_hi), array(stencil)[2:3])
    return LocalStencil{CellCent, 0, 1}(∇stencil_lo, ∇stencil_hi)
end



cent_stencil
cent_stencil = ∇_z(cent_stencil, grid)
cent_stencil = ∇_z(cent_stencil, grid)

# RHS eval: https://github.com/CliMA/SingleColumnModels.jl/blob/master/src/EDMF/edmf_funcs.jl#L88-L147
# Heat equation test: https://github.com/CliMA/SingleColumnModels.jl/blob/master/test/PDEs/HeatEquation.jl#L169
T = Field(CellCent())
# For any prognostic field (e.g., T ∈ CellCent, rhs[:T] ∈ CellCent)
# all other fields are temporary
for (i,point) in enumerate(ColumSpace())
    T_stencil_CO2 = local_stencil(T, i, SecondOrder())
    T_stencil_up = local_stencil(T, i, Upwind())    

    ∇T_s = ∇_staggered(T_stencil_CO2) # Dual-field of CellCent
    # rhs .+= ∇T # Not allowed since rhs ∈ CellCent
    ∇²T = ∇_staggered(T_stencil_CO2) # Dual-field of CellFace
    rhs .+= ∇²T
    
    ∇T_s = ∇_collocated(T_stencil_CO2) # lives in CellCent
    ∇²T_s = ∇²_collocated(T_stencil_CO2) # lives in CellCent
    rhs .+= ∇T_s # allowed
end
=#

n_points_hi(::LocalOperator{OpType, hi}) where {OpType, hi} = hi
n_points_lo(::LocalOperator{OpType, hi, lo}) where {OpType, hi, lo} = lo

struct SparseMatrix{NRows, NCols, A}
    array::A
    function SparseMatrix{NRows, NCols}(
        array::AbstractArray,
    ) where {NRows, NCols}
        return new{NRows, NCols, typeof(array)}(array)
    end
end

Base.parent(sm::SparseMatrix) = sm.array
Base.eltype(sparse_mat::SparseMatrix) = eltype(parent(sparse_mat))
undertype(sm::SparseMatrix) = undertype(first(parent(sm)))

"""
    n_diag(sparse_mat::SparseMatrix)

Number of rows in the sparse matrix.
"""
n_rows(::SparseMatrix{NR}) where {NR} = NR

"""
    n_columns(sparse_mat::SparseMatrix)

Number of columns in the sparse matrix.
"""
n_columns(::SparseMatrix{NR, NC}) where {NR, NC} = NC

# TODO: this is working but unoptimized
function matrix(sparse_mat::SparseMatrix)
    arr = parent(sparse_mat)
    nrows = n_rows(sparse_mat)
    ncols = n_columns(sparse_mat)
    FT = undertype(sparse_mat)
    mat = zeros(FT, nrows, ncols)
    for i in 1:nrows
        local_operator = arr[i]
        n_lo = n_points_lo(local_operator)
        n_hi = n_points_hi(local_operator)
        for j in (1 + n_lo):(ncols - n_hi)
            arr_loc_op = parent(local_operator)
            if i == j
                mat[i, (j - n_lo):(j + n_hi)] .= parent(local_operator)
            end
        end
    end
    return mat
end

function Base.show(io::IO, sparse_mat::SparseMatrix)
    show(io, "text/plain", matrix(sparse_mat))
    println(io)
end



# Toggles:
# - order of accuracy (second order)
# - skewdness (centered / one-sided) (centered + upwind)
# https://github.com/CliMA/SingleColumnModels.jl/blob/master/src/Grids/FiniteDifferenceGrids.jl#L129-L195
# /|
# /|-o-|-x-|-o-|-o-|
# /|


# encode: number of ghost points at top and bottom


#=
TODO: 
struct FaceFiniteDifferenceSpace{T,C} <: AbstractSpace
    topology::T
    internal_coordinates::IC
    ghost_coordinates_min::GC
    ghost_coordinates_max::GC
    all_coordinates::C
    # geometry info:
    # - cell heights
    # -

end
function FaceFiniteDifferenceSpace(topology, nghost=1)
    all_coordinates = ...
    ghost_coordinates_min = view(all_coordinates, 1:nghost)
    internal_coordinates = view(all_coordinates, nghost+1:n+nghost)
    ghost_coordinates_max = view(all_coordinates, n+nghost+1:n+2nghost)
    FaceFiniteDifferenceSpace(topology,
        internal_coordinates,
        ghost_coordinates_min,
        ghost_coordinates_max,
        all_coordinates)
end



struct CenterFiniteDifferenceSpace{T,C} <: AbstractSpace
    topology::T
    coordinates::C
end
=#

# TODO:
# - add interval domain
# - move order out of Space?
# - 
# 

"""
    FiniteDifferenceSpace

Column coordinates containing a collocated
grid at cell centers (`cent`) and cell faces
(`face`).
"""
struct FiniteDifferenceSpace{S, F, C, ICF, ∇CF, ∇FC, ∇FF, ∇CC, O} <:
       AbstractSpace
    face::F
    cent::C
    interp_cent_to_face::ICF
    ∇_cent_to_face::∇CF
    ∇_face_to_cent::∇FC
    ∇_face_to_face::∇FF
    ∇_cent_to_cent::∇CC
    n_cells_ghost::Int
    order::O
end


function FiniteDifferenceSpace{S}(
    face::F,
    cent::C,
    interp_cent_to_face::ICF,
    ∇_cent_to_face::∇CF,
    ∇_face_to_cent::∇FC,
    ∇_face_to_face::∇FF,
    ∇_cent_to_cent::∇CC,
    n_cells_ghost::Integer,
    order::O,
) where {S, F, C, ICF, ∇CF, ∇FC, ∇FF, ∇CC, O}
    FiniteDifferenceSpace{S, F, C, ICF, ∇CF, ∇FC, ∇FF, ∇CC, O}(
        face,
        cent,
        interp_cent_to_face,
        ∇_cent_to_face,
        ∇_face_to_cent,
        ∇_face_to_face,
        ∇_cent_to_cent,
        n_cells_ghost,
        order,
    )
end

const CenterFiniteDifferenceSpace = FiniteDifferenceSpace{CellCent}
const FaceFiniteDifferenceSpace = FiniteDifferenceSpace{CellFace}

#=
function Base.show(io::IO, col_mesh::FiniteDifferenceSpace)
    println(io, col_mesh)
    #=
    println(io, "----- face")
    show(io, col_mesh.face)
    println(io, "----- cent")
    show(io, col_mesh.cent)
    println(io, "----- interp_cent_to_face")
    show(io, col_mesh.interp_cent_to_face)
    println(io, "----- ∇_cent_to_face")
    show(io, col_mesh.∇_cent_to_face)
    println(io, "----- ∇_face_to_cent")
    show(io, col_mesh.∇_face_to_cent)
    println(io, "----- ∇_face_to_face")
    show(io, col_mesh.∇_face_to_face)
    =#
end
=#

coords(c::FiniteDifferenceSpace, ::CellCent) = c.cent
coords(c::FiniteDifferenceSpace, ::CellFace) = c.face

coordinates(c::FiniteDifferenceSpace, ::CellCent) = c.cent.h
coordinates(c::FiniteDifferenceSpace, ::CellFace) = c.face.h
Δcoordinates(c::FiniteDifferenceSpace, ::CellCent) = c.cent.Δh
Δcoordinates(c::FiniteDifferenceSpace, ::CellFace) = c.face.Δh

Base.length(c::FiniteDifferenceSpace, ::CellCent) = length(c.cent)
Base.length(c::FiniteDifferenceSpace, ::CellFace) = length(c.face)

undertype(mesh::FiniteDifferenceSpace) = eltype(mesh.cent.h)
n_cells(col_mesh::FiniteDifferenceSpace) = length(col_mesh.cent)
n_faces(col_mesh::FiniteDifferenceSpace) = length(col_mesh.face)
n_cells_ghost(cm::FiniteDifferenceSpace) = cm.n_cells_ghost

abstract type ColumnMinMax end

"""
    ColumnMin

A type for dispatching on the minimum of the vertical domain.
"""
struct ColumnMin <: ColumnMinMax end

"""
    ColumnMax

A type for dispatching on the maximum of the vertical domain.
"""
struct ColumnMax <: ColumnMinMax end

"""
    n_hat(::ColumnMinMax)

The outward normal vector to the boundary
"""
n_hat(::ColumnMin) = -1
n_hat(::ColumnMax) = 1

"""
    binary(::ColumnMinMax)

Returns 0 for `ColumnMin` and 1 for `ColumnMax`
"""
binary(::ColumnMin) = 0
binary(::ColumnMax) = 1

function interior_face_range(cm::FiniteDifferenceSpace)
    nfaces = n_faces(cm)
    nghost = cm.n_cells_ghost
    return range(nghost + 1, nfaces - nghost, step = 1)
end

function interior_cent_range(cm::FiniteDifferenceSpace)
    ncells = n_cells(cm)
    nghost = cm.n_cells_ghost
    return range(nghost + 1, ncells - nghost, step = 1)
end

boundary_index(cm::FiniteDifferenceSpace, ::CellFace, ::ColumnMin) =
    n_cells_ghost(cm)
boundary_index(cm::FiniteDifferenceSpace, ::CellFace, ::ColumnMax) =
    n_cells(cm) - n_cells_ghost(cm)

# TODO: this assumes stencils of order 2
ghost_index(cm::FiniteDifferenceSpace, ::CellCent, ::ColumnMin) = 1
ghost_index(cm::FiniteDifferenceSpace, ::CellCent, ::ColumnMax) = n_cells(cm)
ghost_index(cm::FiniteDifferenceSpace, ::CellFace, ::ColumnMin) = 1
ghost_index(cm::FiniteDifferenceSpace, ::CellFace, ::ColumnMax) = n_faces(cm)

# TODO: return range for higher order stencil
interior_index(cm::FiniteDifferenceSpace, ::CellCent, ::ColumnMin) =
    1 + n_cells_ghost(cm)
interior_index(cm::FiniteDifferenceSpace, ::CellCent, ::ColumnMax) =
    n_cells(cm) - n_cells_ghost(cm)
interior_index(cm::FiniteDifferenceSpace, ::CellFace, ::ColumnMin) =
    2 + n_cells_ghost(cm)
interior_index(cm::FiniteDifferenceSpace, ::CellFace, ::ColumnMax) =
    n_cells(cm) - n_cells_ghost(cm)

Δcoordinates(c::FiniteDifferenceSpace, ::CellCent, ::ColumnMin) =
    first(c.cent.Δh)
Δcoordinates(c::FiniteDifferenceSpace, ::CellCent, ::ColumnMax) =
    last(c.cent.Δh)
Δcoordinates(c::FiniteDifferenceSpace, ::CellFace, ::ColumnMin) =
    first(c.face.Δh)
Δcoordinates(c::FiniteDifferenceSpace, ::CellFace, ::ColumnMax) =
    last(c.face.Δh)

""" A super-type for dispatching on parts of the domain """
abstract type DomainDecomp end
struct All <: DomainDecomp end
struct Boundary <: DomainDecomp end
struct Interior <: DomainDecomp end
struct Ghost <: DomainDecomp end

function pad!(h, n_cells_ghost::Int) # TODO: use ghost cells
    pushfirst!(h, h[1] - (h[2] - h[1]))
    push!(h, h[end] + (h[end] - h[end - 1]))
end


function FiniteDifferenceSpace{S}(
    h_min::FT,
    h_max::FT,
    n_cells_real::Int;
    n_cells_ghost::Int = 1,
    kwargs...,
) where {S <: Staggering, FT}
    h_face = map(1:(n_cells_real + 1)) do i
        h_min + FT(i - 1) * (h_max - h_min) / FT(n_cells_real)
    end
    h_face = collect(h_face) # TODO: use ArrayType
    pad!(h_face, n_cells_ghost)
    return FiniteDifferenceSpace{S}(
        h_face;
        n_cells_ghost = n_cells_ghost,
        kwargs...,
    )
end

function FiniteDifferenceSpace{S}(
    h_face;
    order = OrderOfAccuracy{2}(),
    n_cells_ghost::Int = 1,
    kwargs...,
) where {S <: Staggering}
    n_cells_real = (length(h_face) - 1) - (2 * n_cells_ghost)
    @assert n_cells_real ≥ 1
    n_faces = n_cells_real + 3 # TODO: use n_cells_ghost
    n_cells = n_cells_real + 2 # TODO: use n_cells_ghost

    @assert length(h_face) == n_cells_real + 3 # TODO: use n_cells_ghost
    Δh_face = map(i -> h_face[i + 1] - h_face[i], 1:(n_faces - 1))
    coords_face =
        CollocatedCoordinates{CellFace, typeof(h_face)}(h_face, Δh_face)

    # Construct cell centers (mid-points between faces):
    h_cent = map(i -> h_face[i] + Δh_face[i] / 2, 1:n_cells)
    @assert length(h_cent) == n_cells_real + 2 # TODO: use n_cells_ghost
    Δh_cent = map(i -> h_cent[i + 1] - h_cent[i], 1:(n_cells - 1))
    coords_cent =
        CollocatedCoordinates{CellCent, typeof(h_cent)}(h_cent, Δh_cent)

    if !(all(isfinite.(h_face)) && all(isfinite.(h_cent)))
        error("FiniteDifferenceSpace is not finite.")
    end

    # Construct interpolation and derivative operators:
    interp_cent_to_face =
        interp_cent_to_face_operator(order, h_face, h_cent, n_cells_ghost)
    ∇_cent_to_face = ∇_cent_to_face_operator(order, Δh_cent, n_cells_ghost)
    ∇_face_to_cent = ∇_face_to_cent_operator(order, Δh_face, n_cells_ghost)
    ∇_face_to_face = ∇_face_to_face_operator(order, Δh_face, n_cells_ghost)
    ∇_cent_to_cent = ∇_cent_to_cent_operator(order, Δh_cent, n_cells_ghost)
    ∇²_cent_to_cent = ∇²_cent_to_cent_operator(order, Δh_cent, n_cells_ghost)
    ∇²_face_to_face = ∇²_face_to_face_operator(order, Δh_face, n_cells_ghost)

    return FiniteDifferenceSpace{S}(
        coords_face,
        coords_cent,
        interp_cent_to_face,
        ∇_cent_to_face,
        ∇_face_to_cent,
        ∇_face_to_face,
        ∇_cent_to_cent,
        n_cells_ghost,
        order,
    )
end


warp_mesh(cm::FiniteDifferenceSpace) = cm

function warp_mesh(
    warp_fn::Function,
    cm::FiniteDifferenceSpace{S},
) where {S <: Staggering}
    cell_face_coords = getproperty(coords(cm, CellFace()), :h)
    warped_cell_face_coords = warp_fn.(cell_face_coords)
    return FiniteDifferenceSpace{S}(
        warped_cell_face_coords;
        n_cells_ghost = cm.n_cells_ghost,
        order = cm.order,
    )
end

#####
##### Order of accuracy type
#####

struct OrderOfAccuracy{N} end

#####
##### Interpolation
#####

"""
    interp_cent_to_face_operator(::OrderOfAccuracy{2}, h_face::AbstractVector, h_cent::AbstractVector)

A second order-accurate interpolation operators, used
to operate on fields that live on cell centers.
The interpolation of these fields live on cell faces.
"""
function interp_cent_to_face_operator(
    ::OrderOfAccuracy{2},
    h_face::AbstractVector,
    h_cent::AbstractVector,
    n_cells_ghost::Int,
)
    n_cells = length(h_cent)
    FT = eltype(h_cent)
    n_rows = n_cells
    sparse_mat = map(1:n_rows) do j
        i = j - 1
        if j == 1 || j == n_rows # padded ghost rows
            diag, upper_diag = FT(0), FT(0)
        else
            diag = (h_face[i + 1] - h_cent[i]) / (h_cent[i + 1] - h_cent[i])
            upper_diag = 1 - diag
        end
        local_operator = SVector(diag, upper_diag)
        hi, lo = 1, 0
        LocalOperator{CentToFace, hi, lo, typeof(local_operator)}(
            local_operator,
        )
    end
    n_cols = n_rows - 1
    return SparseMatrix{n_rows, n_cols}(sparse_mat)
end

#####
##### Derivative operators
#####

"""
    ∇_cent_to_face_operator(::OrderOfAccuracy{2}, Δh_cent::AbstractVector)

The first derivative, using a second order-accurate
finite difference operator. Operates on fields that
live on cell centers. The gradient of these fields
live on cell faces.
"""
function ∇_cent_to_face_operator(
    ::OrderOfAccuracy{2},
    Δh_cent::AbstractVector,
    n_cells_ghost::Int,
)
    n_cells = length(Δh_cent) + 1
    FT = eltype(Δh_cent)
    n_rows = n_cells - 1 + 2 * n_cells_ghost
    sparse_mat = map(1:n_rows) do j
        i = j - n_cells_ghost
        if j == 1 || j == n_rows # padded ghost rows
            diag, upper_diag = FT(0), FT(0)
        else
            diag = -1 / Δh_cent[i]
            upper_diag = 1 / Δh_cent[i]
        end
        local_operator = SVector(diag, upper_diag)
        hi, lo = 1, 0
        LocalOperator{CentToFace, hi, lo, typeof(local_operator)}(
            local_operator,
        )
    end
    n_cols = n_rows - 1
    return SparseMatrix{n_rows, n_cols}(sparse_mat)
end

"""
    ∇_face_to_cent_operator(::OrderOfAccuracy{2}, Δh_face::AbstractVector)

The first derivative, using a second order-accurate
finite difference operator. Operates on fields that
live on cell faces. The gradient of these fields live
on cell centers.
"""
function ∇_face_to_cent_operator(
    ::OrderOfAccuracy{2},
    Δh_face::AbstractVector,
    n_cells_ghost::Int,
)
    n_faces = length(Δh_face) + 1
    FT = eltype(Δh_face)
    n_rows = n_faces - 1 + 2 * n_cells_ghost
    sparse_mat = map(1:n_rows) do j
        i = j - n_cells_ghost
        if j == 1 || j == n_rows # padded ghost rows
            diag, upper_diag = FT(0), FT(0)
        else
            diag = -1 / Δh_face[i]
            upper_diag = 1 / Δh_face[i]
        end
        hi, lo = 1, 0
        local_operator = SVector(diag, upper_diag)
        LocalOperator{FaceToCent, hi, lo, typeof(local_operator)}(
            local_operator,
        )
    end
    n_cols = n_rows + 1
    return SparseMatrix{n_rows, n_cols}(sparse_mat)
end

"""
    ∇_face_to_face_operator(::OrderOfAccuracy{2}, Δh_face::AbstractVector)

The first derivative, using a second order-accurate
finite difference operator. Operates on fields that
live on cell faces. The gradient of these fields live
on cell faces.
"""
function ∇_face_to_face_operator(
    ::OrderOfAccuracy{2},
    Δh_face::AbstractVector,
    n_cells_ghost::Int,
)
    n_cells = length(Δh_face)
    FT = eltype(Δh_face)
    n_rows = n_cells - 1 + 2 * n_cells_ghost
    sparse_mat = map(1:n_rows) do j
        i = j - n_cells_ghost + 1
        if j == 1 || j == n_rows # padded ghost rows
            lower_diag, diag, upper_diag = FT(0), FT(0), FT(0)
        else
            lower_diag =
                -Δh_face[i] / (Δh_face[i - 1] * (Δh_face[i - 1] + Δh_face[i]))
            diag =
                (-Δh_face[i - 1] + Δh_face[i]) / (Δh_face[i - 1] * Δh_face[i])
            upper_diag =
                Δh_face[i - 1] / (Δh_face[i] * (Δh_face[i - 1] + Δh_face[i]))
        end
        hi, lo = 1, 1
        local_operator = SVector(lower_diag, diag, upper_diag)
        LocalOperator{FaceToFace, hi, lo, typeof(local_operator)}(
            local_operator,
        )
    end
    return SparseMatrix{n_rows, n_rows}(sparse_mat)
end

"""
    ∇_cent_to_cent_operator(::OrderOfAccuracy{2}, Δh_cent::AbstractVector)

The first derivative, using a second order-accurate
finite difference operator. Operates on fields that
live on cell centers. The gradient of these fields live
on cell centers.
"""
function ∇_cent_to_cent_operator(
    ::OrderOfAccuracy{2},
    Δh_cent::AbstractVector,
    n_cells_ghost::Int,
)
    n_cells = length(Δh_cent) + 1
    FT = eltype(Δh_cent)
    n_rows = n_cells
    sparse_mat = map(1:n_rows) do j
        i = j - n_cells_ghost + 1
        if j == 1 || j == n_rows # padded ghost rows
            lower_diag, diag, upper_diag = FT(0), FT(0), FT(0)
        else
            lower_diag =
                -Δh_cent[i] / (Δh_cent[i - 1] * (Δh_cent[i - 1] + Δh_cent[i]))
            diag =
                (-Δh_cent[i - 1] + Δh_cent[i]) / (Δh_cent[i - 1] * Δh_cent[i])
            upper_diag =
                Δh_cent[i - 1] / (Δh_cent[i] * (Δh_cent[i - 1] + Δh_cent[i]))
        end
        hi, lo = 1, 1
        local_operator = SVector(lower_diag, diag, upper_diag)
        LocalOperator{FaceToFace, hi, lo, typeof(local_operator)}(
            local_operator,
        )
    end
    return SparseMatrix{n_rows, n_rows}(sparse_mat)
end

#=

TODO: start sketching out ideas for fusing operations

# A is on a center mesh with halo 1
# v is on a face mesh with halo 0
Face(A) # interpolate to face mesh with halo 0
v*Face(A) # elementwise multiplication (face mesh halo 0)
Gradient(v*Face(A)) # center mesh with halo 0
map(Gradient(v*Face(A)), column) # gradient on to CenterFiniteDifferenceSpace


#         center  face
# ----             -2
#          -1
# ----             -1
# Halo      0
# ====              0
#           1
# ----
#           2
# ....
#           n
# ====              n    
# Halo     n+1
# ----             n+1


#         center  face
# ----              1
#           1
# ----              h
# Halo      h
# ====             h+1
#          h+1
# ----
#           4
# ....
#          n+h
# ====            n+h+1   
# Halo    n+h+1
# ----            n+h+2

for i = 1:n
    face_a1 = (A[i-1] + A[i]) / 2
    face_a2 = (A[i] + A[i+1]) / 2
    va1 = v*face_a1
    va2 = v*face_a2
    (va2 - va1) / Δh
end


diff(field)[i] = (field[i] - field[i-1]) / deltaH
=#

#####
##### 2nd Dderivative operators
#####

"""
    ∇²_cent_to_cent_operator(::OrderOfAccuracy{2}, Δh_cent::AbstractVector, n_cells_ghost)


The second derivative, using a second order-accurate
finite difference operator. Operates on fields that
live on cell centers. The gradient of these fields live
on cell centers.
"""
function ∇²_cent_to_cent_operator(
    ::OrderOfAccuracy{2},
    Δh_cent::AbstractVector,
    n_cells_ghost::Int,
)
    n_cells = length(Δh_cent) + 1
    FT = eltype(Δh_cent)
    n_rows = n_cells
    sparse_mat = map(1:n_rows) do j
        i = j
        if j == 1
            lower_diag, diag, upper_diag = FT(0), FT(0), FT(0)
        elseif j == n_rows
            lower_diag, diag, upper_diag = FT(0), FT(0), FT(0)
        else
            lower_diag = 2 / (Δh_cent[i - 1] * (Δh_cent[i - 1] + Δh_cent[i]))
            diag = -2 / (Δh_cent[i - 1] * Δh_cent[i])
            upper_diag = 2 / (Δh_cent[i] * (Δh_cent[i - 1] + Δh_cent[i]))
        end
        local_operator = SVector(lower_diag, diag, upper_diag)
        hi, lo = 1, 1
        LocalOperator{Collocated, hi, lo, typeof(local_operator)}(
            local_operator,
        )
    end
    return SparseMatrix{n_rows, n_rows}(sparse_mat)
end

"""
    ∇²_face_to_face_operator(::OrderOfAccuracy{2}, Δh_face::AbstractVector, n_cells_ghost)

The second derivative, using a second order-accurate
finite difference operator. Operates on fields that
live on cell faces. The gradient of these fields live
on cell faces.

"""
function ∇²_face_to_face_operator(
    ::OrderOfAccuracy{2},
    Δh_face::AbstractVector,
    n_cells_ghost::Int,
)
    n_faces = length(Δh_face) + 1
    FT = eltype(Δh_face)
    n_rows = n_faces
    sparse_mat = map(1:n_rows) do j
        i = j
        if j == 1
            lower_diag, diag, upper_diag = FT(0), FT(0), FT(0)
        elseif j == n_rows
            lower_diag, diag, upper_diag = FT(0), FT(0), FT(0)
        else
            lower_diag = 2 / (Δh_face[i - 1] * (Δh_face[i - 1] + Δh_face[i]))
            diag = -2 / (Δh_face[i - 1] * Δh_face[i])
            upper_diag = 2 / (Δh_face[i] * (Δh_face[i - 1] + Δh_face[i]))
        end
        # Use interior operator everywhere (including boundaries)
        local_operator = SVector(lower_diag, diag, upper_diag)
        hi, lo = 1, 1
        LocalOperator{Collocated, hi, lo, typeof(local_operator)}(
            local_operator,
        )
    end
    return SparseMatrix{n_rows, n_rows}(sparse_mat)
end

#####
##### Space iterator
#####

# Ideally these iteration might return ranges of
# the stencil size, but then they wouldn't be useful
# if we want custom stencils for certain terms. So,
# we're just returning the center of the stencil for now.

# Iterate over cell centers
struct CellCenterIterator{CM}
    col_mesh::CM
end
column(col_mesh::FiniteDifferenceSpace, ::CellCent) =
    CellCenterIterator(col_mesh)
Base.length(cciter::CellCenterIterator) = n_cells(cciter.col_mesh)
Base.eltype(::Type{CellCenterIterator}) = Int

function Base.iterate(cciter::CellCenterIterator, state::Int = 1)
    state > length(cciter) && return nothing
    i_cent = state
    (i_cent, state + 1)
end

# Iterate over interior cell faces
struct CellFaceIterator{CM}
    col_mesh::CM
end
column(col_mesh::FiniteDifferenceSpace, ::CellFace) = CellFaceIterator(col_mesh)
Base.length(cfiter::CellFaceIterator) = n_faces(cfiter.col_mesh)
Base.eltype(::Type{CellFaceIterator}) = Int

function Base.iterate(cciter::CellFaceIterator, state::Int = 1)
    state > length(cciter) && return nothing
    i_face = state
    (i_face, state + 1)
end
