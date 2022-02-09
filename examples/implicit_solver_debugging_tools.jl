using ForwardDiff: Dual
using SparseArrays: spdiagm

using ClimaCore: Spaces, Operators

get_var(obj, ::Tuple{}) = obj
get_var(obj, tup::Tuple) = get_var(getproperty(obj, tup[1]), Base.tail(tup))
function exact_column_jacobian(rhs_implicit!, Y, p, t, i, j, h, Y_name, Yₜ_name)
    T = eltype(Y)
    Y_var = get_var(Y, Y_name)
    Y_var_vert_space = Spaces.column(axes(Y_var), i, j, h)
    bot_level = Operators.left_idx(Y_var_vert_space)
    top_level = Operators.right_idx(Y_var_vert_space)
    partials = ntuple(_ -> zero(T), top_level - bot_level + 1)
    Yᴰ = Dual.(Y, partials...)
    Yᴰ_var = get_var(Yᴰ, Y_name)
    ith_ε(i) = Dual.(zero(T), Base.setindex(partials, one(T), i)...)
    set_level_εs!(level) =
        parent(Spaces.level(Yᴰ_var, level)) .+= ith_ε(level - bot_level + 1)
    foreach(set_level_εs!, bot_level:top_level)
    Yₜᴰ = similar(Yᴰ)
    rhs_implicit!(Yₜᴰ, Yᴰ, p, t)
    col = Spaces.column(get_var(Yₜᴰ, Yₜ_name), i, j, h)
    return vcat(map(dual -> [dual.partials.values...]', parent(col))...)
end

# TODO: This only works for scalar stencils.
function column_matrix(stencil, i, j, h)
    column_stencil = Spaces.column(stencil, i, j, h)

    space = axes(column_stencil)
    n_rows = Spaces.nlevels(space)

    lbw, ubw = Operators.bandwidths(eltype(column_stencil))
    n_diags = ubw - lbw + 1

    # We can only infer the the argument space's axes from NaNs in the stencil.
    loc = Operators.Interior()
    isnt_left_boundary_row(i_row) =
        !isnan(Operators.getidx(column_stencil, loc, i_row)[1])
    isnt_right_boundary_row(i_row) =
        !isnan(Operators.getidx(column_stencil, loc, i_row)[n_diags])
    indices = Operators.left_idx(space):Operators.right_idx(space)
    i_first_interior_row = findfirst(isnt_left_boundary_row, indices)
    i_last_interior_row = findlast(isnt_right_boundary_row, indices)

    n_cols = i_last_interior_row - i_first_interior_row + n_diags

    function diag_key_value(i_diag)
        start_index = max(i_first_interior_row - (i_diag - 1), 1)
        end_index = min(i_last_interior_row + (n_diags - i_diag), n_rows)
        array = parent(getproperty(column_stencil.coefs, i_diag))
        @assert length(array) == size(array, 1)
        return (i_diag - i_first_interior_row) =>
            view(array, start_index:end_index)
    end

    return spdiagm(n_rows, n_cols, ntuple(diag_key_value, n_diags)...)
end

function column_vector(arg, i, j, h)
    array = parent(Spaces.column(arg, i, j, h))
    @assert length(array) == size(array, 1)
    return array
end
