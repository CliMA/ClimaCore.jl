multiple_field_solve!(cache, x, A, b) =
    foreach(matrix_row_keys(keys(A))) do name
        single_field_solve!(cache[name], x[name], A[name, name], b[name])
    end
