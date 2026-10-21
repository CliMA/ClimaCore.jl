# The values that a thread solves for in a slice loop can be read by other threads in later
# statements, so they are synchronized once every field is solved.
function multiple_field_solve!(cache, x, A, b)
    foreach(matrix_row_keys(keys(A))) do name
        single_field_solve!(cache[name], x[name], A[name, name], b[name])
    end
    synchronize_in_loop(x)
end
