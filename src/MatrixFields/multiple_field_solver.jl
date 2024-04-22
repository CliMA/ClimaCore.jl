# TODO: Can different A's be different matrix styles?
#       if so, how can we handle fuse/parallelize?

# First, dispatch based on the first x and the device:
function multiple_field_solve!(cache, x, A, b)
    name1 = first(matrix_row_keys(keys(A)))
    x1 = x[name1]
    multiple_field_solve!(ClimaComms.device(axes(x1)), cache, x, A, b, x1)
end

# TODO: fuse/parallelize
function multiple_field_solve!(
    ::ClimaComms.AbstractCPUDevice,
    cache,
    x,
    A,
    b,
    x1,
)
    foreach(matrix_row_keys(keys(A))) do name
        single_field_solve!(cache[name], x[name], A[name, name], b[name])
    end
end
