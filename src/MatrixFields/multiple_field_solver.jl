function multiple_field_solve!(cache, x, A, b)
    x1 = first(values(x))
    x_bc = FieldNameDict(keys(x), unrolled_map(Base.broadcastable, values(x)))
    b_bc = FieldNameDict(keys(b), unrolled_map(Base.broadcastable, values(b)))
    multiple_field_solve!(ClimaComms.device(axes(x1)), cache, x_bc, A, b_bc)
end

# TODO: fuse/parallelize
multiple_field_solve!(::ClimaComms.AbstractCPUDevice, cache, x, A, b) =
    foreach(matrix_row_keys(keys(A))) do name
        single_field_solve!(cache[name], x[name], A[name, name], b[name])
    end
