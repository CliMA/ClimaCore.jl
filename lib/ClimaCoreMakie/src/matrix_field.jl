
# it is expected eltype(eltype(col_entry)) is a scalar
# there are four situations to consider depending on the transforms
# 1. no transform - in this case the plot should be a n_levels rows and band_width columns
# 2. transform along each level - in this case the plot should be a n_levels rows and some n >=1 columns
# 3. transform along each part of band - in this case the plot should be1 row and n_bands columns
# 4. transform along entire part - in this case the plot should be 1 row and 1 column
# TODO: Make work for fieldvectors
# TODO: check how args work by return from filedmatrixplot


function plot_fieldmatrix!(
    grid_position,
    fieldmatrix;
    entry_transform = identity,
    main_colormap = Makie.cgrad(:tol_iridescent, 21; categorical = true),
    main_colorbar_label_text = "Magnitude of matrix entry",
    main_colorbar_ticks = Makie.automatic,
    main_colorbar_label_size = 7,
    underlay_bandwiths = false,
    bandwidth_colormap = :tab10,
    title = "",
    squashed_bands = false,
)
    first_field_index =
        findfirst(x -> x isa ClimaCore.Fields.Field, fieldmatrix.entries)
    FT =
        isnothing(first_field_index) ? Float64 :
        eltype(parent(fieldmatrix.entries[first_field_index]))

    matrix_keys = keys(fieldmatrix)
    row_keys = ClimaCore.MatrixFields.matrix_row_keys(matrix_keys)
    col_keys = ClimaCore.MatrixFields.matrix_col_keys(matrix_keys)
    entries_info = (
        entry isa ClimaCore.MatrixFields.UniformScaling ? (1, 1, 0, 0) :
        ClimaCore.MatrixFields.band_matrix_info(entry) for
        entry in fieldmatrix.entries
    )
    max_row_size = maximum((entry_info[1] for entry_info in entries_info))
    max_col_size = maximum((entry_info[2] for entry_info in entries_info))


    gap_size = 0
    col_names = [name_to_string(name) for name in col_keys]
    col_bounds = [
        range(start = 0, step = max_col_size, length = length(col_keys) + 1)...,
    ]
    col_centers =
        [(col_bounds[i] + col_bounds[i + 1]) ./ 2 for i in 1:length(col_keys)]

    row_names = [name_to_string(name) for name in row_keys]
    row_bounds = [
        range(start = 0, step = max_row_size, length = length(row_keys) + 1)...,
    ]
    row_centers =
        [(row_bounds[i] + row_bounds[i + 1]) ./ 2 for i in 1:length(row_keys)]

    axis_kwargs = (;
        xticks = (col_centers, col_names),
        xticksvisible = false,
        xticklabelrotation = pi / 4,
        title = title,
        xminorticks = col_bounds,
        xminorticksvisible = true,
        xgridvisible = false,
        xminorgridvisible = true,
        xlabel = "Y index",
        ylabel = "Yₜ index",
        yreversed = true,
        yticks = (row_centers, row_names),
        yticksvisible = false,
        yminorticks = row_bounds,
        yminorticksvisible = true,
        ygridvisible = false,
        yminorgridvisible = true,
        # aspect = Makie.DataAspect(),
        xminorgridcolor = :black,
        yminorgridcolor = :black,
    )
    outer_axis = Makie.Axis(grid_position[1:2, 1]; axis_kwargs...)

    combined_entry_matrices = fill(
        NaN,
        (max_col_size) * length(col_keys),
        (max_row_size) * length(row_keys),
    )
    combined_bw_matrices = fill(
        NaN,
        (max_col_size) * length(col_keys),
        (max_row_size) * length(row_keys),
    )

    key_to_submatrices = Dict{
        Tuple{
            <:ClimaCore.MatrixFields.FieldName,
            <:ClimaCore.MatrixFields.FieldName,
        },
        Tuple{<:SubArray, <:SubArray},
    }()
    for (row_index, row_key) in enumerate(row_keys)
        for (col_index, col_key) in enumerate(col_keys)
            key_to_submatrices[(row_key, col_key)] = (
                view(
                    combined_entry_matrices,
                    ((max_row_size) * (row_index - 1) + 1):((max_row_size) * row_index),
                    ((max_col_size) * (col_index - 1) + 1):((max_col_size) * col_index),
                ),
                view(
                    combined_bw_matrices,
                    ((max_row_size) * (row_index - 1) + 1):((max_row_size) * row_index),
                    ((max_col_size) * (col_index - 1) + 1):((max_col_size) * col_index),
                ),
            )
        end
    end
    for key in matrix_keys
        entry = entry_transform(fieldmatrix[key])
        (entry_matrix, bw_matrix) = key_to_submatrices[key]
        if entry isa ClimaCore.MatrixFields.UniformScaling
            entry_matrix[ClimaCore.MatrixFields.band(0)] .= entry.λ
            bw_matrix .= 1
        elseif entry isa ClimaCore.MatrixFields.DiagonalMatrixRow
            @assert entry.entries.:(1) isa Number
            entry_matrix[ClimaCore.MatrixFields.band(0)] .= entry.entries.:(1)
            bw_matrix .= 1
        else
            n_rows, n_cols, matrix_ld, matrix_ud =
                ClimaCore.MatrixFields.band_matrix_info(entry)
            banded_entry = ClimaCore.MatrixFields.column_field2array_view(entry)
            for d in matrix_ld:matrix_ud
                entry_matrix[ClimaCore.MatrixFields.band(d)] .=
                    banded_entry[ClimaCore.MatrixFields.band(d)]
            end
            bw_matrix .= matrix_ud - matrix_ld + 1
        end
    end

    im = Makie.image!(
        outer_axis,
        combined_entry_matrices;
        interpolate = false,
        colormap = main_colormap,
    )
    Makie.translate!(im, 0, 0, -10)

    if underlay_bandwiths
        max_bandwidth =
            Int(maximum(x -> isnan(x) ? 0 : x, combined_bw_matrices))
        bandwidth_colors = Makie.cgrad(
            [
                Makie.RGB(1, 1, 1),
                Makie.cgrad(
                    bandwidth_colormap,
                    max_bandwidth;
                    categorical = true,
                )...,
            ];
            categorical = true,
        )

        bw = Makie.image!(
            outer_axis,
            combined_bw_matrices;
            interpolate = false,
            colormap = bandwidth_colors,
            colorrange = (
                minimum(x -> isnan(x) ? 0 : x, combined_bw_matrices),
                max_bandwidth,
            ),
            alpha = 0.4,
        )


        Makie.translate!(bw, 0, 0, -100)
        Makie.Colorbar(
            grid_position[2, 2],
            bw;
            label = "entry bandwidth",
            # ticks = Makie.automatic,
            labelsize = main_colorbar_label_size,
            ticklabelsize = 8,
        )

    end
    Makie.Colorbar(
        underlay_bandwiths ? grid_position[1, 2] : grid_position[1:2, 2],
        im;
        label = main_colorbar_label_text,
        ticks = main_colorbar_ticks,
        labelsize = main_colorbar_label_size,
        ticklabelsize = 8,
    )
    return
end

# TODO: make or generated funcs for theese ease of use constructors
plot_fieldmatrix_sign!(grid_position, fieldmatrix;) = plot_fieldmatrix!(
    grid_position,
    fieldmatrix;
    entry_transform = apply_sign,
    # main_colormap = Makie.categorical_colors(:RdBu_3, 3),
    main_colormap = Makie.cgrad(:RdBu_3, 2; categorical = true),
    main_colorbar_label_text = "Sign of matrix entry",
    main_colorbar_ticks = ([-0.5, 0.5], Makie.latexstring.(["-", "+"])),
    underlay_bandwiths = true,
    bandwidth_colormap = :tab10,
)


apply_sign(x) = apply_sign.(x)
apply_sign(x::ClimaCore.MatrixFields.UniformScaling) =
    ClimaCore.MatrixFields.UniformScaling(sign(x.λ))
apply_sign(x::ClimaCore.MatrixFields.BandMatrixRow) = map(apply_sign, x)
apply_sign(x::Number) = sign(x)


function name_to_string(
    ::ClimaCore.MatrixFields.FieldName{name_chain},
) where {name_chain}
    quoted_names = map(name -> name isa Integer ? ":($name)" : name, name_chain)
    return join(quoted_names, ".")
end
