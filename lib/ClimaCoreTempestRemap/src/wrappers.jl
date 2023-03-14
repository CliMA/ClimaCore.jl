"""
    rll_mesh(filename::AbstractString; nlat=90, nlon = round(Int, nlat * 1.6); verbose=false)

Create a regular latitude-longitude (RLL) mesh and write it to `filename` in
Exodus format. `nlat` is the number of latitudinal cells, and `nlon` is the
number of longitudinal cells.

Set `verbose=true` to print information.

See [Tempest remap: mesh generation](https://github.com/ClimateGlobalChange/tempestremap/#mesh-generation)
"""
function rll_mesh(
    filename::String;
    verbose = false,
    nlat = 90,
    nlon = round(Int, nlat * 1.6),
)
    buf = IOBuffer()
    if !success(pipeline(
        ```
$(TempestRemap_jll.GenerateRLLMesh_exe())
--lon $nlon
--lat $nlat
--file $filename
```,
        stdout = buf,
    ))
        output = String(take!(buf))
        error("GenerateRLLMesh failure\n$output")
    end
    if verbose
        output = String(take!(buf))
        print(output)
    end
end

"""
    overlap_mesh(outfile::AbstractString, meshfile_a::AbstractString, meshfile_b::AbstractString; verbose=false)

Create the overlap mesh of `meshfile_a` and `meshfile_b` and write it to `outfile`. All
files should be in Exodus format.

Set `verbose=true` to print information.

See [Tempest remap: mesh generation](https://github.com/ClimateGlobalChange/tempestremap/#mesh-generation)
"""
function overlap_mesh(
    outfile::AbstractString,
    meshfile_a::AbstractString,
    meshfile_b::AbstractString;
    verbose = false,
)
    buf = IOBuffer()
    if !success(
        pipeline(
            ```
            $(TempestRemap_jll.GenerateOverlapMesh_exe())
            --a $meshfile_a
            --b $meshfile_b
            --out $outfile
            ```,
            stdout = buf,
        ),
    )
        output = String(take!(buf))
        error("GenerateRLLMesh failure\n$output")
    end
    if verbose
        output = String(take!(buf))
        print(output)
    end
end

"""
    remap_weights(
        weightfile::AbstractString,
        meshfile_in::AbstractString,
        meshfile_out::AbstractString,
        meshfile_overlap::AbstractString;
        verbose=false,
        kwargs...
    )

Create a file `weightfile` in SCRIP format containing the remapping weights from
`meshfile_in` to `meshfile_out`, where `overlap_meshfile` is constructed via
[`overlap_meshfile(meshfile_overlap, meshfile_in, meshfile_out)`](@ref).

Keyword arguments are passed as command-line options. These include:
- `in_type` / `out_type`: the type of the input and output mesh:
  - `"fv"` (default): finite volume (one value per element)
  - `"cgll"`: continuous GLL finite element method (a single value for colocated nodes)
  - `"dgll"`: discontinuous GLL finite element method (duplicate values for colocated nodes)
- 'in_np'/'out_np': Order of input and output meshes
- 'mono': Monotonicity of remapping

Set `mono = true` for monotone remapping
Set `verbose=true` to print information.

See [Tempest remap: offline map generation](https://github.com/ClimateGlobalChange/tempestremap/#offline-map-generation)
"""
function remap_weights(
    weightfile::AbstractString,
    meshfile_in::AbstractString,
    meshfile_out::AbstractString,
    meshfile_overlap::AbstractString;
    verbose = false,
    kwargs...,
)
    cmd = ```
        $(TempestRemap_jll.GenerateOfflineMap_exe())
        --in_mesh $meshfile_in
        --out_mesh $meshfile_out
        --ov_mesh $meshfile_overlap
        --out_map $weightfile
        ```
    for (k, v) in kwargs
        if typeof(v) == Bool && v
            append!(cmd.exec, [string("--", k)])
        else
            append!(cmd.exec, [string("--", k), string(v)])
        end
    end
    buf = IOBuffer()
    if !success(pipeline(cmd, stdout = buf))
        output = String(take!(buf))
        error("GenerateRLLMesh failure\n$output")
    end
    if verbose
        output = String(take!(buf))
        print(output)
    end

end

"""
    apply_remap(outfile::AbstractString, infile::AbstractString, weightfile::AbstractString, vars; verbose=false)

Remap the NetCDF file `infile` to `outfile`, using the remapping weights
`weightfile` constructed via [`remap_weights`](@ref).
`vars` should be a collection of variable names to remap.

Set `verbose=true` to print information.

See [Tempest remap: offline map application](https://github.com/ClimateGlobalChange/tempestremap/#offline-map-application)
"""
function apply_remap(
    outfile::AbstractString,
    infile::AbstractString,
    weightfile::AbstractString,
    vars;
    verbose = false,
)
    buf = IOBuffer()
    if !success(
        pipeline(
            ```
            $(TempestRemap_jll.ApplyOfflineMap_exe())
            --map $weightfile
            --var $(join(vars,","))
            --in_data $infile
            --out_data $outfile
            ```,
            stdout = buf,
        ),
    )
        output = String(take!(buf))
        error("GenerateRLLMesh failure\n$output")
    end
    if verbose
        output = String(take!(buf))
        print(output)
    end

end
