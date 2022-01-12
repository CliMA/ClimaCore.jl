if !("." in LOAD_PATH)
    push!(LOAD_PATH, ".")
end
import Coverage
import Plots

# Track allocations in ClimaCore.jl plus all _direct_ dependencies
exhaustive = "exhaustive=true" in ARGS
@show exhaustive

# Packages to monitor
import ClimaCore
import Adapt
import BlockArrays
import CUDA
import CubedSphere
import DiffEqBase
import DocStringExtensions
import ForwardDiff
import GaussQuadrature
import IntervalSets
import LinearAlgebra
import RecursiveArrayTools
import Rotations
import SparseArrays
import Static
import StaticArrays
import UnPack

mod_dir(x) = dirname(dirname(pathof(x)))
deps_to_monitor = [
    mod_dir(Adapt),
    mod_dir(BlockArrays),
    mod_dir(CUDA),
    mod_dir(CubedSphere),
    mod_dir(DiffEqBase),
    mod_dir(DocStringExtensions),
    mod_dir(ForwardDiff),
    mod_dir(GaussQuadrature),
    mod_dir(IntervalSets),
    mod_dir(LinearAlgebra),
    mod_dir(RecursiveArrayTools),
    mod_dir(Rotations),
    mod_dir(SparseArrays),
    mod_dir(Static),
    mod_dir(StaticArrays),
    mod_dir(UnPack),
]
deps_to_monitor = exhaustive ? deps_to_monitor : ()
pkg_dir = mod_dir(ClimaCore)

julia_base_dir = "/central/software/julia/1.7.0/share/julia/base/"
julia_base_dir = isdir(julia_base_dir) ? (julia_base_dir,) : ()
all_dirs_to_monitor =
    [julia_base_dir..., mod_dir(ClimaCore), deps_to_monitor...]

# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"

# (filename, ARGs passed to script)
all_cases = [
    (joinpath(pkg_dir, "examples", "hybrid", "bubble_2d.jl"), ""),
    (joinpath(pkg_dir, "examples", "hybrid", "bubble_3d.jl"), ""),
    (
        joinpath(pkg_dir, "examples", "3dsphere", "baroclinic_wave.jl"),
        "baroclinic_wave",
    ),
    (
        joinpath(pkg_dir, "examples", "sphere", "shallow_water.jl"),
        "barotropic_instability",
    ),
]

# only one case for exhaustive alloc analysis
all_cases = if exhaustive
    [(joinpath(pkg_dir, "examples", "hybrid", "bubble_2d.jl"), "")]
else
    all_cases
end

allocs = Dict()
for (case, args) in all_cases
    ENV["ALLOCATION_CASE_NAME"] = case
    if exhaustive
        # run(`julia --project=examples/ --track-allocation=all perf/alloc_per_case_with_init_io.jl`)
        error("Not yet supported")
    else
        run(
            `julia --project=examples/ --track-allocation=all perf/allocs_per_case.jl $args`,
        )
    end

    allocs[case] = Coverage.analyze_malloc(all_dirs_to_monitor)

    # Clean up files
    for d in all_dirs_to_monitor
        all_files = [
            joinpath(root, f) for
            (root, dirs, files) in Base.Filesystem.walkdir(d) for f in files
        ]
        all_mem_files = filter(x -> endswith(x, ".mem"), all_files)
        for f in all_mem_files
            rm(f)
        end
    end
end

@info "Post-processing allocations"

const PKG_NAME = "ClimaCore.jl"

function plot_allocs(
    folder,
    case_name,
    allocs_per_case,
    script_args,
    n_unique_bytes,
)
    p = Plots.plot()
    case_name_args =
        script_args == "" ? case_name : case_name * "_$(script_args)"
    @info "Allocations for $case_name_args"

    function filename_only(fn)
        if occursin(".jl", fn)
            fn = join(split(fn, ".jl")[1:(end - 1)], ".jl") * ".jl"
        end
        fn = replace(fn, "\\" => "/") # for windows...
        splitby = PKG_NAME
        if occursin(splitby, fn)
            fn = PKG_NAME * last(split(fn, splitby))
        end
        splitby = "climacore-ci/"
        if occursin(splitby, fn)
            fn = PKG_NAME * last(split(fn, splitby))
        end
        return fn
    end
    function compile_pkg(fn, linenumber)
        c1 = endswith(filename_only(fn), PKG_NAME)
        c2 = linenumber == 1
        return c1 && c2
    end

    filter!(x -> x.bytes ≠ 0, allocs_per_case)
    filter!(x -> !compile_pkg(x.filename, x.linenumber), allocs_per_case)

    for alloc in allocs_per_case
        println(alloc)
    end
    println("Number of allocating sites: $(length(allocs_per_case))")
    case_bytes = getproperty.(allocs_per_case, :bytes)[end:-1:1]
    case_filename = getproperty.(allocs_per_case, :filename)[end:-1:1]
    case_linenumber = getproperty.(allocs_per_case, :linenumber)[end:-1:1]
    all_bytes = Int[]
    filenames = String[]
    linenumbers = Int[]
    loc_ids = String[]
    for (bytes, filename, linenumber) in
        zip(case_bytes, case_filename, case_linenumber)
        compile_pkg(filename, linenumber) && continue # Skip loading module
        loc_id = "$(filename_only(filename))" * "$linenumber"
        if !(bytes in all_bytes) && !(loc_id in loc_ids)
            push!(all_bytes, bytes)
            push!(filenames, filename)
            push!(linenumbers, linenumber)
            push!(loc_ids, loc_id)
            if length(all_bytes) ≥ n_unique_bytes
                break
            end
        end
    end

    all_bytes = all_bytes ./ 10^3
    max_bytes = maximum(all_bytes)
    @info "$case_name_args: $all_bytes"
    xtick_name(filename, linenumber) = "$filename, line number: $linenumber"
    markershape = (:square, :hexagon, :circle, :star, :utriangle, :dtriangle)
    for (bytes, filename, linenumber) in zip(all_bytes, filenames, linenumbers)
        label = xtick_name(filename_only(filename), linenumber)
        Plots.plot!(
            [0],
            [bytes];
            seriestype = :scatter,
            label = label,
            markershape = markershape[1],
            markersize = 1 + bytes / max_bytes * 10,
        )
        markershape = (markershape[end], markershape[1:(end - 1)]...)
    end
    p1 = Plots.plot!(
        ylabel = "Allocations (KB)",
        title = case_name_args,
        legendfontsize = 6,
    )
    n_subset = min(length(allocs_per_case) - 1, 100)
    subset_allocs_per_case = allocs_per_case[end:-1:(end - n_subset)]
    p2 = Plots.plot(
        1:length(subset_allocs_per_case),
        getproperty.(subset_allocs_per_case, :bytes) ./ 1000;
        xlabel = "i-th allocating line (truncated and sorted)",
        ylabel = "Allocations (KB)",
        markershape = :circle,
    )
    Plots.plot(p1, p2, layout = Plots.grid(2, 1))
    Plots.savefig(joinpath(folder, "allocations_$case_name_args.png"))
end

folder =
    exhaustive ? "perf/allocations_output_exhaustive" :
    "perf/allocations_output"
mkpath(folder)

@info "Allocated bytes for single tendency per case:"
for (case, args) in all_cases
    plot_allocs(
        folder,
        first(split(basename(case), ".jl")),
        allocs[case],
        args,
        10,
    )
end
