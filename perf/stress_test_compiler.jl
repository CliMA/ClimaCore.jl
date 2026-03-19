"""
Stress test for the ClimaCore compiler on pointwise/broadcast operations.

This script tests when inlining fails, CUDA performance degrades, or the
compiler fails altogether. The test suite covers:

1. **Arithmetic operations** - varying nesting depth and argument counts
   When do we fail to inline deeply chained operations?

2. **Projection operations** - axis tensor coordinate transformations
   How does projection complexity affect compilation?

3. **Multiple arguments** - varying numbers of nonlocal field arguments
   When does broadcasting with many arguments become problematic?

4. **Function composition** - log, sqrt, and other transcendental functions
   How do special functions interact with inlining?

5. **Divergence operations** - differential operator on vector fields
   How does the divergence operator compilation scale with mesh complexity?

6. **Curl operations** - differential operator on vector fields
   How does the curl operator compilation scale with mesh complexity?

Each test is run in a subprocess to avoid compilation state leakage.
The Julia project is automatically instantiated before running tests.
Uses the `.buildkite` project environment for reproducibility.

USAGE EXAMPLES:
  # Run all tests on CPU (default)
  julia --project=.buildkite perf/stress_test_compiler.jl

  # Run tests matching a filter on CPU
  julia --project=.buildkite perf/stress_test_compiler.jl arithmetic

  # Run on CUDA with GPU reservation
  CLIMACOMMS_DEVICE=CUDA srun --mpi=none --gpus=1 julia --project=.buildkite perf/stress_test_compiler.jl

  # Run a single specific test
  julia --project=.buildkite perf/stress_test_compiler.jl arithmetic_depth_5

  # Run with sbatch
  mkdir -p perf/logs && sbatch --job-name=cc-stress-suite-v2 \
    --output=perf/logs/stress_suite_v2_%j.log --gpus=1 \
    --wrap='cd /home/pbachant/dev/ClimaCore.jl && CLIMACOMMS_DEVICE=CUDA julia --project=.buildkite perf/stress_test_compiler.jl'

NOTE: The script automatically detects and uses the `.buildkite` project directory,
so it should be run from the ClimaCore.jl root or from the perf/ directory.
"""

using Pkg
using BenchmarkTools
using Printf
using Dates
using Logging
using Statistics
using Sockets

# Set up logging to suppress info messages during testing
disable_logging(Logging.Info)

# Determine project root and buildkite directory
const PROJECT_ROOT = dirname(dirname(abspath(PROGRAM_FILE)))
const PROJECT_DIR = joinpath(PROJECT_ROOT, ".buildkite")

# Default to CUDA; can still be overridden explicitly by environment
const DEVICE = get(ENV, "CLIMACOMMS_DEVICE", "CUDA")

"""
    initialize_project()

Initialize the Julia project before running tests.
"""
function initialize_project()
    # Change to project directory
    cd(PROJECT_DIR)

    # Instantiate project if needed
    if !isfile(joinpath(PROJECT_DIR, "Manifest.toml"))
        @info "Instantiating project at $PROJECT_DIR"
        Pkg.instantiate()
    else
        @info "Project manifest found, skipping instantiation"
    end
end

"""
    has_cuda_env()

Check if CUDA device is requested via environment variable.
"""
has_cuda_env() = DEVICE == "CUDA"

"""
    run_test_subprocess(test_code::String, test_name::String) -> (success::Bool, output::String, error::String)

Run a test in a subprocess to avoid compilation state leakage.
Returns a tuple of (success, stdout, stderr).

If already running under srun (detected via SLURM_* environment variables),
subprocesses inherit the parent's GPU allocation and don't request their own.
Otherwise, subprocesses request their own GPU allocation via srun.
"""
function run_test_subprocess(test_code::String, test_name::String)
    tmp_file = tempname() * ".jl"
    try
        write(tmp_file, test_code)

        # Check if we're already in a srun environment
        in_srun = !isempty(get(ENV, "SLURM_JOB_ID", ""))

        # Build command: skip srun if parent is already in srun to avoid resource contention
        if in_srun
            cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_DIR) $tmp_file`
        else
            # Need to allocate GPU for standalone subprocess
            cmd = `srun --mpi=none --gpus=1 $(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_DIR) $tmp_file`
        end

        try
            # Use withenv to set environment variables
            output = withenv(
                "CLIMACOMMS_DEVICE" => DEVICE,
                "CLIMACORE_COLLECT_KERNEL_STATS" => (has_cuda_env() ? "1" : "0"),
            ) do
                read(cmd, String)
            end
            return (true, output, "")
        catch e
            return (false, "", sprint(showerror, e))
        end
    finally
        rm(tmp_file, force = true)
    end
end

"""
    parse_timings_from_output(output::String) -> Dict{String, Float64}

Parse timing information from test output (expected format: "TIMING: name = value_seconds").
"""
function parse_timings_from_output(output::String)
    timings = Dict{String, Float64}()
    for line in split(output, '\n')
        if startswith(line, "TIMING:")
            parts = split(line, '=')
            if length(parts) == 2
                name = strip(split(parts[1], ':')[2])
                value_str = strip(parts[2])
                if endswith(value_str, "s")
                    value_str = value_str[1:(end - 1)]
                end
                try
                    timings[name] = parse(Float64, value_str)
                catch
                end
            end
        end
    end
    return timings
end

"""
    parse_cuda_profile_from_output(output::String) -> Vector{String}

Parse CUDA profile lines from subprocess output.
Expected format: "CUDA_PROFILE: ...".
"""
function parse_cuda_profile_from_output(output::String)
    profiles = String[]
    for line in split(output, '\n')
        if startswith(line, "CUDA_PROFILE:")
            push!(profiles, strip(line))
        end
    end
    return profiles
end

function parse_cuda_profile_metrics(profile::String)
    metrics = Dict{String, String}()
    payload = strip(replace(profile, "CUDA_PROFILE:" => ""))
    for token in split(payload)
        if contains(token, '=')
            key, value = split(token, '='; limit = 2)
            metrics[key] = value
        end
    end
    return metrics
end

function summarize_cuda_profiles(profiles::Vector{String})
    isempty(profiles) && return String[]

    parsed_profiles =
        [(profile, parse_cuda_profile_metrics(profile)) for profile in profiles]

    function metric_int(metrics::Dict{String, String}, key::String)
        try
            return parse(Int, get(metrics, key, "0"))
        catch
            return 0
        end
    end

    primary_profile, primary_metrics = parsed_profiles[1]
    primary_score = (
        metric_int(primary_metrics, "registers"),
        metric_int(primary_metrics, "local"),
        metric_int(primary_metrics, "shared"),
    )
    for candidate in parsed_profiles[2:end]
        _, metrics = candidate
        candidate_score = (
            metric_int(metrics, "registers"),
            metric_int(metrics, "local"),
            metric_int(metrics, "shared"),
        )
        if candidate_score > primary_score
            primary_profile, primary_metrics = candidate
            primary_score = candidate_score
        end
    end

    local_memory_count = count(parsed_profiles) do (_, metrics)
        metric_int(metrics, "local") > 0
    end

    registers = get(primary_metrics, "registers", "?")
    local_bytes = metric_int(primary_metrics, "local")
    shared_bytes = get(primary_metrics, "shared", "?")
    kernel_name = get(primary_metrics, "kernel", "unknown")
    status = local_bytes > 0 ? "local_memory_used" : "no_local_memory"

    return [
        "CUDA_PROFILE: primary_kernel=$(kernel_name) registers=$(registers) local=$(local_bytes) shared=$(shared_bytes) status=$(status) local_memory_kernels=$(local_memory_count)/$(length(parsed_profiles))",
    ]
end

function _command_output(cmd::Cmd)
    try
        return chomp(read(cmd, String))
    catch
        return nothing
    end
end

function _command_lines(cmd::Cmd)
    output = _command_output(cmd)
    isnothing(output) && return String[]
    isempty(output) && return String[]
    return split(output, '\n')
end

function _git_cmd(args...)
    return Cmd(vcat(["git", "-C", PROJECT_ROOT], collect(args)))
end

function collect_run_metadata(test_filter::Union{String, Nothing})
    git_status = _command_lines(_git_cmd("status", "--porcelain"))
    gpu_lines =
        has_cuda_env() ?
        _command_lines(`nvidia-smi --query-gpu=name,uuid --format=csv,noheader`) :
        String[]
    visible_gpu_env = get(ENV, "CUDA_VISIBLE_DEVICES", "")
    allocated_gpu_ids =
        isempty(strip(visible_gpu_env)) ? String[] : split(visible_gpu_env, ',')

    return Dict{String, Any}(
        "timestamp_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "project_root" => PROJECT_ROOT,
        "device" => DEVICE,
        "hostname" => gethostname(),
        "julia_version" => string(VERSION),
        "test_filter" => something(test_filter, "all"),
        "git_commit" =>
            something(_command_output(_git_cmd("rev-parse", "HEAD")), "unknown"),
        "git_branch" =>
            something(
                _command_output(_git_cmd("rev-parse", "--abbrev-ref", "HEAD")),
                "unknown",
            ),
        "git_describe" =>
            something(
                _command_output(_git_cmd("describe", "--always", "--dirty", "--tags")),
                "unknown",
            ),
        "git_dirty" => !isempty(git_status),
        "git_status" => git_status,
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", nothing),
        "slurm_job_name" => get(ENV, "SLURM_JOB_NAME", nothing),
        "slurm_nodelist" => get(ENV, "SLURM_JOB_NODELIST", nothing),
        "allocated_gpu_count" => length(allocated_gpu_ids),
        "allocated_gpu_ids" => allocated_gpu_ids,
        "node_gpu_count" => length(gpu_lines),
        "gpu_devices" => [
            Dict{String, Any}(
                "index" => i,
                "name" => strip(first(split(line, ','))),
                "uuid" => strip(last(split(line, ','))),
            ) for (i, line) in enumerate(gpu_lines)
        ],
    )
end

function arithmetic_expression(depth::Int)
    expr = "x"
    ops = ["+", "*", "/", "-"]
    for i in 1:depth
        op = ops[mod(i, length(ops)) + 1]
        expr = "($expr $op $(i).0)"
    end
    return expr
end

function functions_expression(funcs::Vector{String}, depth::Int)
    expr = "x + 0.5"
    for i in depth:-1:1
        func = funcs[mod1(i, length(funcs))]
        if func == "log"
            expr = "log(abs($expr) + 1.5)"
        elseif func == "sqrt"
            expr = "sqrt(abs($expr) + 1.5)"
        else
            expr = "$func($expr)"
        end
    end
    return expr
end

function render_test_expression(test)
    if test.operation_type == "arithmetic"
        expr = arithmetic_expression(test.complexity)
        return "op(x) = $expr\nop.(f)"
    elseif test.operation_type == "multiarg"
        args = ["f$i" for i in 1:(test.num_args)]
        sum_expr = join(args[1:(end - 1)], " + ")
        op_expr = "($sum_expr) / ($(last(args)) + 1.0)"
        return "op($(join(args, ", "))) = $op_expr\nop.($(join(args, ", ")))"
    elseif test.operation_type == "functions"
        funcs = if occursin("_log_", test.name)
            ["log"]
        elseif occursin("_sqrt_", test.name)
            ["sqrt"]
        else
            ["log", "sqrt", "abs"]
        end
        expr = functions_expression(funcs, test.complexity)
        return "op(x) = $expr\nop.(f)"
    elseif test.operation_type == "projection"
        proj_terms = join(
            [
                "Geometry.project(Geometry.Covariant12Axis(), v)" for
                _ in 1:(test.complexity)
            ],
            " .+ ",
        )
        return "@. $proj_terms"
    elseif test.operation_type == "divergence"
        return join(["div_op.(v .* $(i).0)" for i in 1:(test.complexity)], " .+ ")
    elseif test.operation_type == "curl"
        return join(["curl_op.(v .* $(i).0)" for i in 1:(test.complexity)], " .+ ")
    elseif test.operation_type == "interpolate"
        return join(["interp.(ᶜf .* $(i).0)" for i in 1:(test.complexity)], " .+ ")
    elseif test.operation_type == "weighted_interpolate"
        return join(["winterp.(ᶜw, ᶜf .* $(i).0)" for i in 1:(test.complexity)], " .+ ")
    elseif test.operation_type == "upwinding"
        return join(["upwind.(ᶠv, ᶜf .* $(i).0)" for i in 1:(test.complexity)], " .+ ")
    else
        return test.description
    end
end

function result_profile_metrics(result)
    isempty(result.cuda_profiles) && return Dict{String, String}()
    return parse_cuda_profile_metrics(first(result.cuda_profiles))
end

function result_to_record(result)
    metrics = result_profile_metrics(result)
    local_memory_kernels = get(metrics, "local_memory_kernels", "0/0")
    return Dict{String, Any}(
        "name" => result.test_def.name,
        "description" => result.test_def.description,
        "operation_type" => result.test_def.operation_type,
        "complexity" => result.test_def.complexity,
        "num_args" => result.test_def.num_args,
        "uses_geometry" => result.test_def.uses_geometry,
        "success" => result.success,
        "time_seconds" => result.time_seconds,
        "time_microseconds" =>
            isnothing(result.time_seconds) ? nothing : result.time_seconds * 1e6,
        "error_msg" => result.error_msg,
        "expression" => render_test_expression(result.test_def),
        "primary_kernel" => get(metrics, "primary_kernel", nothing),
        "registers" => try
            parse(Int, get(metrics, "registers", "0"))
        catch
            nothing
        end,
        "local_bytes" => try
            parse(Int, get(metrics, "local", "0"))
        catch
            nothing
        end,
        "shared_bytes" => try
            parse(Int, get(metrics, "shared", "0"))
        catch
            nothing
        end,
        "local_memory_status" => get(metrics, "status", nothing),
        "local_memory_kernels" => local_memory_kernels,
        "cuda_profile_lines" => result.cuda_profiles,
    )
end

function build_report(results, test_filter::Union{String, Nothing})
    successful = filter(r -> r.success, results)
    times_μs = [r.time_seconds * 1e6 for r in successful]
    return Dict{String, Any}(
        "schema_version" => 1,
        "run_metadata" => collect_run_metadata(test_filter),
        "summary" => Dict{String, Any}(
            "total_tests" => length(results),
            "successful_tests" => length(successful),
            "failed_tests" => length(results) - length(successful),
            "minimum_time_microseconds" =>
                isempty(times_μs) ? nothing : minimum(times_μs),
            "maximum_time_microseconds" =>
                isempty(times_μs) ? nothing : maximum(times_μs),
            "mean_time_microseconds" => isempty(times_μs) ? nothing : mean(times_μs),
            "median_time_microseconds" =>
                length(times_μs) < 2 ? nothing : median(times_μs),
        ),
        "results" => [result_to_record(result) for result in results],
    )
end

function json_escape(s::AbstractString)
    return replace(
        s,
        '\\' => "\\\\",
        '"' => "\\\"",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
    )
end

function to_json(x)
    if x === nothing
        return "null"
    elseif x isa Bool
        return x ? "true" : "false"
    elseif x isa Integer || x isa AbstractFloat
        return string(x)
    elseif x isa AbstractString
        return '"' * json_escape(x) * '"'
    elseif x isa AbstractVector
        return "[" * join([to_json(v) for v in x], ",") * "]"
    elseif x isa AbstractDict
        pairs_json = String[]
        for key in sort!(collect(keys(x)); by = string)
            push!(pairs_json, to_json(string(key)) * ":" * to_json(x[key]))
        end
        return "{" * join(pairs_json, ",") * "}"
    else
        return to_json(string(x))
    end
end

function ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    isempty(parent) || mkpath(parent)
    return nothing
end

resolve_output_path(path::AbstractString) =
    isabspath(path) ? path : joinpath(PROJECT_ROOT, path)

function write_json_report(path::AbstractString, report::Dict{String, Any})
    path = resolve_output_path(path)
    ensure_parent_dir(path)
    open(path, "w") do io
        write(io, to_json(report) * "\n")
    end
    return nothing
end

function html_escape(s::AbstractString)
    return replace(replace(replace(s, "&" => "&amp;"), "<" => "&lt;"), ">" => "&gt;")
end

function markdown_expression_cell(expr::AbstractString)
    # Replace newlines with &#10; so the entire <details> element stays on one line.
    # Markdown table parsers (including VS Code) treat literal newlines inside a cell
    # as row breaks, which creates phantom blank rows in the rendered table.
    escaped = replace(html_escape(expr), "\n" => "&#10;")
    return "<details><summary>show</summary><pre><code class=\"language-julia\">$(escaped)</code></pre></details>"
end

function markdown_table_row(record::Dict{String, Any}, comparison_by_test = nothing)
    cmp =
        isnothing(comparison_by_test) ? nothing :
        get(comparison_by_test, record["name"], nothing)
    baseline_time = "-"
    delta_time = "-"
    baseline_regs = "-"
    baseline_local = "-"
    if !isnothing(cmp)
        bt = get(cmp, "baseline_time_microseconds", nothing)
        dt = get(cmp, "delta_time_percent", nothing)
        br = get(cmp, "baseline_registers", nothing)
        bl = get(cmp, "baseline_local_bytes", nothing)
        baseline_time = isnothing(bt) ? "-" : @sprintf("%.3f", bt)
        delta_time = isnothing(dt) ? "-" : @sprintf("%+.2f%%", dt)
        baseline_regs = isnothing(br) ? "-" : string(br)
        baseline_local = isnothing(bl) ? "-" : string(bl)
    end

    if record["success"]
        time_cell = @sprintf("%.3f", record["time_microseconds"])
        return "| $(record["name"]) | $(record["operation_type"]) | $(time_cell) | $(baseline_time) | $(delta_time) | $(something(record["primary_kernel"], "-")) | $(something(record["registers"], "-")) | $(baseline_regs) | $(something(record["local_bytes"], "-")) | $(baseline_local) | $(something(record["shared_bytes"], "-")) | $(something(record["local_memory_status"], "-")) | $(record["local_memory_kernels"]) | $(markdown_expression_cell(record["expression"])) |"
    else
        return "| $(record["name"]) | $(record["operation_type"]) | FAILED | $(baseline_time) | $(delta_time) | - | - | $(baseline_regs) | - | $(baseline_local) | - | - | - | $(markdown_expression_cell(record["expression"])) |"
    end
end

function write_markdown_report(
    path::AbstractString,
    report::Dict{String, Any};
    comparison::Union{Dict{String, Any}, Nothing} = nothing,
)
    path = resolve_output_path(path)
    ensure_parent_dir(path)
    metadata = report["run_metadata"]
    summary = report["summary"]
    records = report["results"]

    lines = String[]
    push!(lines, "# ClimaCore Stress Test Report")
    push!(lines, "")
    push!(lines, "## Run Metadata")
    push!(lines, "")
    push!(lines, "- Timestamp (UTC): $(metadata["timestamp_utc"])")
    push!(lines, "- Git commit: $(metadata["git_commit"])")
    push!(lines, "- Git branch: $(metadata["git_branch"])")
    push!(lines, "- Git describe: $(metadata["git_describe"])")
    push!(lines, "- Git dirty: $(metadata["git_dirty"])")
    push!(lines, "- Hostname: $(metadata["hostname"])")
    push!(lines, "- Julia version: $(metadata["julia_version"])")
    push!(lines, "- Device backend: $(metadata["device"])")
    push!(lines, "- Test filter: $(metadata["test_filter"])")
    push!(lines, "- Allocated GPU count: $(metadata["allocated_gpu_count"])")
    if !isempty(metadata["allocated_gpu_ids"])
        push!(lines, "- Allocated GPU IDs: $(join(metadata["allocated_gpu_ids"], ", "))")
    end
    push!(lines, "- Node GPU inventory: $(metadata["node_gpu_count"])")
    if !isempty(metadata["gpu_devices"])
        for gpu in metadata["gpu_devices"]
            push!(lines, "- GPU $(gpu["index"]): $(gpu["name"]) ($(gpu["uuid"]))")
        end
    end
    if !isnothing(metadata["slurm_job_id"])
        push!(lines, "- Slurm job ID: $(metadata["slurm_job_id"])")
    end
    if !isnothing(comparison)
        comparable = get(comparison, "comparable", false)
        push!(lines, "- Baseline comparable: $(comparable)")
    end
    push!(lines, "")
    push!(lines, "## Summary")
    push!(lines, "")
    push!(lines, "- Total tests: $(summary["total_tests"])")
    push!(lines, "- Successful tests: $(summary["successful_tests"])")
    push!(lines, "- Failed tests: $(summary["failed_tests"])")
    if !isnothing(summary["minimum_time_microseconds"])
        push!(
            lines,
            "- Minimum time (μs): $(@sprintf("%.3f", summary["minimum_time_microseconds"]))",
        )
        push!(
            lines,
            "- Maximum time (μs): $(@sprintf("%.3f", summary["maximum_time_microseconds"]))",
        )
        push!(
            lines,
            "- Mean time (μs): $(@sprintf("%.3f", summary["mean_time_microseconds"]))",
        )
    end
    if !isnothing(summary["median_time_microseconds"])
        push!(
            lines,
            "- Median time (μs): $(@sprintf("%.3f", summary["median_time_microseconds"]))",
        )
    end
    push!(lines, "")
    push!(lines, "## Results")
    push!(lines, "")
    push!(
        lines,
        "| Test | Type | Time (μs) | Baseline (μs) | Δ Time | Primary kernel | Regs | Base Regs | Local B | Base Local B | Shared B | Local memory | Local-memory kernels | Expression |",
    )
    push!(
        lines,
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    )
    comparison_by_test =
        isnothing(comparison) ? nothing : get(comparison, "by_test_name", nothing)
    for record in records
        push!(lines, markdown_table_row(record, comparison_by_test))
    end

    open(path, "w") do io
        write(io, join(lines, "\n") * "\n")
    end
    return nothing
end

struct CliOptions
    test_filter::Union{String, Nothing}
    output_json::Union{String, Nothing}
    output_markdown::Union{String, Nothing}
    compare_against::Union{String, Nothing}
end

function parse_cli_args(args::Vector{String})
    test_filter = nothing
    output_json = nothing
    output_markdown = nothing
    compare_against = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--output-json=")
            output_json = split(arg, "="; limit = 2)[2]
        elseif arg == "--output-json"
            i += 1
            i > length(args) && error("Missing path after --output-json")
            output_json = args[i]
        elseif startswith(arg, "--output-markdown=")
            output_markdown = split(arg, "="; limit = 2)[2]
        elseif arg == "--output-markdown"
            i += 1
            i > length(args) && error("Missing path after --output-markdown")
            output_markdown = args[i]
        elseif startswith(arg, "--compare-against=")
            compare_against = split(arg, "="; limit = 2)[2]
        elseif arg == "--compare-against"
            i += 1
            i > length(args) && error("Missing path after --compare-against")
            compare_against = args[i]
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        elseif isnothing(test_filter)
            test_filter = arg
        else
            error(
                "Multiple positional arguments provided; expected at most one test filter",
            )
        end
        i += 1
    end

    return CliOptions(test_filter, output_json, output_markdown, compare_against)
end

function _skip_ws(s::AbstractString, i::Int)
    while i <= lastindex(s)
        c = s[i]
        if c == ' ' || c == '\n' || c == '\r' || c == '\t'
            i = nextind(s, i)
        else
            break
        end
    end
    return i
end

function _json_parse_string(s::AbstractString, i::Int)
    @assert s[i] == '"'
    i = nextind(s, i)
    buf = IOBuffer()
    while i <= lastindex(s)
        c = s[i]
        if c == '"'
            i = nextind(s, i)
            return String(take!(buf)), i
        elseif c == '\\'
            i = nextind(s, i)
            i > lastindex(s) && error("Invalid JSON escape")
            esc = s[i]
            if esc == '"'
                write(buf, '"')
            elseif esc == '\\'
                write(buf, '\\')
            elseif esc == '/'
                write(buf, '/')
            elseif esc == 'b'
                write(buf, '\b')
            elseif esc == 'f'
                write(buf, '\f')
            elseif esc == 'n'
                write(buf, '\n')
            elseif esc == 'r'
                write(buf, '\r')
            elseif esc == 't'
                write(buf, '\t')
            elseif esc == 'u'
                # Minimal unicode handling: keep literal for unsupported code-point decoding
                # Our generated JSON doesn't emit \u escapes, so this keeps parser robust enough.
                i = nextind(s, i)
                for _ in 1:4
                    i > lastindex(s) && error("Invalid JSON unicode escape")
                    i = nextind(s, i)
                end
                write(buf, '?')
                continue
            else
                error("Unsupported JSON escape: \\$esc")
            end
            i = nextind(s, i)
        else
            write(buf, c)
            i = nextind(s, i)
        end
    end
    error("Unterminated JSON string")
end

function _json_parse_number(s::AbstractString, i::Int)
    start = i
    while i <= lastindex(s)
        c = s[i]
        if isdigit(c) || c in ('-', '+', '.', 'e', 'E')
            i = nextind(s, i)
        else
            break
        end
    end
    token = s[start:prevind(s, i)]
    if occursin('.', token) || occursin('e', token) || occursin('E', token)
        return parse(Float64, token), i
    else
        return parse(Int, token), i
    end
end

function _json_parse_value(s::AbstractString, i::Int)
    i = _skip_ws(s, i)
    i > lastindex(s) && error("Unexpected end of JSON")
    c = s[i]
    if c == '"'
        return _json_parse_string(s, i)
    elseif c == '{'
        obj = Dict{String, Any}()
        i = nextind(s, i)
        i = _skip_ws(s, i)
        if i <= lastindex(s) && s[i] == '}'
            return obj, nextind(s, i)
        end
        while true
            key, i = _json_parse_string(s, _skip_ws(s, i))
            i = _skip_ws(s, i)
            s[i] == ':' || error("Expected ':' in JSON object")
            i = nextind(s, i)
            value, i = _json_parse_value(s, i)
            obj[key] = value
            i = _skip_ws(s, i)
            if s[i] == '}'
                return obj, nextind(s, i)
            elseif s[i] == ','
                i = nextind(s, i)
            else
                error("Expected ',' or '}' in JSON object")
            end
        end
    elseif c == '['
        arr = Any[]
        i = nextind(s, i)
        i = _skip_ws(s, i)
        if i <= lastindex(s) && s[i] == ']'
            return arr, nextind(s, i)
        end
        while true
            value, i = _json_parse_value(s, i)
            push!(arr, value)
            i = _skip_ws(s, i)
            if s[i] == ']'
                return arr, nextind(s, i)
            elseif s[i] == ','
                i = nextind(s, i)
            else
                error("Expected ',' or ']' in JSON array")
            end
        end
    elseif c == 't' && i + 3 <= lastindex(s) && s[i:(i + 3)] == "true"
        return true, i + 4
    elseif c == 'f' && i + 4 <= lastindex(s) && s[i:(i + 4)] == "false"
        return false, i + 5
    elseif c == 'n' && i + 3 <= lastindex(s) && s[i:(i + 3)] == "null"
        return nothing, i + 4
    else
        return _json_parse_number(s, i)
    end
end

function parse_json_text(s::AbstractString)
    value, i = _json_parse_value(s, firstindex(s))
    i = _skip_ws(s, i)
    i <= lastindex(s) && error("Trailing content after JSON value")
    return value
end

function read_json_report(path::AbstractString)
    path = resolve_output_path(path)
    text = read(path, String)
    parsed = parse_json_text(text)
    parsed isa Dict{String, Any} || error("Expected top-level JSON object in $(path)")
    return parsed
end

function _record_lookup(records)
    table = Dict{String, Dict{String, Any}}()
    for rec_any in records
        rec_any isa Dict{String, Any} || continue
        haskey(rec_any, "name") || continue
        table[string(rec_any["name"])] = rec_any
    end
    return table
end

function _to_float_or_nothing(x)
    x === nothing && return nothing
    x isa Integer && return Float64(x)
    x isa AbstractFloat && return Float64(x)
    return nothing
end

function compare_reports(current::Dict{String, Any}, baseline::Dict{String, Any})
    current_records = _record_lookup(current["results"])
    baseline_records = _record_lookup(baseline["results"])

    run_cur = current["run_metadata"]
    run_base = baseline["run_metadata"]
    comparable =
        get(run_cur, "device", nothing) == get(run_base, "device", nothing) &&
        get(run_cur, "julia_version", nothing) == get(run_base, "julia_version", nothing)

    by_name = Dict{String, Dict{String, Any}}()
    for name in sort(collect(keys(current_records)))
        rec_cur = current_records[name]
        rec_base = get(baseline_records, name, nothing)
        cur_time = _to_float_or_nothing(get(rec_cur, "time_microseconds", nothing))
        base_time =
            isnothing(rec_base) ? nothing :
            _to_float_or_nothing(get(rec_base, "time_microseconds", nothing))
        delta_pct = if !isnothing(cur_time) && !isnothing(base_time) && base_time != 0
            100.0 * (cur_time - base_time) / base_time
        else
            nothing
        end
        by_name[name] = Dict{String, Any}(
            "baseline_found" => !isnothing(rec_base),
            "baseline_success" =>
                isnothing(rec_base) ? nothing : get(rec_base, "success", false),
            "baseline_time_microseconds" => base_time,
            "current_time_microseconds" => cur_time,
            "delta_time_percent" => delta_pct,
            "baseline_registers" =>
                isnothing(rec_base) ? nothing : get(rec_base, "registers", nothing),
            "current_registers" => get(rec_cur, "registers", nothing),
            "baseline_local_bytes" =>
                isnothing(rec_base) ? nothing : get(rec_base, "local_bytes", nothing),
            "current_local_bytes" => get(rec_cur, "local_bytes", nothing),
        )
    end

    return Dict{String, Any}(
        "comparable" => comparable,
        "baseline_metadata" => run_base,
        "current_metadata" => run_cur,
        "by_test_name" => by_name,
    )
end

# ============================================================================
# TEST CASE GENERATION
# ============================================================================

"""
    generate_field_test_code(test_name::String, test_impl::String) -> String

Generate a complete test code block that can be run in a subprocess.
Includes ClimaCore setup and proper device handling.
"""
function generate_field_test_code(test_name::String, test_impl::String)
    device_init = if has_cuda_env()
        """
        using CUDA
        CUDA.allowscalar(false)
        """
    else
        ""
    end

    return """
    import Pkg
    using Printf
    using BenchmarkTools
    import ClimaComms
    ClimaComms.@import_required_backends

    using ClimaCore
    using ClimaCore.Fields
    using ClimaCore.Spaces
    using ClimaCore.Domains
    using ClimaCore.Meshes
    using ClimaCore.Geometry
    using ClimaCore.Topologies
    using ClimaCore.Quadratures

    $device_init

    # Suppress informational logging
    using Logging
    disable_logging(Logging.Info)

    try
        $test_impl
    catch e
        @error "Test failed: \$e"
        rethrow()
    end
    """
end

"""
    create_spectral_space()

Helper function to create a reusable spectral element space setup code string.
"""
function create_spectral_space()
    return """
    FT = Float64
    context = ClimaComms.context()
    if context isa ClimaComms.MPICommsContext
        ClimaComms.init(context)
    end

    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(Geometry.XPoint(-1.0), Geometry.XPoint(1.0); periodic=true),
        Domains.IntervalDomain(Geometry.YPoint(-1.0), Geometry.YPoint(1.0); periodic=true),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{4}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    """
end

"""
    arithmetic_test(depth::Int) -> String

Generate test code for arithmetic operations with given nesting depth.
ClimaCore automatically generates kernels from the broadcast operation.
"""
function arithmetic_test(depth::Int)
    # Build expression with `depth` levels of nesting
    expr = "x"
    ops = ["+", "*", "/", "-"]
    for i in 1:depth
        op = ops[mod(i, length(ops)) + 1]
        val = i
        expr = "($expr $op $val.0)"
    end

    test_impl = create_spectral_space() * """

    f = Fields.Field(FT, space)
    fill!(Fields.field_values(f), 1.5)

    op(x) = $expr

    # Warm up compilation
    _ = op(1.5)
    _ = op.(f)

    # Benchmark ClimaCore's generated kernel
    trial = @benchmark \$op.(\$f) samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: arithmetic_depth_$(depth) = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("arithmetic_depth_$(depth)", test_impl)
end

"""
    multiarg_test(nargs::Int) -> String

Generate test code for operations with multiple field arguments.
ClimaCore automatically generates kernels from the broadcast operation.
"""
function multiarg_test(nargs::Int)
    # Build argument list
    args_decl = join(
        [
            "f$i = Fields.Field(FT, space);\n    fill!(Fields.field_values(f$i), $(Float64(i)))"
            for i in 1:nargs
        ],
        "\n    ",
    )
    args_list = join(["f$i" for i in 1:nargs], ", ")
    bench_args_list = join(["\$f$i" for i in 1:nargs], ", ")

    # Build operation: (f1 + f2 + ...) / (f_last + 1)
    sum_expr = join(["f$i" for i in 1:(nargs - 1)], " + ")
    op_expr = "($sum_expr) / (f$nargs + 1.0)"

    test_impl = create_spectral_space() * """

    $args_decl

    op($args_list) = $op_expr

    # Warm up compilation
    _ = op($(join(["$(Float64(i))" for i in 1:nargs], ", ")))
    _ = op.($args_list)

    # Benchmark ClimaCore's generated kernel
    trial = @benchmark \$op.($bench_args_list) samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: multiarg_$(nargs)_args = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("multiarg_$(nargs)_args", test_impl)
end

"""
    functions_test(funcs::Vector{String}, depth::Int) -> String

Generate test code for composed mathematical functions.
ClimaCore automatically generates kernels from the broadcast operation.
"""
function functions_test(funcs::Vector{String}, depth::Int)
    label = if funcs == ["log"]
        "log"
    elseif funcs == ["sqrt"]
        "sqrt"
    elseif funcs == ["log", "sqrt", "abs"]
        "mixed"
    else
        join(funcs, "_")
    end

    test_name = "functions_$(label)_depth_$(depth)"

    # Build nested function composition with domain-safe wrappers for real-valued log/sqrt
    expr = "x + 0.5"
    for i in depth:-1:1
        func = funcs[mod1(i, length(funcs))]
        if func == "log"
            expr = "log(abs($expr) + 1.5)"
        elseif func == "sqrt"
            expr = "sqrt(abs($expr) + 1.5)"
        else
            expr = "$func($expr)"
        end
    end

    test_impl = create_spectral_space() * """

    f = Fields.Field(FT, space)
    fill!(Fields.field_values(f), 1.5)

    op(x) = $expr

    # Warm up compilation
    _ = op(1.5)
    _ = op.(f)

    # Benchmark ClimaCore's generated kernel
    trial = @benchmark \$op.(\$f) samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    timing_name = $(repr(test_name))
    @printf "TIMING: %s = %.6f s\\n" timing_name time_μs / 1e6
    """

    return generate_field_test_code(test_name, test_impl)
end

"""
    projection_test(complexity::Int) -> String

Generate test code for projection operations on geometric objects.
ClimaCore automatically generates kernels from the broadcast operation.
"""
function projection_test(complexity::Int)
    # Use @. macro so ClimaCore can supply LocalGeometry during the fused broadcast
    proj_terms = join(
        ["Geometry.project(Geometry.Covariant12Axis(), v)" for _ in 1:complexity],
        " .+ ",
    )
    test_name = "projection_$(complexity)x_chained"

    test_impl = create_spectral_space() * """

    v = Fields.Field(Geometry.Covariant12Vector{FT}, space)
    fill!(Fields.field_values(v), Geometry.Covariant12Vector(1.0, 2.0))

    # Warm up compilation
    _ = @. $proj_terms

    # Benchmark: $complexity fused project calls in one @. expression
    trial = @benchmark (v = \$v; @. $proj_terms) samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: $(test_name) = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code(test_name, test_impl)
end

"""
    create_column_space()

Helper function to create a vertical column (finite difference) space setup code string.
Produces both center and face spaces needed for C2F/F2C operators.
"""
function create_column_space()
    return """
    FT = Float64
    context = ClimaComms.context()
    if context isa ClimaComms.MPICommsContext
        ClimaComms.init(context)
    end

    col_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(1.0);
        boundary_names = (:bottom, :top),
    )
    col_mesh = Meshes.IntervalMesh(col_domain; nelems = 16)
    col_topology = Topologies.IntervalTopology(context, col_mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(col_topology)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)
    """
end

"""
    div_test(n::Int) -> String

Generate test code that packs n Divergence calls into a single broadcast expression.
Tests how many spectral-element divergences the compiler can inline before giving up.
"""
function div_test(n::Int)
    warm = join(["div_op.(v .* $(i).0)" for i in 1:n], " .+ ")
    bench = join(["\$div_op.(\$v .* $(i).0)" for i in 1:n], " .+ ")

    test_impl = create_spectral_space() * """

    using ClimaCore.Operators

    div_op = Operators.Divergence()
    v = Fields.Field(Geometry.Contravariant12Vector{FT}, space)
    fill!(Fields.field_values(v), Geometry.Contravariant12Vector(1.0, 2.0))

    # Warm up: $n divergence calls in one expression
    _ = $warm

    # Benchmark ClimaCore's kernel: $n divergence calls fused into one expression
    trial = @benchmark $bench samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: div_$(n)_ops = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("div_$(n)_ops", test_impl)
end

"""
    curl_test(n::Int) -> String

Generate test code that packs n Curl calls into a single broadcast expression.
Tests how many spectral-element curls the compiler can inline before giving up.
"""
function curl_test(n::Int)
    warm = join(["curl_op.(v .* $(i).0)" for i in 1:n], " .+ ")
    bench = join(["\$curl_op.(\$v .* $(i).0)" for i in 1:n], " .+ ")

    test_impl = create_spectral_space() * """

    using ClimaCore.Operators

    curl_op = Operators.Curl()
    v = Fields.Field(Geometry.Covariant12Vector{FT}, space)
    fill!(Fields.field_values(v), Geometry.Covariant12Vector(1.0, 2.0))

    # Warm up: $n curl calls in one expression
    _ = $warm

    # Benchmark ClimaCore's kernel: $n curl calls fused into one expression
    trial = @benchmark $bench samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: curl_$(n)_ops = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("curl_$(n)_ops", test_impl)
end

"""
    interp_test(n::Int) -> String

Generate test code that packs n InterpolateC2F calls into a single broadcast expression.
Tests how many center-to-face interpolations the compiler can inline before giving up.
"""
function interp_test(n::Int)
    warm = join(["interp.(ᶜf .* $(i).0)" for i in 1:n], " .+ ")
    bench = join(["\$interp.(\$ᶜf .* $(i).0)" for i in 1:n], " .+ ")

    test_impl = create_column_space() * """

    using ClimaCore.Operators

    interp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜf = Fields.Field(FT, center_space)
    fill!(Fields.field_values(ᶜf), 1.5)

    # Warm up: $n InterpolateC2F calls in one expression
    _ = $warm

    # Benchmark ClimaCore's kernel: $n C2F interpolations fused into one expression
    trial = @benchmark $bench samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: interp_c2f_$(n)_ops = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("interp_c2f_$(n)_ops", test_impl)
end

"""
    weighted_interp_test(n::Int) -> String

Generate test code that packs n WeightedInterpolateC2F calls into a single broadcast expression.
Tests how many weighted center-to-face interpolations the compiler can inline before giving up.
"""
function weighted_interp_test(n::Int)
    warm = join(["winterp.(ᶜw, ᶜf .* $(i).0)" for i in 1:n], " .+ ")
    bench = join(["\$winterp.(\$ᶜw, \$ᶜf .* $(i).0)" for i in 1:n], " .+ ")

    test_impl = create_column_space() * """

    using ClimaCore.Operators

    winterp = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜw = Fields.Field(FT, center_space)
    ᶜf = Fields.Field(FT, center_space)
    fill!(Fields.field_values(ᶜw), 1.0)
    fill!(Fields.field_values(ᶜf), 1.5)

    # Warm up: $n WeightedInterpolateC2F calls in one expression
    _ = $warm

    # Benchmark ClimaCore's kernel: $n weighted C2F interpolations fused into one expression
    trial = @benchmark $bench samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: weighted_interp_c2f_$(n)_ops = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("weighted_interp_c2f_$(n)_ops", test_impl)
end

"""
    upwinding_test(n::Int) -> String

Generate test code that packs n Upwind3rdOrderBiasedProductC2F calls into a single
broadcast expression. Tests how many 3rd-order upwind flux evaluations the compiler
can inline before giving up.
"""
function upwinding_test(n::Int)
    warm = join(["upwind.(ᶠv, ᶜf .* $(i).0)" for i in 1:n], " .+ ")
    bench = join(["\$upwind.(\$ᶠv, \$ᶜf .* $(i).0)" for i in 1:n], " .+ ")

    test_impl = create_column_space() * """

    using ClimaCore.Operators

    upwind = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    ᶠv = Fields.Field(Geometry.WVector{FT}, face_space)
    ᶜf = Fields.Field(FT, center_space)
    fill!(Fields.field_values(ᶠv), Geometry.WVector(1.0))
    fill!(Fields.field_values(ᶜf), 1.5)

    # Warm up: $n Upwind3rdOrderBiasedProductC2F calls in one expression
    _ = $warm

    # Benchmark ClimaCore's kernel: $n 3rd-order upwind calls fused into one expression
    trial = @benchmark $bench samples=10 evals=1

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: upwinding_3rdorder_$(n)_ops = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("upwinding_3rdorder_$(n)_ops", test_impl)
end

# ============================================================================
# TEST CATALOG: Define all test cases
# ============================================================================

"""
    struct TestDef

Definition of a single test case for generation and execution.
"""
struct TestDef
    name::String
    description::String
    operation_type::String    # "arithmetic", "projection", "multiarg", "functions", "divergence", "curl", "interpolate", "weighted_interpolate", "upwinding"
    complexity::Int           # nesting depth or argument count
    num_args::Int
    uses_geometry::Bool
    code_generator::Function  # Function that generates test code
end

# Create test definitions
const ALL_TESTS =
    [
        # Arithmetic with varying depth
        [
            TestDef("arithmetic_depth_$i", "Arithmetic operations with depth $i",
                "arithmetic", i, 1, false,
                () -> arithmetic_test(i)) for i in [1, 3, 5, 7, 10]
        ]

        # Multiple arguments
        [
            TestDef("multiarg_$(i)_args", "Operations with $i field arguments",
                "multiarg", 1, i, false,
                () -> multiarg_test(i)) for i in [2, 3, 4, 6, 8]
        ]

        # Function compositions
        [
            TestDef("functions_log_depth_$i", "Log function composed $i times",
                "functions", i, 1, false,
                () -> functions_test(["log"], i)) for i in [1, 2, 3]
        ]
        [
            TestDef("functions_sqrt_depth_$i", "Sqrt function composed $i times",
                "functions", i, 1, false,
                () -> functions_test(["sqrt"], i)) for i in [1, 2, 3]
        ]
        [
            TestDef("functions_mixed_depth_$i", "Mixed functions (log, sqrt, abs) depth $i",
                "functions", i, 1, false,
                () -> functions_test(["log", "sqrt", "abs"], i)) for i in [1, 2]
        ]

        # Projection operations
        [
            TestDef("projection_$(i)x_chained", "Chained projection operations x$i",
                "projection", i, 1, true,
                () -> projection_test(i)) for i in [1, 2, 3, 5]
        ]

        # Divergence operations: pack N calls into one expression
        [
            TestDef("div_$(i)_ops", "$i Divergence calls in one expression",
                "divergence", i, 1, true,
                () -> div_test(i)) for i in [1, 2, 4, 6, 8]
        ]

        # Curl operations: pack N calls into one expression
        [
            TestDef("curl_$(i)_ops", "$i Curl calls in one expression",
                "curl", i, 1, true,
                () -> curl_test(i)) for i in [1, 2, 4, 6, 8]
        ]

        # C2F interpolation: pack N calls into one expression
        [
            TestDef("interp_c2f_$(i)_ops", "$i InterpolateC2F calls in one expression",
                "interpolate", i, 1, false,
                () -> interp_test(i)) for i in [1, 2, 4, 6, 8]
        ]

        # Weighted C2F interpolation: pack N calls into one expression
        [
            TestDef("weighted_interp_c2f_$(i)_ops",
                "$i WeightedInterpolateC2F calls in one expression",
                "weighted_interpolate", i, 1, false,
                () -> weighted_interp_test(i)) for i in [1, 2, 4, 6, 8]
        ]

        # Van Leer upwinding: pack N calls into one expression
        [
            TestDef("upwinding_3rdorder_$(i)_ops",
                "$i Upwind3rdOrderBiasedProductC2F calls in one expression",
                "upwinding", i, 1, false,
                () -> upwinding_test(i)) for i in [1, 2, 4, 6, 8]
        ]
    ] |> vec

# ============================================================================
# EXECUTION AND REPORTING
# ============================================================================

"""
    mutable struct TestResult

Stores the result of a single test execution.
"""
mutable struct TestResult
    test_def::TestDef
    success::Bool
    time_seconds::Union{Float64, Nothing}
    error_msg::String
    cuda_profiles::Vector{String}

    TestResult(test_def) = new(test_def, false, nothing, "", String[])
end

"""
    run_test(test_def::TestDef) -> TestResult

Run a single test case in a subprocess and collect results.
"""
function run_test(test_def::TestDef)
    result = TestResult(test_def)

    # Generate test code
    test_code = test_def.code_generator()

    # Run in subprocess
    success, output, error = run_test_subprocess(test_code, test_def.name)

    if success
        result.cuda_profiles =
            summarize_cuda_profiles(parse_cuda_profile_from_output(output))
        # Parse timing from output
        timings = parse_timings_from_output(output)
        if haskey(timings, test_def.name)
            result.time_seconds = timings[test_def.name]
            result.success = true
        else
            result.error_msg = "Failed to parse timing from output"
        end
    else
        result.error_msg = error
    end

    return result
end

"""
    print_result(result::TestResult)

Pretty-print a test result.
"""
function print_result(result::TestResult)
    test = result.test_def
    if result.success
        time_μs = result.time_seconds * 1e6
        @printf "  %-45s %10.3f μs" test.name time_μs
        @printf " (depth=%d, args=%d)" test.complexity test.num_args
        if test.uses_geometry
            print(" [geometry]")
        end
        println()
        for line in result.cuda_profiles
            println("    " * line)
        end
    else
        @printf "  %-45s ERROR: %s\n" test.name result.error_msg
    end
end

"""
    main(; test_filter::Union{String, Nothing}=nothing)

Run all tests and produce a report.
"""
function main(;
    test_filter::Union{String, Nothing} = nothing,
    output_json::Union{String, Nothing} = nothing,
    output_markdown::Union{String, Nothing} = nothing,
    compare_against::Union{String, Nothing} = nothing,
)
    println("="^90)
    println("ClimaCore Compiler Stress Test Suite - Pointwise/Broadcast Operations")
    println("Device: $(DEVICE)")
    has_cuda_env() && println("CUDA warnings disabled to catch only actual failures")
    println("="^90)
    println()

    # Filter tests if requested
    tests = if isnothing(test_filter)
        ALL_TESTS
    else
        filter(t -> contains(t.name, test_filter), ALL_TESTS)
    end

    if isempty(tests)
        println("No tests matching filter: $test_filter")
        println("\nAvailable test categories:")
        for op_type in sort(unique(t.operation_type for t in ALL_TESTS))
            matching = filter(t -> t.operation_type == op_type, ALL_TESTS)
            println("  $op_type: $(length(matching)) tests")
            for test in matching[1:min(3, length(matching))]
                println("    - $(test.name)")
            end
            length(matching) > 3 && println("    ... and $(length(matching) - 3) more")
        end
        return nothing
    end

    println("Running $(length(tests)) test case(s):")
    println()

    results = TestResult[]

    # Run all tests
    for (i, test) in enumerate(tests)
        @printf "[%2d/%2d] %-45s ... " i length(tests) test.name
        flush(stdout)

        result = run_test(test)
        push!(results, result)

        if result.success
            println("✓")
        else
            println("✗")
        end
    end

    println()
    println("="^90)
    println("Results")
    println("="^90)
    println()

    # Group by operation type
    by_type = Dict{String, Vector{TestResult}}()
    for result in results
        op_type = result.test_def.operation_type
        if !haskey(by_type, op_type)
            by_type[op_type] = []
        end
        push!(by_type[op_type], result)
    end

    # Print results by type
    for op_type in sort(collect(keys(by_type)))
        type_results = by_type[op_type]
        successful = count(r -> r.success, type_results)

        println("$op_type operations ($successful/$(length(type_results)) successful):")

        for result in sort(type_results, by = r -> r.test_def.complexity)
            print_result(result)
        end

        println()
    end

    # Summary statistics
    successful = filter(r -> r.success, results)
    if !isempty(successful)
        println("="^90)
        println("Performance Summary")
        println("="^90)

        times = [r.time_seconds * 1e6 for r in successful]  # convert to microseconds

        println("Execution times (microseconds):")
        @printf "  Minimum:     %.3f μs\n" minimum(times)
        @printf "  Maximum:     %.3f μs\n" maximum(times)
        @printf "  Mean:        %.3f μs\n" mean(times)
        if length(times) >= 2
            @printf "  Median:      %.3f μs\n" median(times)
        end
    end

    println()
    num_successful = length(successful)
    num_failed = length(results) - num_successful

    if num_failed == 0
        println("✓ All $(num_successful) tests passed!")
    else
        println("✗ $(num_failed) test(s) failed out of $(length(results)) total")
    end

    report = build_report(results, test_filter)
    comparison = nothing
    if !isnothing(compare_against)
        baseline_path = resolve_output_path(compare_against)
        baseline_report = read_json_report(baseline_path)
        comparison = compare_reports(report, baseline_report)
        report["comparison"] = comparison
    end
    if !isnothing(output_json)
        output_json = resolve_output_path(output_json)
        write_json_report(output_json, report)
        println("Wrote JSON report: $(output_json)")
    end
    if !isnothing(output_markdown)
        output_markdown = resolve_output_path(output_markdown)
        write_markdown_report(output_markdown, report; comparison)
        println("Wrote markdown report: $(output_markdown)")
    end

    return report
end

# ============================================================================
# Entry point
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Initialize project first
    initialize_project()

    options = parse_cli_args(ARGS)
    main(
        ;
        test_filter = options.test_filter,
        output_json = options.output_json,
        output_markdown = options.output_markdown,
        compare_against = options.compare_against,
    )
end
