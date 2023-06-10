struct BuildkiteJob{C, L, K, E}
    commands::C
    label::L
    key::K
    env::E
    function BuildkiteJob(dict::AbstractDict)
        command = get(dict, "command", "")
        if command isa AbstractString
            command = BuildkiteCommand[BuildkiteCommand(command)]
        elseif command isa AbstractArray
            command = BuildkiteCommand.(command)
        end
        label = get(dict, "label", "")
        key = get(dict, "key", "")
        env = get(dict, "env", "")
        t = typeof.((command, label, key, env))
        return new{t...}(command, label, key, env)
    end
end

struct BuildkiteCommand{C <: AbstractString}
    command::C
end

Base.string(bkc::BuildkiteCommand) = bkc.command

commands(j::BuildkiteJob) = j.commands

get_file(bkc::BuildkiteCommand) = get_file(bkc.command)
get_file(s::AbstractString) =
    !occursin(".jl", s) ? "" : last(split(first(split(s, ".jl")), " ")) * ".jl"
get_files(v::Vector{<:BuildkiteCommand}) =
    filter(x -> !isempty(x), get_file.(string.(v)))
get_files(j::BuildkiteJob) = get_files(commands(j))
get_key(j::BuildkiteJob) = j.key
get_label(j::BuildkiteJob) = j.label

get_env(j::BuildkiteJob) = j.env
keyword_match(j::BuildkiteJob, keywords) = any(
    Base.Iterators.flatten(map(commands(j)) do cmd
            map(x -> occursin(x, string(cmd)), keywords)
        end),
)
function get_device(j::BuildkiteJob)
    if keyword_match(j, ["gpu", "CUDADevice"])
        return "GPU"
    else
        return "CPU"
    end
end
function float_type(j::BuildkiteJob)
    if keyword_match(j, ["Float64"])
        return "Float64"
    elseif keyword_match(j, ["Float32"])
        return "Float32"
    else
        return ""
    end
end
function get_context(j::BuildkiteJob)
    if Pair("CLIMACOMMS_CONTEXT", "MPI") in get_env(j)
        return "MPI"
    else
        return "Singleton"
    end
end

function get_config(j::BuildkiteJob)
    c = [get_device(j), float_type(j), get_context(j)]
    filter!(x -> !isempty(x), c)
    join(c, ", ")
end

function test_job(j::BuildkiteJob)
    @eval Main begin
        # Configure CL arguments
        local env = get_env($j)
        empty!(ARGS)
        for (k, d) in get_env($j)
            push!(ARGS, "--$k", "$d")
        end

        cl_args = string.($(j.commands))
        for (k, d) in get_env($j)
            push!(ARGS, "--$k", "$d")
        end

        # include()
    end
end

using PrettyTables
function tabulate_jobs(bkjs; verbose = false)
    title = "Tests run in test/runtests.jl:"
    test_names = get_label.(bkjs)
    if verbose
        test_env = get_env.(bkjs)
        test_cmds = commands.(bkjs)
        data = hcat(test_names, test_env, test_cmds)
        header = ["Name", "env", "Commands"]
    else
        test_config = get_config.(bkjs)
        test_files = get_files.(bkjs)
        data = hcat(test_names, test_config, test_files)
        header = ["Name", "config", "File"]
    end
    PrettyTables.pretty_table(data; title, header, alignment = :l, crop = :none)
end
