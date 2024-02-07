import Cassette
import MethodAnalysis

# TODO: Should we use @nospecialize on f everywhere?

all_symbols_in_module(m) = Base.unsorted_names(m, all = true, imported = true)

function llvm_function_string(f)
    type_string = string(typeof(f).name.name)
    return startswith(type_string, '#') ? split(type_string, '#')[2] :
           type_string
end

call_string(f, args...) = "$f($(join(map(repr, args), ", ")))"

args_type(args) =
    Tuple{map(arg -> arg isa Type ? Type{arg} : typeof(arg), args)...}

function_type_from_value(::F) where {F <: Function} = F # regular functions
function_type_from_value(::Type{F}) where {F} = F # anonymous functions,
# closures, or generic structs

function method_instance(f, T)
    call = MethodAnalysis.methodinstance(f, (T.parameters...,))
    isnothing(call) || return call

    # Workaround for a bug in MethodAnalysis from using structs as functions
    possible_methods = methods(f, T)
    @assert length(possible_methods) == 1
    possible_calls = MethodAnalysis.methodinstances(possible_methods[1])
    @assert length(possible_calls) == 1
    return possible_calls[1]
end

function function_type_in_modules(f_string, modules)
    possible_Fs = []
    for m in modules
        available_symbols = all_symbols_in_module(m)
        if Symbol(f_string) in available_symbols
            F = function_type_from_value(getfield(m, Symbol(f_string)))
            if !(F in possible_Fs)
                push!(possible_Fs, F)
            end
        else
            most_recent_anonymous_F = nothing
            most_recent_anonymous_F_id = 0 # the id number is always positive
            anonymous_F_prefix =
                startswith(f_string, '#') ? f_string : '#' * f_string
            for symbol in available_symbols
                if startswith(string(symbol), anonymous_F_prefix)
                    @assert length(split(string(symbol), '#')) == 3
                    anonymous_F_id = parse(Int, split(string(symbol), '#')[3])
                    if anonymous_F_id > most_recent_anonymous_F_id
                        most_recent_anonymous_F =
                            function_type_from_value(getfield(m, symbol))
                        most_recent_anonymous_F_id = anonymous_F_id
                    end
                end
            end
            if !(
                isnothing(most_recent_anonymous_F) ||
                most_recent_anonymous_F in possible_Fs
            )
                push!(possible_Fs, most_recent_anonymous_F)
            end
        end
    end
    return if isempty(possible_Fs)
        nothing
    elseif length(possible_Fs) == 1
        possible_Fs[1]
    else
        error("Ambiguous function string: \"$f_string\" could refer to \
               the function types $(join(possible_Fs, ", ", " or "))")
    end
end

# Finds the type of the function (or the function-like object) represented by
# `f_string`, which comes from the LLVM code of a method that was defined
# in `calling_module`.
function inferred_function_type(f_string, calling_module)
    searched_modules = []

    new_modules = [calling_module, Base.active_module()]
    possible_F = function_type_in_modules(f_string, new_modules)
    isnothing(possible_F) || return possible_F
    append!(searched_modules, new_modules)

    new_module_names = filter(all_symbols_in_module(calling_module)) do symbol
        value = getfield(calling_module, symbol)
        value isa Module && !(value in searched_modules)
    end
    new_modules =
        map(symbol -> getfield(calling_module, symbol), new_module_names)
    possible_F = function_type_in_modules(f_string, new_modules)
    isnothing(possible_F) || return possible_F
    append!(searched_modules, new_modules)

    new_modules = Iterators.flatmap(new_modules) do m
        new_module_names_in_m = filter(all_symbols_in_module(m)) do symbol
            value = getfield(m, symbol)
            value isa Module && !(value in searched_modules)
        end
        map(symbol -> getfield(m, symbol), new_module_names_in_m)
    end
    possible_F = function_type_in_modules(f_string, new_modules)
    isnothing(possible_F) || return possible_F
    append!(searched_modules, new_modules)

    new_modules = filter(collect(values(Base.loaded_modules))) do m
        !(m in searched_modules)
    end
    possible_F = function_type_in_modules(f_string, new_modules)
    isnothing(possible_F) || return possible_F
    append!(searched_modules, new_modules)

    @warn "Unable to find \"$f_string\" in any of the currently loaded modules"
    return Union{}
end

################################################################################

Cassette.@context UnoptimizedCallCounterCtx
Cassette.prehook(ctx::UnoptimizedCallCounterCtx, _, _...) = ctx.metadata[] += 1
function count_unoptimized(
    f::F,
    args...;
    expr_string = call_string(f, args...),
) where {F}
    ctx = UnoptimizedCallCounterCtx(; metadata = Ref(0))
    Cassette.prehook(ctx, f, args...)
    try
        Cassette.overdub(ctx, f, args...)
    catch e
        @warn "Error during analysis of $expr_string: $e"
    end
    println("@count_unoptimized $expr_string:\n  $(ctx.metadata[])\n")
    return ctx.metadata[]
end

################################################################################

# Categorized list of LLVM instructions from https://llvm.org/docs/LangRef.html.
# The only instruction not in this list is "call", which is counted separately.
# TODO: Handle "invoke" and "callbr" in the same way as "call".
# TODO: Handle "br", "switch", and other branching instructions separately.
const int_instructions = (
    "add",
    "sub",
    "mul",
    "udiv",
    "sdiv",
    "urem",
    "srem",
    "icmp",
    "shl",
    "lshr",
    "ashr",
    "and",
    "or",
    "xor",
)
const float_instructions =
    ("fneg", "fadd", "fsub", "fmul", "fdiv", "frem", "fcmp")
const conversion_instructions = (
    "trunc",
    "zext",
    "sext",
    "fptrunc",
    "fpext",
    "fptoui",
    "fptosi",
    "uitofp",
    "sitofp",
    "ptrtoint",
    "inttoptr",
    "bitcast",
    "addrspacecast",
)
const struct_instructions = (
    "extractelement",
    "insertelement",
    "shufflevector",
    "extractvalue",
    "insertvalue",
)
const memory_instructions = (
    "alloca",
    "load",
    "store",
    "fence",
    "cmpxchg",
    "atomicrmw",
    "getelementptr",
)
const control_flow_instructions = (
    "phi",
    "select",
    "ret",
    "br",
    "switch",
    "indirectbr",
    "invoke",
    "callbr",
    "resume",
    "catchswitch",
    "catchret",
    "cleanupret",
    "unreachable",
)
const other_instructions =
    ("freeze", "va_arg", "landingpad", "catchpad", "cleanuppad")

# TODO: Estimate the register pressure by the maximum number of values that, at
#       any point during execution, have yet to be used for their last time
#       (see https://llvm.org/devmtg/2014-10/Slides/Baev-Controlling_VRP.pdf).
struct LLVMInfo
    flags::Vector{String}
    branch_count::Int
    assignment_count::Int
    generic_julia_call_count::Int
    inferred_julia_call_counts::Dict{String, Int}
    recursive_julia_call_counts::Dict{String, Int}
    julia_stacktrace_strings::Dict{String, Vector{String}}
    llvm_call_counts::Dict{String, Int}
    other_call_counts::Dict{String, Int}
    int_instruction_counts::Dict{String, Int}
    float_instruction_counts::Dict{String, Int}
    conversion_instruction_counts::Dict{String, Int}
    struct_instruction_counts::Dict{String, Int}
    memory_instruction_counts::Dict{String, Int}
    control_flow_instruction_counts::Dict{String, Int}
    other_instruction_counts::Dict{String, Int}
end

first_match_group(string, regex) = first(eachmatch(regex, string))[1]
add_one!(dict, key) =
    if haskey(dict, key)
        dict[key] += 1
    else
        dict[key] = 1
    end
function llvm_info(io, f, T)
    flags = String[]
    branch_count = 0
    assignment_count = 0
    generic_julia_call_count = 0
    inferred_julia_call_counts = Dict{String, Int}()
    recursive_julia_call_counts = Dict{String, Int}()
    julia_stacktrace_strings = Dict{String, Vector{String}}()
    llvm_call_counts = Dict{String, Int}()
    other_call_counts = Dict{String, Int}()
    int_instruction_counts = Dict{String, Int}()
    float_instruction_counts = Dict{String, Int}()
    conversion_instruction_counts = Dict{String, Int}()
    struct_instruction_counts = Dict{String, Int}()
    memory_instruction_counts = Dict{String, Int}()
    control_flow_instruction_counts = Dict{String, Int}()
    other_instruction_counts = Dict{String, Int}()

    previous_stack_frame = nothing
    stack_frames = String[]

    code_llvm(io, f, T)
    seekstart(io)
    while !eof(io)
        line = readline(io)
        if startswith(line, "; Function Attrs: ")
            append!(flags, eachsplit(chopprefix(line, "; Function Attrs: ")))
            continue
        elseif startswith(line, "define ")
            next_line = readline(io)
            @assert startswith(next_line, r"\S+:") # often "top:" but not always
            continue
        elseif startswith(line, r"; │* @ \S+ within `.+`")
            previous_stack_frame =
                first_match_group(line, r"; │* @ \S+ within `(.+)`")
            continue
        elseif startswith(line, r"; │*┌ @ \S+ within `.+`")
            stack_frame = first_match_group(line, r"; │*┌ @ \S+ within `(.+)`")
            pushfirst!(stack_frames, stack_frame)
            continue
        elseif startswith(line, r"; │*└+")
            for _ in 1:count('└', line)
                popfirst!(stack_frames)
            end
            continue
        elseif line == ""
            next_line = readline(io)
            @assert startswith(next_line, r"\S+:\s+; preds = %")
            branch_count += 1
            continue
        elseif line == "}"
            next_line = readline(io)
            @assert eof(io)
            continue
        end

        if endswith(line, " [")
            next_line = readline(io)
            while startswith(next_line, "    ")
                line *= next_line
                next_line = readline(io)
            end
            @assert next_line == "  ]"
            line *= next_line
        end

        if startswith(line, r" \s+%\S+ = ")
            assignment_count += 1
            line = chopprefix(line, r" \s+%\S+ = ")
        else
            @assert startswith(line, r" \s+")
            line = chopprefix(line, r" \s+")
        end

        @assert startswith(line, r"\w+ ") || line == "unreachable"
        instruction = first(eachsplit(line))
        if instruction == "call"
            if startswith(line, r"call .+? @")
                line = chopprefix(line, r"call .+? @")
                @assert !startswith(line, r"j1_\S+_\d+\(") # What does j1 mean?
                if (
                    startswith(line, r"j_\S+_\d+\(") ||
                    startswith(line, r"j1_\S+_\d+\(") ||
                    startswith(line, r"\"j_\S+_\d+\"\(") ||
                    startswith(line, r"julia_\S+_\d+\(")
                )
                    if startswith(line, r"\"?j1?_(\S+)_\d+\"?\(")
                        f_string =
                            first_match_group(line, r"\"?j1?_(\S+)_\d+\"?\(")
                        add_one!(inferred_julia_call_counts, f_string)
                    else
                        f_string = first_match_group(line, r"julia_(\S+)_\d+\(")
                        @assert f_string == llvm_function_string(f)
                        add_one!(recursive_julia_call_counts, f_string)
                    end
                    stacktrace_string = if isempty(stack_frames)
                        previous_stack_frame
                    else
                        stacktrace = [
                            stack_frames[1],
                            previous_stack_frame,
                            stack_frames[3:end]...,
                        ]
                        join(stacktrace, ", ")
                    end
                    if haskey(julia_stacktrace_strings, f_string)
                        stacktrace_strings = julia_stacktrace_strings[f_string]
                        if !(stacktrace_string in stacktrace_strings)
                            push!(stacktrace_strings, stacktrace_string)
                        end
                    else
                        julia_stacktrace_strings[f_string] = [stacktrace_string]
                    end
                elseif startswith(line, r"llvm\.[\w\.]+\S*\(")
                    call_name = first_match_group(line, r"(llvm\.[\w\.]+)\S*\(")
                    add_one!(llvm_call_counts, call_name)
                else
                    @assert startswith(line, r"[\w_]+\(")
                    call_name = first_match_group(line, r"([\w_]+)\(")
                    add_one!(other_call_counts, call_name)
                end
            else
                generic_julia_call_count += 1
            end
        else
            if instruction in int_instructions
                add_one!(int_instruction_counts, instruction)
            elseif instruction in float_instructions
                add_one!(float_instruction_counts, instruction)
            elseif instruction in conversion_instructions
                add_one!(conversion_instruction_counts, instruction)
            elseif instruction in struct_instructions
                add_one!(struct_instruction_counts, instruction)
            elseif instruction in memory_instructions
                add_one!(memory_instruction_counts, instruction)
            elseif instruction in control_flow_instructions
                add_one!(control_flow_instruction_counts, instruction)
            else
                @assert instruction in other_instructions
                add_one!(other_instruction_counts, instruction)
            end
        end
    end

    # Reset io to a clean state, as is done at the end of the take! function.
    io.ptr = 1
    io.size = 0

    return LLVMInfo(
        flags,
        branch_count,
        assignment_count,
        generic_julia_call_count,
        inferred_julia_call_counts,
        recursive_julia_call_counts,
        julia_stacktrace_strings,
        llvm_call_counts,
        other_call_counts,
        int_instruction_counts,
        float_instruction_counts,
        conversion_instruction_counts,
        struct_instruction_counts,
        memory_instruction_counts,
        control_flow_instruction_counts,
        other_instruction_counts,
    )
end

################################################################################

@kwdef struct CallCounter{F}
    f_string::String # necessary when F is an UnknownFunctionType
    stacktrace_strings::Vector{String} # used to determine which calls to count
    f::Base.RefValue{F} = Ref{F}() # need a concrete instance of F for code_llvm
    invocation_counts::Dict{DataType, Tuple{Int, Int}} =
        Dict{DataType, Tuple{Int, Int}}()
end

function count_call!(counter, f, args...)
    T = args_type(args)
    is_root_invocation = counter.stacktrace_strings == [""]
    llvm_stacktrace = []
    for stack_frame in stacktrace()
        (; func) = stack_frame
        func in (:count_call!, :prehook, :overdub, :_apply) && continue
        contains(string(func), '(') && continue
        !isempty(llvm_stacktrace) && func == llvm_stacktrace[end] && continue
        push!(llvm_stacktrace, func)
    end
    stacktrace_string = join(llvm_stacktrace, ", ")
    has_matching_stacktrace =
        !is_root_invocation && any(counter.stacktrace_strings) do target_string
            startswith(stacktrace_string, target_string)
        end # this is always true when is_root_invocation is true
    has_recursively_matching_stacktrace =
        any(counter.stacktrace_strings) do target_string
            contains(stacktrace_string, target_string)
        end
    if haskey(counter.invocation_counts, T)
        @assert typeof(counter.f[]) == typeof(f)
        recursive_count_increment =
            Int(!has_matching_stacktrace && has_recursively_matching_stacktrace)
        counter.invocation_counts[T] =
            counter.invocation_counts[T] .+ (1, recursive_count_increment)
    elseif (
        (is_root_invocation && isempty(counter.invocation_counts)) ||
        has_matching_stacktrace
    )
        counter.f[] = f
        counter.invocation_counts[T] = (1, 0)
    end
end

Cassette.@context MethodCounterCtx
Cassette.prehook(
    ctx::MethodCounterCtx{CallCounter{F}},
    f::F,
    args...,
) where {F <: Function} = count_call!(ctx.metadata, f, args...)
Cassette.prehook(
    ctx::MethodCounterCtx{CallCounter{F}},
    f::Union{Type{F}, F}, # either F(args...) or f(args...) where f isa F
    args...,
) where {F} = count_call!(ctx.metadata, f, args...)
Cassette.prehook(ctx::MethodCounterCtx{CallCounter{Union{}}}, f, args...) =
    llvm_function_string(f) == ctx.metadata.f_string &&
    count_call!(ctx.metadata, f, args...)

function count_matching_calls(
    F,
    f_string,
    stacktrace_strings,
    outer_f,
    args...;
    expr_string = call_string(f, args...),
)
    counter = CallCounter{F}(; f_string, stacktrace_strings)
    ctx = MethodCounterCtx(; metadata = counter)
    Cassette.prehook(ctx, outer_f, args...)
    try
        Cassette.overdub(ctx, outer_f, args...)
    catch e
        @warn "Error during analysis of $expr_string: $e"
    end
    return isempty(ctx.metadata.invocation_counts) ? nothing :
           (ctx.metadata.f[], ctx.metadata.invocation_counts)
end

sorted_dict_pairs(dict) = sort(collect(dict); by = first)
function analyze_llvm(f, args...; expr_string = call_string(f, args...))
    llvm_summary =
        Dict{MethodAnalysis.MethodInstance, Tuple{Int, Int, LLVMInfo}}()

    io = IOBuffer()
    inferred_calls = Any[(typeof(f), llvm_function_string(f), [""])]
    analyzed_calls = []
    while !isempty(inferred_calls)
        F, f_string, stacktrace_strings = popfirst!(inferred_calls)
        call_info = count_matching_calls(
            F,
            f_string,
            stacktrace_strings,
            f,
            args...;
            expr_string,
        )
        isnothing(call_info) && continue
        new_f, invocation_counts = call_info
        for (T, (total_invocations, recursive_invocations)) in invocation_counts
            call = method_instance(new_f, T)
            call in analyzed_calls && continue
            call_llvm_info = llvm_info(io, new_f, T)
            llvm_summary[call] =
                (total_invocations, recursive_invocations, call_llvm_info)
            new_inferred_calls =
                map(collect(call_llvm_info.julia_stacktrace_strings)) do pair
                    f_string, stacktrace_strings = pair
                    F = inferred_function_type(f_string, call.def.module)
                    (F, f_string, stacktrace_strings)
                end
            append!(inferred_calls, new_inferred_calls)
            push!(analyzed_calls, call)
        end
    end

    close(io)

    println("@analyze_llvm $expr_string")
    println("  runtime calls: $(sum(first, values(llvm_summary)))")
    for call in analyzed_calls
        total_invocations, recursive_invocations, _ = llvm_summary[call]
        invocations_string = if recursive_invocations == 0
            "$total_invocations"
        else
            "$total_invocations ($recursive_invocations recursive)"
        end
        arg_type_strings = map(string, call.specTypes.parameters[2:end])
        signature_string =
            length(arg_type_strings) == 1 ? "(::$(arg_type_strings[1]),)" :
            "($(join(map(s -> "::" * s, arg_type_strings), ", ")))"
        file_string =
            startswith(string(call.def.file), "$(homedir())/") ?
            chopprefix(string(call.def.file), "$(homedir())/") :
            string(call.def.file)
        println("    $(call.def.name): $invocations_string")
        println("      module: $(call.def.module)")
        println("      signature: $signature_string")
        println("      source: $file_string:$(call.def.line)")
    end
    for call in analyzed_calls
        llvm_info = llvm_summary[call][3]
        println("  LLVM info for $(call.def.name)")
        if !isempty(llvm_info.flags)
            println("    flags: $(join(llvm_info.flags, ", "))")
        end
        if llvm_info.branch_count != 0
            println("    branches: $(llvm_info.branch_count)")
        end
        if llvm_info.assignment_count != 0
            println("    assignments: $(llvm_info.assignment_count)")
        end

        non_inlined_instruction_count =
            llvm_info.generic_julia_call_count +
            sum(values(llvm_info.inferred_julia_call_counts)) +
            sum(values(llvm_info.recursive_julia_call_counts)) +
            sum(values(llvm_info.llvm_call_counts)) +
            sum(values(llvm_info.other_call_counts))
        if non_inlined_instruction_count != 0
            println("    non-inlined instructions: \
                     $non_inlined_instruction_count")
            if llvm_info.generic_julia_call_count != 0
                println("      generic Julia calls: \
                         $(llvm_info.generic_julia_call_count)")
            end
            if !isempty(llvm_info.inferred_julia_call_counts)
                println("      inferred Julia calls: \
                         $(sum(values(llvm_info.inferred_julia_call_counts)))")
                inferred_julia_call_counts =
                    sorted_dict_pairs(llvm_info.inferred_julia_call_counts)
                for (f_string, count) in inferred_julia_call_counts
                    println("        $f_string: $count")
                end
            end
            if !isempty(llvm_info.recursive_julia_call_counts)
                println("      recursive Julia calls: \
                         $(sum(values(llvm_info.recursive_julia_call_counts)))")
                recursive_julia_call_counts =
                    sorted_dict_pairs(llvm_info.recursive_julia_call_counts)
                for (f_string, count) in recursive_julia_call_counts
                    println("        $f_string: $count")
                end
            end
            if !isempty(llvm_info.llvm_call_counts)
                println("      LLVM calls: \
                         $(sum(values(llvm_info.llvm_call_counts)))")
                llvm_call_counts = sorted_dict_pairs(llvm_info.llvm_call_counts)
                for (call_name, count) in llvm_call_counts
                    println("        $call_name: $count")
                end
            end
            if !isempty(llvm_info.other_call_counts)
                println("      other calls: \
                         $(sum(values(llvm_info.other_call_counts)))")
                other_call_counts =
                    sorted_dict_pairs(llvm_info.other_call_counts)
                for (call_name, count) in other_call_counts
                    println("        $call_name: $count")
                end
            end
        end

        inlined_instruction_count =
            sum(values(llvm_info.int_instruction_counts)) +
            sum(values(llvm_info.float_instruction_counts)) +
            sum(values(llvm_info.conversion_instruction_counts)) +
            sum(values(llvm_info.struct_instruction_counts)) +
            sum(values(llvm_info.memory_instruction_counts)) +
            sum(values(llvm_info.control_flow_instruction_counts)) +
            sum(values(llvm_info.other_instruction_counts))
        @assert inlined_instruction_count != 0 # there's always at least 1 "ret"
        println("    inlined instructions: $inlined_instruction_count")
        if !isempty(llvm_info.int_instruction_counts)
            println("      int instructions: \
                     $(sum(values(llvm_info.int_instruction_counts)))")
            int_instruction_counts =
                sorted_dict_pairs(llvm_info.int_instruction_counts)
            for (instruction, count) in int_instruction_counts
                println("        $instruction: $count")
            end
        end
        if !isempty(llvm_info.float_instruction_counts)
            println("      float instructions: \
                     $(sum(values(llvm_info.float_instruction_counts)))")
            float_instruction_counts =
                sorted_dict_pairs(llvm_info.float_instruction_counts)
            for (instruction, count) in float_instruction_counts
                println("        $instruction: $count")
            end
        end
        if !isempty(llvm_info.conversion_instruction_counts)
            println("      conversion instructions: \
                     $(sum(values(llvm_info.conversion_instruction_counts)))")
            conversion_instruction_counts =
                sorted_dict_pairs(llvm_info.conversion_instruction_counts)
            for (instruction, count) in conversion_instruction_counts
                println("        $instruction: $count")
            end
        end
        if !isempty(llvm_info.struct_instruction_counts)
            println("      struct instructions: \
                     $(sum(values(llvm_info.struct_instruction_counts)))")
            struct_instruction_counts =
                sorted_dict_pairs(llvm_info.struct_instruction_counts)
            for (instruction, count) in struct_instruction_counts
                println("        $instruction: $count")
            end
        end
        if !isempty(llvm_info.memory_instruction_counts)
            println("      memory instructions: \
                     $(sum(values(llvm_info.memory_instruction_counts)))")
            memory_instruction_counts =
                sorted_dict_pairs(llvm_info.memory_instruction_counts)
            for (instruction, count) in memory_instruction_counts
                println("        $instruction: $count")
            end
        end
        if !isempty(llvm_info.control_flow_instruction_counts)
            println("      control flow instructions: \
                     $(sum(values(llvm_info.control_flow_instruction_counts)))")
            control_flow_instruction_counts =
                sorted_dict_pairs(llvm_info.control_flow_instruction_counts)
            for (instruction, count) in control_flow_instruction_counts
                println("        $instruction: $count")
            end
        end
        if !isempty(llvm_info.other_instruction_counts)
            println("      other instructions: \
                     $(sum(values(llvm_info.other_instruction_counts)))")
            other_instruction_counts =
                sorted_dict_pairs(llvm_info.other_instruction_counts)
            for (instruction, count) in other_instruction_counts
                println("        $instruction: $count")
            end
        end
    end
    println()

    return llvm_summary
end

################################################################################

call_expression_components(expr) =
    expr.head == :call ? (expr.args[1], expr.args[2:end]) :
    error("Expression is not a function call: $expr")
expression_string(expr) = replace(
    string(expr),
    r"\s*#=.+=# @__dot__" => "@.",
    r"\s*#=.+=# @" => '@',
    r"\s*#=.+=#\s+" => ' ',
    r"\s+" => ' ',
)
macro count_unoptimized(expr)
    f_expr, arg_exprs = call_expression_components(expr)
    expr_string = expression_string(expr)
    :(count_unoptimized($f_expr, $(arg_exprs...); expr_string = $expr_string),)
end
macro analyze_llvm(expr)
    f_expr, arg_exprs = call_expression_components(expr)
    expr_string = expression_string(expr)
    :(analyze_llvm($f_expr, $(arg_exprs...); expr_string = $expr_string))
end

################################################################################

analyze_llvm((x, y) -> sum((x, y)), 1, 2)
analyze_llvm((x, y) -> sum([x, y]), 1, 2)

@analyze_llvm 2^0.5

using StaticArrays: @SArray, @MArray

@analyze_llvm sum((1:32...,))
@analyze_llvm sum(@SArray([1:32...]))
@analyze_llvm sum(@MArray([1:32...]))
@analyze_llvm sum([1:32...])
@analyze_llvm sum([1:1025...])

@count_unoptimized sum([1:32...])
@count_unoptimized sum([1:1025...])

import ClimaCore.Utilities.UnrolledFunctions: unrolled_map

@analyze_llvm map(x -> x + 1, ntuple(_ -> 0, Val(32)))
@analyze_llvm unrolled_map(x -> x + 1, ntuple(_ -> 0, Val(32)))
@analyze_llvm unrolled_map(_ -> 2 // 1 == 2 ? 1 : 2, ntuple(_ -> 0, Val(32)))

@count_unoptimized unrolled_map(x -> x + 1, ntuple(_ -> 0, Val(32)))
@count_unoptimized unrolled_map(
    _ -> 2 // 1 == 2 ? 1 : 2,
    ntuple(_ -> 0, Val(32)),
)

import StaticArrays: @SMatrix
import ClimaCore: Geometry
import ClimaCore.MatrixFields: rmul_with_projection, rmul_return_type

FT = Float64
coord = Geometry.LatLongZPoint(rand(FT), rand(FT), rand(FT))
∂x∂ξ = Geometry.AxisTensor(
    (Geometry.LocalAxis{(1, 2, 3)}(), Geometry.CovariantAxis{(1, 2, 3)}()),
    (@SMatrix rand(FT, 3, 3)),
)
lg = Geometry.LocalGeometry(coord, rand(FT), rand(FT), ∂x∂ξ)
vector = Geometry.Covariant123Vector(rand(FT), rand(FT), rand(FT))
covector = Geometry.Covariant12Vector(rand(FT), rand(FT))'
tensor = vector * covector

@analyze_llvm rmul_return_type(typeof(covector), typeof(tensor))
@analyze_llvm rmul_with_projection(covector, tensor, lg)

fib(n) = n < 2 ? 1 : fib(n - 2) + fib(n - 1)
@analyze_llvm fib(3)

struct Fib{T} end
(fib::Fib{T})(n) where {T} = n < 2 ? T(1) : fib(n - 2) + fib(n - 1)
@analyze_llvm (Fib{Float64}())(3)

@analyze_llvm println("Test")

@analyze_llvm 0 // 0
@analyze_llvm Int(1 // 0)
@analyze_llvm sqrt(-1)

function do_stuff(x)
    do_thing(y) = y < 2 ? 1 : do_stuff(y - 1) + x
    return do_thing(x - 1) + do_thing(x - 2)
end
@analyze_llvm do_stuff(4)

@noinline foo(x) = x^x / x
blah(x) = foo.(x)
@analyze_llvm blah([1, 2, 3])

# TODO: Fix these
@analyze_llvm ntuple(Fib{Float64}(), Val(3))
@analyze_llvm ntuple(Fib{Float64}(), Val(4))
@analyze_llvm unrolled_map(
    f -> ntuple(f, Val(4)),
    (Fib{Float32}(), Fib{Float64}()),
)
