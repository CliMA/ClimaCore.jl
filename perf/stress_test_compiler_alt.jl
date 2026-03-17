using ClimaCore, CUDA, LinearAlgebra
import ClimaCore: Operators, Fields, Geometry, Grids, Meshes

# --- Setup: Minimal ClimaCore Context ---
function setup_context()
    domain = Geometry.SphereDomain(6.37122e6)
    mesh = Meshes.EquiangularCubedSphere(domain, 4)
    grid_obj = Grids.LevelGrid(mesh, 4)
    ftype = NamedTuple{(:rho, :u, :w), NTuple{3, Float64}}
    return Fields.zeros(ftype, grid_obj)
end

# --- Expression Factory ---
function apply_ops(field, depth, mode)
    if depth <= 0
        return field
    end
    U = Operators.UpwindBiasedProductC2F()
    WI = Operators.WeightedInterpolateC2F()
    prev = apply_ops(field, depth - 1, mode)

    if mode == :upwind
        return @. U(field.u, prev.rho)
    elseif mode == :weighted
        return @. WI(field.w, prev.rho)
    elseif mode == :clima_chaos
        # Combining non-local operators with math
        return @. U(field.u, WI(field.w, prev.rho)) + sin(prev.rho)
    end
end

# --- PTX Introspection ---
function analyze_ptx(ptx_string)
    # 1. Extract Register Count
    reg_match = match(r"\.reg\s+\.\w+\s+%\w+<(\d+)>", ptx_string)
    regs = reg_match === nothing ? "0" : reg_match.captures[1]

    # 2. Extract Local (Stack) Memory Usage
    # Look for ".local .align 8 .b8 __local_depot0[128];"
    local_match = match(r"\.local\s+\.align\s+\d+\s+\.\w+\s+\w+\[(\d+)\]", ptx_string)
    stack_bytes = local_match === nothing ? "0" : local_match.captures[1]

    return regs, stack_bytes
end

# --- The Analysis Loop ---
function run_deep_analysis()
    field = setup_context()
    modes = [:upwind, :clima_chaos]

    println(rpad("Mode", 12), "| Depth | Regs | Stack (B) | Time (s) | Status")
    println("-"^60)

    for mode in modes
        for d in 1:15
            # Invalidate compiler cache by generating a unique type
            # We use a wrapper with a unique integer to force a fresh compile
            t_start = time()
            regs, stack = "0", "0"
            status = "Success"

            try
                ptx = CUDA.code_ptx() do
                    # The broadcast kernel we want to inspect
                    res = apply_ops(field, d, mode)
                    CUDA.@sync res
                end
                regs, stack = analyze_ptx(ptx)
            catch e
                status = "COMPILER FAIL"
            end

            elapsed = round(time() - t_start, digits = 2)

            # Highlight if spilling is occurring
            stack_display = parse(Int, stack) > 0 ? "! $stack !" : stack

            println(
                rpad(mode, 12), " | ",
                rpad(d, 5), " | ",
                rpad(regs, 4), " | ",
                rpad(stack_display, 9), " | ",
                rpad(elapsed, 8), " | ",
                status,
            )

            if status != "Success"
                break
            end
        end
    end
end

run_deep_analysis()
