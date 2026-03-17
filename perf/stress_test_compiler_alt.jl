using ClimaCore, CUDA, LinearAlgebra
import ClimaCore: Operators, Fields, Geometry, Grids, Meshes

# --- Context Setup ---
function setup_context()
    domain = Geometry.SphereDomain(6.37122e6)
    mesh = Meshes.EquiangularCubedSphere(domain, 4)
    grid_obj = Grids.LevelGrid(mesh, 4)
    ftype = NamedTuple{
        (:s, :v, :w),
        Tuple{Float64, Geometry.Cartesian123Vector{Float64}, Float64},
    }
    return Fields.zeros(ftype, grid_obj)
end

# --- Individual Operation Factories ---
# These functions wrap a single type of operation to isolate its cost
ops = Dict(
    :arithmetic => (p, f) -> p + f.s * 0.1,
    :log_exp => (p, f) -> log(abs(p) + 1.1) + exp(-abs(p)),
    :interp => (p, f) -> Operators.InterpolateC2F()(p),
    :gradient => (p, f) -> Operators.GradientC2F()(p),
    :div => (p, f) -> Operators.DivergenceF2C()(f.v) * p, # DIV returns scalar
    :curl => (p, f) -> Operators.CurlC2F()(f.v),           # Note: Result is Vector
    :upwind => (p, f) -> Operators.UpwindBiasedProductC2F()(f.s, p),
    :weighted => (p, f) -> Operators.WeightedInterpolateC2F()(f.w, p),
)

function build_expr(field, depth, op_key)
    if depth <= 0
        return field.s
    end
    prev = build_expr(field, depth - 1, op_key)
    return ops[op_key](prev, field)
end

# --- Analysis Logic ---
function analyze_ptx(ptx_string)
    reg_match = match(r"\.reg\s+\.\w+\s+%\w+<(\d+)>", ptx_string)
    local_match = match(r"\.local\s+\.align\s+\d+\s+\.\w+\s+\w+\[(\d+)\]", ptx_string)
    return (reg_match === nothing ? "0" : reg_match.captures[1],
        local_match === nothing ? "0" : local_match.captures[1])
end

function run_isolated_thresholds()
    field = setup_context()
    # Ordered from simple to complex
    categories =
        [:arithmetic, :log_exp, :interp, :gradient, :upwind, :weighted, :div, :curl]

    println(rpad("Operation", 15), "| Max Depth | Last Regs | Last Stack | Failure Mode")
    println("-"^70)

    for op in categories
        max_depth = 0
        last_regs = "0"
        last_stack = "0"
        failure_reason = "Inference Limit"

        for d in 1:50 # High limit to find the absolute ceiling
            try
                ptx = CUDA.code_ptx() do
                    res = build_expr(field, d, op)
                    CUDA.@sync res
                end
                regs, stack = analyze_ptx(ptx)

                # If we see a spill, we mark it but keep going to see hard failure
                if parse(Int, stack) > 0 && last_stack == "0"
                    # First time spilling
                end

                max_depth = d
                last_regs = regs
                last_stack = stack

                # If we hit the 255 register wall, it's effectively a hardware failure
                if parse(Int, regs) >= 255 && parse(Int, stack) > 1024
                    failure_reason = "Register Spill"
                    break
                end
            catch e
                break
            end
        end
        println(
            rpad(op, 15), " | ",
            rpad(max_depth, 9), " | ",
            rpad(last_regs, 9), " | ",
            rpad(last_stack, 10), " | ",
            failure_reason,
        )
    end
end

run_isolated_thresholds()
