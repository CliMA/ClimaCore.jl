using ClimaCore, CUDA, LinearAlgebra
import ClimaCore: Operators, Fields, Geometry, Grids, Meshes

function setup_context()
    domain = Geometry.SphereDomain(6.37122e6)
    mesh = Meshes.EquiangularCubedSphere(domain, 4)
    grid_obj = Grids.LevelGrid(mesh, 4)

    # State: Scalar (rho), Velocity (u), and a Weight field (w)
    ftype = NamedTuple{(:rho, :u, :w), NTuple{3, Float64}}
    field = Fields.zeros(ftype, grid_obj)
    return field
end

function apply_ops(field, depth, mode)
    if depth <= 0
        return field
    end

    # --- Operators ---
    I = Operators.InterpolateC2F()
    # Upwinding requires a "velocity" or direction
    U = Operators.UpwindBiasedProductC2F()
    # Weighted interpolation uses a third field as a weight
    WI = Operators.WeightedInterpolateC2F()

    prev = apply_ops(field, depth - 1, mode)

    if mode == :upwind
        # Upwinding increases register pressure by tracking 'upwind' and 'downwind' states
        return @. U(field.u, prev.rho)
    elseif mode == :weighted
        # WeightedInterpolate requires more registers to store the weight intermediate
        return @. WI(field.w, prev.rho)
    elseif mode == :clima_chaos
        # Combining everything: The "Ultimate Stressor"
        # Nested Upwinding inside a Weighted Interpolation
        return @. U(field.u, WI(field.w, prev.rho))
    end
end

function get_register_count(ptx_string)
    m = match(r"\.reg\s+\.\w+\s+%\w+<(\d+)>", ptx_string)
    return m === nothing ? "N/A" : m.captures[1]
end

function run_comprehensive_analysis()
    field = setup_context()
    modes = [:upwind, :weighted, :clima_chaos]

    println(rpad("Mode", 15), "| Depth | Registers | Time (s) | Status")
    println("-"^55)

    for mode in modes
        for d in 1:10
            # We add a random float to the expression to 'poison' the cache
            # This ensures the compiler sees a unique expression every time
            poison = rand()

            t_start = time()
            reg_count = "Error"
            status = "Success"

            try
                ptx = CUDA.code_ptx() do
                    # Adding poison forces a re-compile if you were to run this twice
                    res = @. apply_ops(field, d, mode) + poison
                    CUDA.@sync res
                end
                reg_count = get_register_count(ptx)
            catch e
                status = "Compiler Fail"
                reg_count = ">255"
            end

            elapsed = round(time() - t_start, digits = 2)
            println(
                rpad(mode, 15),
                " | ",
                rpad(d, 5),
                " | ",
                rpad(reg_count, 9),
                " | ",
                rpad(elapsed, 8),
                " | ",
                status,
            )

            if status != "Success"
                break
            end
        end
    end
end

run_comprehensive_analysis()
