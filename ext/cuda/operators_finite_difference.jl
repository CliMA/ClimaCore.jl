import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore: Operators
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: setidx!, getidx
import ClimaCore.Operators: StencilBroadcasted
import ClimaCore.Operators: LeftBoundaryWindow, RightBoundaryWindow, Interior
import ClimaCore.MatrixFields: FaceToCenter, CenterToFace, Square
using UnrolledUtilities
using Statistics


struct CUDAColumnStencilStyle <: AbstractStencilStyle end
struct CUDAWithShmemColumnStencilStyle <: AbstractStencilStyle end

AbstractStencilStyle(bc, ::ClimaComms.CUDADevice) = CUDAColumnStencilStyle

Base.Broadcast.BroadcastStyle(
    x::Operators.ColumnStencilStyle,
    y::CUDAColumnStencilStyle,
) = y

include("operators_fd_shmem_is_supported.jl")
include("newmm.jl")
# include("exps.jl")
# include("matmul.jl")
struct ShmemParams{Nv} end
interior_size(::ShmemParams{Nv}) where {Nv} = (Nv,)
boundary_size(::ShmemParams{Nv}) where {Nv} = (1,)

function call_tree(bc::Broadcasted)
    return string(nameof(bc.f)) * "(" * join(map(call_tree, bc.args), ", ") * ")"
end

function call_tree(bc::StencilBroadcasted)
    return string(nameof(typeof(bc.op))) * "(" * join(map(call_tree, bc.args), ", ") * ")"
end
summary_string(entry) = summary_string(entry, 0)
summary_string(entry, indent_level) = "$("    "^indent_level)$entry"
function summary_string(field::Fields.Field, indent_level)
    staggering_string =
        hasproperty(axes(field), :staggering) ?
        string(typeof(axes(field).staggering).name.name) : "Single Level"
    return "$("    "^indent_level)Field{$(eltype(field)), $staggering_string}"
end
function summary_string(bc::Base.AbstractBroadcasted, indent_level)
    func = bc isa Operators.OperatorBroadcasted ? bc.op : bc.f
    bc isa Operators.OperatorBroadcasted || @info methods(func)[1]
    arg_strings = map(arg -> summary_string(arg, indent_level + 1), bc.args)
    tab = "    "^indent_level
    return "$(tab)Broadcasted{$func}(\n$(join(arg_strings, ",\n")),\n$tab)"
end


function print_datatypes(T)
    if fieldcount(T) == 0
        # println(T)
         return
    end
    for t_i in 1:fieldcount(T)
        t = fieldtype(T, t_i)
        println(fieldname(T, t_i), ": ", t, " (", sizeof(t), " bytes)")
        println("")
        print_datatypes(t)
    end
    println("\n")
end

call_tree(x) = string(nameof(eltype(x)))
function Base.copyto!(
    out::Field,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAWithShmemColumnStencilStyle},
    },
    mask = Spaces.get_mask(axes(out)),
)
    space = axes(out)
    bounds = Operators.window_bounds(space, bc)
    # @show bounds
    out_fv = Fields.field_values(out)
    us = DataLayouts.UniversalSize(out_fv)

    fspace = Spaces.face_space(space)
    n_face_levels = Spaces.nlevels(fspace)
    high_resolution = !(n_face_levels ≤ 256)
    # https://github.com/JuliaGPU/CUDA.jl/issues/2672
    # max_shmem = 166912 # CUDA.limit(CUDA.LIMIT_SHMEM_SIZE) #
    max_shmem = CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    )
    total_shmem = fd_shmem_needed_per_column(bc)
    enough_shmem = total_shmem ≤ max_shmem

    # TODO: Use CUDA.limit(CUDA.LIMIT_SHMEM_SIZE) to determine how much shmem should be used
    # TODO: add shmem support for masked operations
    if Operators.any_fd_shmem_supported(bc) &&
       !high_resolution &&
       mask isa NoMask &&
       enough_shmem &&
       Operators.use_fd_shmem()
        shmem_params = ShmemParams{n_face_levels}()
        p = fd_shmem_stencil_partition(us, n_face_levels)
        args = (
            strip_space(out, space),
            strip_space(bc, space),
            axes(out),
            bounds,
            us,
            mask,
            shmem_params,
        )
        auto_launch!(
            copyto_stencil_kernel_shmem!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    else
        bc′ = disable_shmem_style(bc)
        (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(out_fv)
        #  Specialized kernel launch for common case.  This uses block and grid indices
        # instead of computing cartesian indices from a linear index
        if true && (Nv == 64 || Nv == 63) && mask isa NoMask && Ni == 4 && Nj == 4 && Nh >= 1500# && !has_lin_vanleer(bc′)
             if true && mask isa NoMask #&& !(bc′ isa Broadcasted && bc′.f == identity)# && eltype(out) <: ClimaCore.MatrixFields.BandMatrixRow && typeof(bc′) <: Union{ClimaCore.Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle, ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField}, Base.Broadcast.Broadcasted}#&& length(bc′.args) == 2 && bc′ isa StencilBroadcasted && bc′.op isa ClimaCore.MatrixFields.MultiplyColumnwiseBandMatrixField && length(bc′.args) == 2
                new_bc = recursively_replace_fd_ops(bc′)
                args = (
                    strip_space(out, space),
                    strip_space(new_bc, space),
                    axes(out),
                )
                # converted_args = map(CUDA.cudaconvert, args)
                # @show sizeof(typeof(converted_args))
                # print_datatypes(typeof(converted_args[2]))
                # println(summary_string(new_bc))
                # println("\n")
                # println(summary_string(bc′))
                mykr = CUDA.@cuda  always_inline = true launch=false fastmath=true new_stencil_entry!(args...)
                # println("\n")
                mykr(args...; threads = (64, 1, 1), blocks = (4, 4, Nh), shmem = 64*9*4 )
                
                # @show CUDA.memory(mykr)
                #  @show CUDA.registers(mykr)
                #  println(summary_string(strip_space(bc′, space)))
                if CUDA.memory(mykr).local > 100
                    @show  CUDA.memory(mykr).local
                    println(summary_string(strip_space(new_bc, space)))
                end
                # println("\n\n")
                #  && println(summary_string(strip_space(bc′, space)))
                # if CUDA.memory(mykr).local < 100
                #     # @show CUDA.memory(mykr).local
                #     # @show CUDA.memory(mykr)
                #     # CUDA.@cuda  always_inline = true launch=true fastmath=true threads = (64, 1, 1) blocks = (4, 4, Nh) shmem = 64*8*4 new_stencil_entry!(args...)
                #     # mykr(args...; threads = (64, 1, 1), blocks = (4, 4, Nh), shmem = 64*9*4 )
                #     # mykr(args...; threads = (64, 1, 1), blocks = (4, 4, Nh))
                    return out
                # end
             end

            args = (
                strip_space(out, space),
                strip_space(bc′, space),
                axes(out),
                bounds,
                Val(Nv == 63),
            )
            auto_launch!(
                copyto_stencil_kernel_64!,
                args;
                threads_s = (64, 1, 1),
                blocks_s = (Ni, Nj, Nh),
            )
            return out
        end
        @assert !any_fd_shmem_style(bc′)
        cart_inds = if mask isa NoMask
            cartesian_indices(us)
        else
            cartesian_indicies_mask(us, mask)
        end
        args = (
            strip_space(out, space),
            strip_space(bc′, space),
            axes(out),
            bounds,
            us,
            mask,
            cart_inds,
        )

        threads = threads_via_occupancy(copyto_stencil_kernel!, args)
        n_max_threads = min(threads, get_N(us))
        p = if mask isa NoMask
            linear_partition(prod(size(out_fv)), n_max_threads)
        else
            masked_partition(mask, n_max_threads, us)
        end
        auto_launch!(
            copyto_stencil_kernel!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    end
    call_post_op_callback() && post_op_callback(out, out, bc)
    return out
end
import ClimaCore.DataLayouts: get_N, get_Nv, get_Nij, get_Nij, get_Nh

function has_lin_vanleer(bc::Broadcasted)
    return UnrolledUtilities.unrolled_any(has_lin_vanleer, bc.args)
end

function has_lin_vanleer(bc::StencilBroadcasted)
    bc.op isa Operators.LinVanLeerC2F && return true
    return unrolled_any(has_lin_vanleer, bc.args)
end
has_lin_vanleer(_) = false
"""
    copyto_stencil_kernel_64!(
        out,
        bc::Union{
            StencilBroadcasted{CUDAColumnStencilStyle},
            Broadcasted{CUDAColumnStencilStyle},
        },
        space,
        bds,
        ::Val{P},
    )

Kernel for fd operators on VIJFHStyle{63,4} and VIJFHStyle{64,4} datalayouts. P is a boolean
indicating if the column is padded (true for 63, false for 64).
"""
function copyto_stencil_kernel_64!(
    out,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
    },
    space,
    bds,
    ::Val{P},
) where {P}
    @inbounds begin
        # P is a boolean, indicating if the column is padded
        P && threadIdx().x == 64 && return nothing
        i = blockIdx().x
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        (li, lw, rw, ri) = bds
        idx = v - 1 + li
        val = Operators.getidx(space, bc, idx, hidx)
        setidx!(space, out, idx, hidx, val)
    end
    return nothing
end

function copyto_stencil_kernel!(
    out,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
    },
    space,
    bds,
    us,
    mask,
    cart_inds,
)
    @inbounds begin
        out_fv = Fields.field_values(out)
        tidx = linear_thread_idx()
        if linear_is_valid_index(tidx, us) && tidx ≤ length(unval(cart_inds))
            I = if mask isa NoMask
                unval(cart_inds)[tidx]
            else
                masked_universal_index(mask, cart_inds)
            end
            (li, lw, rw, ri) = bds
            (i, j, _, v, h) = I.I
            hidx = (i, j, h)
            idx = v - 1 + li
            val = Operators.getidx(space, bc, idx, hidx)
            setidx!(space, out, idx, hidx, val)
        end
    end
    return nothing
end

function copyto_stencil_kernel_shmem!(
    out,
    bc′::Union{StencilBroadcasted, Broadcasted},
    space,
    bds,
    us,
    mask,
    shmem_params::ShmemParams,
)
    @inbounds begin
        out_fv = Fields.field_values(out)
        us = DataLayouts.UniversalSize(out_fv)
        I = fd_shmem_stencil_universal_index(space, us)
        if fd_shmem_stencil_is_valid_index(I, us) # check that hidx is in bounds
            (li, lw, rw, ri) = bds
            (i, j, _, v, h) = I.I
            hidx = (i, j, h)
            idx = v - 1 + li
            bc = Operators.reconstruct_placeholder_broadcasted(space, bc′)
            bc_shmem = fd_allocate_shmem(shmem_params, bc) # allocates shmem

            fd_resolve_shmem!(bc_shmem, idx, hidx, bds) # recursively fills shmem
            CUDA.sync_threads()

            nv = Spaces.nlevels(space)
            isactive = if space isa Operators.AllFaceFiniteDifferenceSpace # check that idx is in bounds
                idx + half <= nv
            else
                idx <= nv
            end
            if isactive
                # Call getidx overloaded in operators_fd_shmem_common.jl
                val = Operators.getidx(space, bc_shmem, idx, hidx)
                setidx!(space, out, idx, hidx, val)
            end
        end
    end
    return nothing
end
