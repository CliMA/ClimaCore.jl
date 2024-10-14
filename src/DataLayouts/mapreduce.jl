# This is only defined for testing.
function mapreduce_cuda end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::BroadcastedUnionDataF{<:Any, A},
) where {F, Op, A}
    mapreduce(op, 1) do v
        Base.@_inline_meta
        @inbounds fn(bc[])
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::BroadcastedUnionIJFH{<:Any, Nij, A},
) where {F, Op, Nij, A}
    # mapreduce across DataSlab2D
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, 1:Nh) do h
        Base.@_inline_meta
        slabview = @inbounds slab(bc, h)
        mapreduce(fn, op, slabview)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::BroadcastedUnionIFH{<:Any, Ni, A},
) where {F, Op, Ni, A}
    # mapreduce across DataSlab1D
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, 1:Nh) do h
        Base.@_inline_meta
        slabview = @inbounds slab(bc, h)
        mapreduce(fn, op, slabview)
    end
end

function Base.mapreduce(fn::F, op::Op, bc::IJF{S, Nij}) where {F, Op, S, Nij}
    # mapreduce across DataSlab2D nodes
    mapreduce(op, Iterators.product(1:Nij, 1:Nij)) do (i, j)
        Base.@_inline_meta
        idx = CartesianIndex(i, j, 1, 1, 1)
        node = @inbounds bc[idx]
        fn(node)
    end
end

function Base.mapreduce(fn::F, op::Op, bc::IF{S, Ni}) where {F, Op, S, Ni}
    # mapreduce across DataSlab1D nodes
    mapreduce(op, 1:Ni) do i
        Base.@_inline_meta
        idx = CartesianIndex(i, 1, 1, 1, 1)
        node = @inbounds bc[idx]
        fn(node)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::BroadcastedUnionVF{<:Any, Nv, A},
) where {F, Op, Nv, A}
    # mapreduce across DataColumn levels
    mapreduce(op, 1:Nv) do v
        Base.@_inline_meta
        idx = CartesianIndex(1, 1, 1, v, 1)
        level = @inbounds bc[idx]
        fn(level)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::BroadcastedUnionVIFH{<:Any, Nv, Ni, A},
) where {F, Op, Nv, Ni, A}
    # mapreduce across columns
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, Iterators.product(1:Ni, 1:Nh)) do (i, h)
        Base.@_inline_meta
        columnview = @inbounds column(bc, i, h)
        mapreduce(fn, op, columnview)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::BroadcastedUnionVIJFH{<:Any, Nv, Nij, A},
) where {F, Op, Nv, Nij, A}
    # mapreduce across columns
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, Iterators.product(1:Nij, 1:Nij, 1:Nh)) do (i, j, h)
        Base.@_inline_meta
        columnview = @inbounds column(bc, i, j, h)
        mapreduce(fn, op, columnview)
    end
end
