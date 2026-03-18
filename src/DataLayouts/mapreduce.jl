# This is only defined for testing.
function mapreduce_cuda end

Base.mapreduce(
    fn::F,
    op::Op,
    data::Union{AbstractData, Base.Broadcast.Broadcasted{<:DataStyle}},
) where {F, Op} =
    disable_auto_broadcasting(mapreduce_data(fn, op, Base.broadcastable(data)))

function mapreduce_data(
    fn::F,
    op::Op,
    bc::BroadcastedUnionDataF{<:Any, A},
) where {F, Op, A}
    @inbounds fn(bc[])
end

function mapreduce_data(
    fn::F,
    op::Op,
    bc::Union{
        BroadcastedUnionIJFH{<:Any, Nij, A},
        BroadcastedUnionIJHF{<:Any, Nij, A},
    },
) where {F, Op, Nij, A}
    # mapreduce across DataSlab2D
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, 1:Nh) do h
        Base.@_inline_meta
        slabview = @inbounds slab(bc, h)
        mapreduce_data(fn, op, slabview)
    end
end

function mapreduce_data(
    fn::F,
    op::Op,
    bc::Union{
        BroadcastedUnionIFH{<:Any, Ni, A},
        BroadcastedUnionIHF{<:Any, Ni, A},
    },
) where {F, Op, Ni, A}
    # mapreduce across DataSlab1D
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, 1:Nh) do h
        Base.@_inline_meta
        slabview = @inbounds slab(bc, h)
        mapreduce_data(fn, op, slabview)
    end
end

function mapreduce_data(
    fn::F,
    op::Op,
    bc::BroadcastedUnionIJF{<:Any, Nij, A},
) where {F, Op, Nij, A}
    # mapreduce across DataSlab2D nodes
    mapreduce(op, Iterators.product(1:Nij, 1:Nij)) do (i, j)
        Base.@_inline_meta
        idx = CartesianIndex(i, j, 1, 1, 1)
        node = @inbounds bc[idx]
        fn(node)
    end
end

function mapreduce_data(
    fn::F,
    op::Op,
    bc::BroadcastedUnionIF{<:Any, Ni, A},
) where {F, Op, Ni, A}
    # mapreduce across DataSlab1D nodes
    mapreduce(op, 1:Ni) do i
        Base.@_inline_meta
        idx = CartesianIndex(i, 1, 1, 1, 1)
        node = @inbounds bc[idx]
        fn(node)
    end
end

function mapreduce_data(
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

function mapreduce_data(
    fn::F,
    op::Op,
    bc::Union{
        BroadcastedUnionVIFH{<:Any, Nv, Ni, A},
        BroadcastedUnionVIHF{<:Any, Nv, Ni, A},
    },
) where {F, Op, Nv, Ni, A}
    # mapreduce across columns
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, Iterators.product(1:Ni, 1:Nh)) do (i, h)
        Base.@_inline_meta
        columnview = @inbounds column(bc, i, h)
        mapreduce_data(fn, op, columnview)
    end
end

function mapreduce_data(
    fn::F,
    op::Op,
    bc::Union{
        BroadcastedUnionVIJFH{<:Any, Nv, Nij, A},
        BroadcastedUnionVIJHF{<:Any, Nv, Nij, A},
    },
) where {F, Op, Nv, Nij, A}
    # mapreduce across columns
    (_, _, _, _, Nh) = size(bc)
    mapreduce(op, Iterators.product(1:Nij, 1:Nij, 1:Nh)) do (i, j, h)
        Base.@_inline_meta
        columnview = @inbounds column(bc, i, j, h)
        mapreduce_data(fn, op, columnview)
    end
end
