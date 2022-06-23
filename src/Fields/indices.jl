
struct SlabIndex{VIdx}
    v::VIdx
    h::Int
end




Base.getindex(field::Field, slabidx::SlabIndex) = slab(field, slabidx)

function slab(field::SpectralElementField, slabidx::SlabIndex{Nothing})
    slab(field, slabidx.h)
end
function slab(
    field::CenterExtrudedFiniteDifferenceField,
    colidx::SlabIndex{Int},
)
    slab(field, slabidx.v, slabidx.h)
end
function slab(
    field::FaceExtrudedFiniteDifferenceField,
    colidx::SlabIndex{PlusHalf{Int}},
)
    slab(field, slabidx.v + half, slabidx.h)
end

function byslab(fn, space::Spaces.AbstractSpectralElementSpace)
    Nh = Topologies.nlocalelems(space.topology)::Int
    Threads.@threads for h in 1:Nh
        fn(SlabIndex(nothing, h))
    end
end
function byslab(fn, space::Spaces.CenterExtrudedFiniteDifferenceSpace)
    Nh = Topologies.nlocalelems(topology)::Int
    Nv = Spaces.nlevels(space)::Int
    Threads.@threads for h in 1:Nh
        for v in 1:Nv
            fn(SlabIndex(v, h))
        end
    end
end
function byslab(fn, space::Spaces.FaceExtrudedFiniteDifferenceSpace)
    Nh = Topologies.nlocalelems(topology)::Int
    Nv = Spaces.nlevels(space)::Int
    Threads.@threads for h in 1:Nh
        for v in 1:Nv
            fn(SlabIndex(v - half, h))
        end
    end
end
