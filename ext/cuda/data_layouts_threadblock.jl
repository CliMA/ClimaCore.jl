##### linear partition
@inline function linear_partition(nitems::Integer, n_max_threads::Integer)
    @assert nitems > 0
    threads = min(nitems, n_max_threads)
    blocks = cld(nitems, threads)
    return (; threads, blocks)
end
