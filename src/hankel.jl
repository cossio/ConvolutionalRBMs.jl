@doc raw"""
    hankel(img, channel_size, kernel_size)

Creates a Hankel array from `img` with channel and kernel dimensions
`channel_size` and `kernel_size`, respectively. The returned array `A` satisfies:

```math
A_{c,j,k,n} = v_{c,j+k-1,n}
```

where `c,j,k,n` are multi-indices, with `c` traversing channels,
`j` traversing the kernel, `k` the convolution output, and `n` batches.
"""
function hankel(
    img::AbstractArray, channel_sz::NTuple{C,Int}, kernel_sz::NTuple{N,Int},
    stride::NTuple{N,Int}, pad::NTuple{N2,Int}, dilation::NTuple{N,Int}, padval=zero(eltype(img))
) where {C,N,N2}
    in_sz = size(img)[(C + 1):(C + N)]
    batch_sz = size(img)[(C + N + 1):end]
    @assert size(img) == (channel_sz..., in_sz..., batch_sz...)
    out_sz = output_size(kernel_sz, in_sz; stride, pad, dilation)
    H = similar(img, eltype(img), channel_sz..., kernel_sz..., out_sz..., batch_sz...)
    pad_lo = splitpad(pad).lo
    for k in CartesianIndices(out_sz), j in CartesianIndices(kernel_sz)
        i = CartesianIndex(
            (Tuple(j) .- 1) .* dilation .+ (Tuple(k) .- 1) .* stride .+ 1 .- pad_lo
        )
        inside = all(1 .≤ Tuple(i) .≤ in_sz)
        for b in CartesianIndices(batch_sz), c in CartesianIndices(channel_sz)
            H[c,j,k,b] = inside ? img[c,i,b] : padval
        end
    end
    return H
end

function hankel(
    img::AbstractArray, channel_sz::NTuple{C,Int}, kernel_sz::NTuple{N,Int};
    stride = 1, pad = 0, dilation = 1, padval = zero(eltype(img))
) where {C,N}
    return hankel(
        img, channel_sz, kernel_sz,
        expand_tuple(Val(N), stride),
        parsepad(kernel_sz, dilation, pad),
        expand_tuple(Val(N), dilation),
        padval
    )
end

function hankel(img::AbstractArray, C::Int, kernel_size::TupleN{Int}; kwargs...)
    channel_size = size(img)[1:C]
    return hankel(img, channel_size, kernel_size; kwargs...)
end
