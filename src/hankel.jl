@doc raw"""
    hankel_image(v, channel_size, kernel_size)

Creates a Hankel array from `v` with channel and kernel dimensions
`channel_size` and `kernel_size`, respectively. The returned array `V` satisfies:

```math
V_{c,j,k,n} = V_{c,j+k-1,n}
```

where `c,j,k,n` are multi-indices, with `c` traversing channels,
`j` traversing the kernel, `k` the convolution output, and `n` batches.
"""
function hankel_image(
    img::AbstractArray, channel_sz::NTuple{C,Int}, kernel_sz::NTuple{N,Int},
    stride::NTuple{N,Int}, pad::NTuple{N2,Int}, dilation::NTuple{N,Int}
) where {C,N,N2}
    input_sz = size(img)[(C + 1):(C + N)]
    batch_sz = size(img)[(C + N + 1):end]
    @assert size(img) == (channel_sz..., input_sz..., batch_sz...)
    out_sz = output_size(kernel_sz, input_sz; stride, pad, dilation)
    pad_lo = CartesianIndex(splitpad(pad).lo)
    X = zeros(eltype(img), channel_sz..., kernel_sz..., out_sz..., batch_sz...)
    for k in CartesianIndices(out_sz), j in CartesianIndices(kernel_sz)
        i = out2in(j, k; dilation, stride) - pad_lo
        all(1 .≤ Tuple(i) .≤ input_sz) || continue
        for b in CartesianIndices(batch_sz), c in CartesianIndices(channel_sz)
            X[c,j,k,b] = img[c,i,b]
        end
    end
    return X
end

function hankel_image(
    img::AbstractArray, channel_sz::NTuple{C,Int}, kernel_sz::NTuple{N,Int};
    stride = 1, pad = 0, dilation = 1
) where {C,N}
    return hankel_image(
        img, channel_sz, kernel_sz,
        expand_tuple(Val(N), stride),
        parsepad(kernel_sz, dilation, pad),
        expand_tuple(Val(N), dilation)
    )
end

function hankel_image(img::AbstractArray, C::Int, kernel_size::TupleN{Int}; kwargs...)
    channel_size = size(img)[1:C]
    return hankel_image(img, channel_size, kernel_size; kwargs...)
end

@doc raw"""
    hankel_weight(w, channel_size, input_size)

Creates a Hankel array from `v` with channel and kernel dimensions
`channel_size` and `kernel_size`, respectively. The returned array `V` satisfies:

```math
V_{c,j,k,n} = V_{c,j+k-1,n}
```

where `c,j,k,n` are multi-indices, with `c` traversing channels,
`j` traversing the kernel, `k` the convolution output, and `n` batches.
"""
function hankel_weight(
    w::AbstractArray, channel_sz::NTuple{C,Int}, in_sz::NTuple{N,Int};
    stride = 1, pad = 0, dilation = 1
) where {C,N}
    kernel_sz = size(w)[(C + 1):(C + N)]
    return hankel_weight(
        w, channel_sz, in_sz,
        expand_tuple(Val(N), stride),
        parsepad(kernel_sz, dilation, pad),
        expand_tuple(Val(N), dilation)
    )
end

function hankel_weight(
    w::AbstractArray, channel_sz::NTuple{C,Int}, in_sz::NTuple{N,Int},
    stride::NTuple{N,Int}, pad::NTuple{N2,Int}, dilation::NTuple{N,Int}
) where {C,N,N2}
    kernel_sz = size(w)[(C + 1):(C + N)]
    hidden_sz = size(w)[(C + N + 1):end]
    @assert size(w) == (channel_sz..., kernel_sz..., hidden_sz...)
    out_sz = output_size(kernel_sz, in_sz; stride, pad, dilation)
    pad_lo = CartesianIndex(splitpad(pad).lo)
    A = zeros(eltype(w), channel_sz..., in_sz..., hidden_sz..., out_sz...)
    for k in CartesianIndices(out_sz), j in CartesianIndices(kernel_sz)
        i = out2in(j, k; dilation, stride) - pad_lo
        all(1 .≤ Tuple(i) .≤ in_sz) || continue
        for μ in CartesianIndices(hidden_sz), c in CartesianIndices(channel_sz)
            A[c,i,μ,k] = w[c,j,μ]
        end
    end
    return A
end

function hankel_weight(w::AbstractArray, C::Int, in_sz::TupleN{Int}; kwargs...)
    channel_size = size(w)[1:C]
    return hankel_weight(w, channel_size, in_sz; kwargs...)
end

"""
    hankel(convrbm, input_size)

Returns a dense `RBM` equivalent to `convrbm` with replicated weights.
"""
function hankel(rbm::ConvRBM, in_size::NTuple{N,Int}) where {N}
    @assert kernel_ndims(rbm) == N
    @assert rbm.groups == 1 # groups > 1 not supported
    w = hankel_weight(rbm.w, channel_size(rbm), in_size; rbm.stride, rbm.pad, rbm.dilation)
    out_size = output_size(rbm, in_size)
    vis = replicate(visible(rbm), in_size...)
    hid = replicate(hidden(rbm), out_size...)
    return RBM(vis, hid, w)
end

"""
    replicate(layer, n...)

Returns a new layer of size `(size(layer)..., n...)` by repeating the original `layer`
along the new dimensions.
"""
replicate(l::AbstractLayer, n::Int...) = repeat(l, map(Returns(1), size(l))..., n...)
