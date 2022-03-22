# NNlib seems about 30x faster than using Tullio.

@doc raw"""
    conv_v2h(w, v)

Internal function used to compute inputs from a visible configurations `v` to the hidden
layer, where `w` are the convolutional RBM weights.

```math
I_\mu^{k_1,\dots,k_n,b} = \sum_{c,j_1,\dots,j_n} w_{c,j_1,\dots,j_n,\mu} v_{c,j_1+k_1-1,\dots,j_n+k_n-1,b}
```

Assumes that:

* `w` is of size `(C,J₁,...,Jₙ,M)`
* `v` is of size `(C,N₁,...,Nₙ,B)`

Here `C` is the flattened channel dimension,
`M` is the number of hidden units and `B` is the batch size.
Therefore the hidden and batch dimensions must be flattened before calling `conv_v2h`.
In the default case with `stride = 1`, `pad = 0`, `dilation = 1`, ...,
the output `I` is of size `(M, N₁ - J₁ + 1, ..., Nₙ - Jₙ + 1, B)`,

!!! warning
    This is an internal function and is not part of the public API.

!!! warning
    Only works for `n = 1, 2, 3` due to a technical limitations.
"""
function conv_v2h(
    w::AbstractTensor{N}, v::AbstractTensor{N};
    stride = 1, pad = 0, dilation = 1, groups = 1
) where {N}
    @assert size(w, 1) == size(v, 1) # channel size
    @assert all((size(w) .≤ size(v))[(begin + 1):(end - 1)]) # kernel dims
    @assert 3 ≤ N ≤ 5 # NNlib limitation
    w_ = nnlib_conv_permdims(w)
    v_ = nnlib_conv_permdims(v)
    I_ = NNlib.conv(v_, w_; stride, pad, dilation, groups, flipped=true)
    @assert ndims(I_) == N
    return nnlib_conv_invpermdims(I_)
end

@doc raw"""
    conv_h2v(w, v)

Internal function used to compute inputs from a hidden configurations `h` to the visible
layer, where `w` are the convolutional RBM weights.

```math
I_{i_1,\dots,i_n}^{k_1,\dots,k_n,b} = \sum_{c,j_1,\dots,j_n} w_{c,j_1,\dots,j_n,\mu} v_{c,j_1+k_1-1,\dots,j_n+k_n-1,b}
```

Assumes that:

* `w` is of size `(C,J₁,...,Jₙ,M)`
* `h` is of size `(M,K₁,...,Kₙ,B)`

Here `C` is the channel dimension, `M` is the number of hidden units and `B` is the batch size.
These three dimensions are flat, so that the hidden and batch dimensions must be
flattened before calling `conv_h2v`.
In the default case with `stride = 1`, `pad = 0`, `dilation = 1`, ...,
the output `I` is of size `(C, K₁ + J₁ - 1, ..., Kₙ + Jₙ - 1, B)`.

!!! warning
    This is an internal function, not part of the public API. It is subject to change.
"""
function conv_h2v(
    w::AbstractTensor{N}, h::AbstractTensor{N};
    stride = 1, pad = 0, dilation = 1, groups = 1
) where {N}
    @assert size(w)[end] == size(h, 1) # hidden units
    w_ = nnlib_conv_permdims(w)
    h_ = nnlib_conv_permdims(h)
    layer = Flux.ConvTranspose(w_, false; stride, pad, dilation, groups)
    cdims = NNlib.DenseConvDims(Flux.conv_transpose_dims(layer, h_); F=true)
    I_ = NNlib.∇conv_data(h_, w_, cdims)
    return nnlib_conv_invpermdims(I_)
end

function nnlib_conv_permdims(x::AbstractArray{T,N}) where {T,N}
    # NNlib expects dims in this order: spatial (convolved), channels, batch
    return permutedims(x, (ntuple(d -> d + 1, N - 2)..., 1, N))
end

function nnlib_conv_invpermdims(x::AbstractArray{T,N}) where {T,N}
    return permutedims(x, (N - 1, ntuple(identity, N - 2)..., N))
end
