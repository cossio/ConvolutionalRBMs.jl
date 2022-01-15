const TupleN{T,N} = NTuple{N,T}
const AbstractTensor{N,T} = AbstractArray{T,N}

"""
    reshape_maybe(x, shape)

Like `reshape(x, shape)`, except that zero-dimensional outputs are returned as scalars.
"""
reshape_maybe(x::Number, ::Tuple{}) = x
reshape_maybe(x::AbstractArray, ::Tuple{}) = only(x)
reshape_maybe(x::AbstractArray, sz::Tuple{Int,Vararg{Int}}) = reshape(x, sz)
reshape_maybe(x::Union{Number,AbstractArray}, sz::Int...) = reshape_maybe(x, sz)

"""
    samepad(kernel_size, dilation = 1)

Computes padding such that the input and output spatial sizes of the convolution
are the same.

If stride > 1, then `size(output, d) * stride[d] == size(input, d)` for
each spatial dimension `d`.
"""
function samepad end

function samepad(kernel_size::NTuple{K,Int}, dilation::NTuple{K,Int}) where {K}
    sz = (kernel_size .- 1) .* dilation
    return ntuple(Val(2K)) do k
        isodd(k) ? cld(sz[(k + 1) ÷ 2], 2) : fld(sz[k ÷ 2], 2)
    end
end

function samepad(kernel_size::NTuple{K,Int}, dilation::Int = 1) where {K}
    dilation_tuple = expand_size(Val(K), dilation)
    return samepad(kernel_size, dilation_tuple)
end

"""
    expand_size(Val(N), sz)

Expands `sz` into a tuple of length `N`.
"""
expand_size(::Val{N}, sz::Int) where {N} = ntuple(Returns(sz), Val(N))
expand_size(::Val{N}, sz::NTuple{N,Int}) where {N} = sz
