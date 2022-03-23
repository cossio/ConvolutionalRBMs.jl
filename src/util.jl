const TupleN{T,N} = NTuple{N,T}
const AbstractTensor{N,T} = AbstractArray{T,N}

"""
    reshape_maybe(x, shape)

Like `reshape(x, shape)`, except that zero-dimensional outputs are returned as scalars.
"""
reshape_maybe(x::Number, ::Tuple{}) = x
reshape_maybe(x::AbstractArray, ::Tuple{}) = only(x)
reshape_maybe(x::AbstractArray, sz::Tuple{Int,Vararg{Int}}) = reshape(x, sz)
reshape_maybe(x::Number) = x
reshape_maybe(x::AbstractArray, sz::Int...) = reshape_maybe(x, sz)

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
    dilation_tuple = ntuple(Returns(dilation), Val(K))
    return samepad(kernel_size, dilation_tuple)
end

function parsepad(kernel_size::NTuple{N,Int}, dilation, pad) where {N}
    if pad === :same
        return samepad(kernel_size, dilation)
    elseif pad isa Int
        return ntuple(Returns(pad), Val(2N))
    elseif pad isa NTuple{N,Int}
        return ntuple(d -> pad[d ÷ 2 + 1], Val(2N))
    else
        return pad::NTuple{2N,Int}
    end
end

"""
    splitpad(pad)

Given `pad = (x_lo, x_hi, y_lo, y_hi, ...)`, returns
`lo = (x_lo, y_lo, ...), hi = (x_hi, y_hi, ...)`.
"""
function splitpad(pad::NTuple{N,Int}) where {N}
    @assert iseven(N)
    pad_lo = ntuple(d -> pad[2d - 1], Val(N ÷ 2))
    pad_hi = ntuple(d -> pad[2d], Val(N ÷ 2))
    return (lo = pad_lo, hi = pad_hi)
end

"""
    expand_tuple(Val(N), t)

Expands `t` into a tuple of length `N`.
"""
expand_tuple(::Val{N}, t::Int) where {N} = ntuple(Returns(t), Val(N))
expand_tuple(::Val{N}, t::NTuple{N,Int}) where {N} = t

"""
    output_size(kernel_size, input_size; stride = 1, pad = 0, dilation = 1)

Output size of the convolution.
"""
output_size(k::NTuple{N,Int}, i::NTuple{N,Int}; stride = 1, pad = 0, dilation = 1) where {N} = output_size(
    k, i,
    expand_tuple(Val(N), stride),
    parsepad(k, dilation, pad),
    expand_tuple(Val(N), dilation)
)

function output_size(
    kernel::NTuple{N,Int}, input::NTuple{N,Int},
    stride::NTuple{N,Int}, pad::NTuple{N2,Int}, dilation::NTuple{N,Int}
) where {N,N2}
    @assert N2 == 2N
    return ntuple(Val(N)) do i
        (input[i] + pad[2i - 1] + pad[2i] - (kernel[i] - 1) * dilation[i] - 1) ÷ stride[i] + 1
    end
end

"""
    out2in(j, k; stride = 1, dilation = 1)

Given kernel index `j` and output index `k`, gets the corresponding input index `i`.
"""
function out2in(j::CartesianIndex{N}, k::CartesianIndex{N}; stride = 1, dilation = 1) where {N}
    i = (Tuple(j) .- 1) .* dilation .+ (Tuple(k) .- 1) .* stride .+ 1
    return CartesianIndex(i)
end

"""
    translate(v, channel_size, input_size, Δ; mode = :pad, fillvalue = 0)

Translates an image along input dimensions by `Δ`.
"""
function translate(
    v::AbstractArray,
    channel_size::NTuple{C,Int},
    input_size::NTuple{N,Int},
    Δ::CartesianIndex{N};
    mode::Symbol = :pad,
    fillvalue = zero(eltype(v))
) where {C,N}
    batch_size = size(v)[(C + N + 1):end]
    @assert size(v) == (channel_size..., input_size..., batch_size...)
    if mode === :pad
        Δv = similar(v)
        for i in CartesianIndices(input_size)
            if i + Δ ∈ CartesianIndices(input_size)
                for n in CartesianIndices(batch_size), c in CartesianIndices(channel_size)
                    Δv[c,i,n] = v[c, i + Δ, n]
                end
            else
                for n in CartesianIndices(batch_size), c in CartesianIndices(channel_size)
                    Δv[c,i,n] = fillvalue
                end
            end
        end
        return Δv
    elseif mode === :periodic
        Δv = similar(v)
        for i in CartesianIndices(input_size)
            p = mod1.(Tuple(i + Δ), input_size)
            for n in CartesianIndices(batch_size), c in CartesianIndices(channel_size)
                Δv[c,i,n] = v[c, p, n]
            end
        end
        return Δv
    end
end

function translate(
    v::AbstractArray,
    channel_size::NTuple{C,Int},
    input_size::NTuple{N,Int},
    Δ::NTuple{N,Int};
    mode::Symbol = :pad,
    fillvalue = zero(eltype(v))
) where {C,N}
    return translate(v, channel_size, input_size, CartesianIndex(Δ); mode, fillvalue)
end
