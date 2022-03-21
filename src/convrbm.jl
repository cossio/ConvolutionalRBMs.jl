struct ConvRBM{V,H,W,K,K2}
    visible::V
    hidden::H
    w::W
    stride::NTuple{K,Int}
    pad::NTuple{K2,Int}
    dilation::NTuple{K,Int}
    groups::Int
    function ConvRBM(
        visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
        stride::NTuple{K,Int}, pad::NTuple{K2,Int}, dilation::NTuple{K,Int}, groups::Int
    ) where {K,K2}
        @assert ndims(w) == ndims(visible) + K + ndims(hidden)
        @assert size(w)[begin:ndims(visible)] == size(visible)
        @assert size(w)[(ndims(visible) + K + 1):end] == size(hidden)
        @assert K2 == 2K
        return new{typeof(visible), typeof(hidden), typeof(w), K, K2}(
            visible, hidden, w, stride, pad, dilation, groups
        )
    end
end

"""
    ConvRBM(visible, hidden, weights; stride = 1, pad = 0, dilation = 1, groups = 1)

Convolutional RBM.

`pad == :same` uses [samepad](@ref) to infer a padding that preserves spatial dimensions.
"""
function ConvRBM(
    visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray;
    stride = 1, pad = 0, dilation = 1, groups::Int = 1
)
    K = ndims(w) - ndims(visible) - ndims(hidden)
    kernel_size = size(w)[(ndims(visible) + 1):(ndims(visible) + K)]
    return ConvRBM(
        visible, hidden, w,
        expand_tuple(Val(K), stride),
        parsepad(kernel_size, dilation, pad),
        expand_tuple(Val(K), dilation),
        groups
    )
end

"""
    ConvRBM(visible, hidden, kernel_size; kwargs...)

Convolutional RBM with given `kernel_size`.
"""
function ConvRBM(
    visible::AbstractLayer, hidden::AbstractLayer, kernel_size::TupleN{Int}, ::Type{T} = Float64;
    kwargs...
) where {T}
    w = zeros(T, size(visible)..., kernel_size..., size(hidden)...)
    return ConvRBM(visible, hidden, w; kwargs...)
end

RBMs.visible(rbm::ConvRBM) = rbm.visible
RBMs.hidden(rbm::ConvRBM) = rbm.hidden
RBMs.weights(rbm::ConvRBM) = rbm.w

channel_size(rbm::ConvRBM) = size(visible(rbm))
channel_length(rbm::ConvRBM) = length(visible(rbm))
channel_ndims(rbm::ConvRBM) = ndims(visible(rbm))

kernel_size(rbm::ConvRBM) = size(weights(rbm))[kernel_dims(rbm)]
kernel_ndims(rbm::ConvRBM) = length(kernel_size(rbm))
kernel_dims(rbm::ConvRBM) = (channel_ndims(rbm) + 1):(ndims(weights(rbm)) - ndims(hidden(rbm)))

"""
    vsizes(convrbm, v) -> (channel_size, input_size, batch_size)

Returns a (named) tuple decomposition of the size of `v`, such that:

    size(v) == (channel_size..., input_size..., batch_size...)

Throws an error if `v` is not consistent with this size.
"""
function vsizes(rbm::ConvRBM, v::AbstractArray)
    input_size = size(v)[(channel_ndims(rbm) + 1):(channel_ndims(rbm) + kernel_ndims(rbm))]
    batch_size = size(v)[(channel_ndims(rbm) + kernel_ndims(rbm) + 1):end]
    @assert size(v) == (channel_size(rbm)..., input_size..., batch_size...)
    return (channel_size = channel_size(rbm), input_size = input_size, batch_size = batch_size)
end

"""
    hsizes(convrbm, h) -> (hidden_size, output_size, batch_size)

Returns a (named) tuple decomposition of the size of `h`, such that:

    size(h) == (hidden_size..., output_size..., batch_size...)

Throws an error if `h` is not consistent with this size.
"""
function hsizes(rbm::ConvRBM, h::AbstractArray)
    output_size = size(h)[(ndims(hidden(rbm)) + 1):(ndims(hidden(rbm)) + kernel_ndims(rbm))]
    batch_size = size(h)[(ndims(hidden(rbm)) + kernel_ndims(rbm) + 1):end]
    @assert size(h) == (size(hidden(rbm))..., output_size..., batch_size...)
    return (hidden_size = size(hidden(rbm)), output_size = output_size, batch_size = batch_size)
end

function RBMs.batch_size(rbm::ConvRBM, v::AbstractArray, h::AbstractArray)
    vsz = vsizes(rbm, v)
    hsz = hsizes(rbm, h)
    if !isempty(vsz.batch_size) && !isempty(hsz.batch_size)
        @assert vsz.batch_size == hsz.batch_size
        return vsz.batch_size
    elseif isempty(hsz.batch_size)
        return vsz.batch_size
    elseif isempty(vsz.batch_size)
        return hsz.batch_size
    end
end

RBMs.batch_size(rbm::ConvRBM, v::AbstractArray) = vsizes(rbm, v).batch_size

"""
    output_size(rbm, v)

Output size of the convolution.

If `v` has `batch_size` batches, then `output_size(rbm, v)` returns `output_size` such that:

    size(h) == (hidden_size..., output_size..., batch_size...)
"""
output_size(rbm::ConvRBM, v::AbstractArray) = output_size(
    kernel_size(rbm), vsizes(rbm, v).input_size, rbm.stride, rbm.pad, rbm.dilation
)

function RBMs.inputs_v_to_h(rbm::ConvRBM, v::AbstractArray)
    vsz = vsizes(rbm, v)
    wflat = reshape(rbm.w, channel_length(rbm), kernel_size(rbm)..., length(hidden(rbm)))
    vflat = reshape(v, channel_length(rbm), vsz.input_size..., prod(vsz.batch_size))
    Iflat = conv_v2h(
        wflat, activations_convert_maybe(wflat, vflat);
        stride = rbm.stride, pad = rbm.pad, dilation = rbm.dilation, groups = rbm.groups
    )
    @assert size(Iflat) == (length(hidden(rbm)), size(Iflat)[2:end-1]..., prod(vsz.batch_size))
    return reshape(Iflat, size(hidden(rbm))..., size(Iflat)[2:end-1]..., vsz.batch_size...)
end

function RBMs.inputs_h_to_v(rbm::ConvRBM, h::AbstractArray)
    hsz = hsizes(rbm, h)
    wflat = reshape(rbm.w, length(visible(rbm)), kernel_size(rbm)..., length(hidden(rbm)))
    hflat = reshape(h, length(hidden(rbm)), hsz.output_size..., prod(hsz.batch_size))
    Iflat = conv_h2v(
        wflat, activations_convert_maybe(wflat, hflat);
        stride = rbm.stride, pad = rbm.pad, dilation = rbm.dilation, groups = rbm.groups
    )
    @assert size(Iflat) == (channel_length(rbm), size(Iflat)[2:end-1]..., prod(hsz.batch_size))
    return reshape(Iflat, channel_size(rbm)..., size(Iflat)[2:end-1]..., hsz.batch_size...)
end

function RBMs.energy(rbm::ConvRBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(visible(rbm), v)
    @assert size(Ev) == (vsizes(rbm, v).input_size..., vsizes(rbm, v).batch_size...)
    Eh = energy(hidden(rbm), h)
    @assert size(Eh) == (hsizes(rbm, h).output_size..., hsizes(rbm, h).batch_size...)
    Ew = interaction_energy(rbm, v, h)
    @assert size(Ew) == batch_size(rbm, v, h)
    Ev_reduced = reshape_maybe(sum(Ev; dims=1:kernel_ndims(rbm)), vsizes(rbm, v).batch_size)
    Eh_reduced = reshape_maybe(sum(Eh; dims=1:kernel_ndims(rbm)), hsizes(rbm, h).batch_size)
    E = Ev_reduced .+ Eh_reduced .+ Ew
    @assert size(E) == batch_size(rbm, v, h)
    return E
end

function RBMs.interaction_energy(rbm::ConvRBM, v::AbstractArray, h::AbstractArray)
    bsz = batch_size(rbm, v, h) # do size checks early
    I = inputs_v_to_h(rbm, v) # takes care of the convolution
    @assert size(I) == (size(hidden(rbm))..., hsizes(rbm, h).output_size..., vsizes(rbm, v).batch_size...)
    E = -sum(I .* h; dims=1:(ndims(hidden(rbm)) + kernel_ndims(rbm)))
    return reshape_maybe(E, bsz)
end

function RBMs.free_energy(rbm::ConvRBM, v::AbstractArray; β::Real = true)
    E = energy(rbm.visible, v)
    I = inputs_v_to_h(rbm, v)
    F = free_energy(rbm.hidden, I; β)
    @assert size(E) == (vsizes(rbm, v).input_size..., vsizes(rbm, v).batch_size...)
    @assert size(F) == (hsizes(rbm, I).output_size..., vsizes(rbm, v).batch_size...)
    # reduce over kernel dims
    E_conv = reshape_maybe(sum(E; dims=1:kernel_ndims(rbm)), vsizes(rbm, v).batch_size)
    F_conv = reshape_maybe(sum(F; dims=1:kernel_ndims(rbm)), vsizes(rbm, v).batch_size)
    return E_conv + F_conv
end

function RBMs.sample_h_from_v(rbm::ConvRBM, v::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_sample(hidden(rbm), inputs; β)
end

function RBMs.sample_v_from_h(rbm::ConvRBM, h::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_h_to_v(rbm, h)
    return RBMs.transfer_sample(visible(rbm), inputs; β)
end

function RBMs.sample_v_from_v(rbm::ConvRBM, v::AbstractArray; β::Real = true, steps::Int = 1)
    for _ in 1:steps
        v = oftype(v, RBMs.sample_v_from_v_once(rbm, v; β))
    end
    return v
end

function RBMs.sample_h_from_h(rbm::ConvRBM, h::AbstractArray; β::Real = true, steps::Int = 1)
    for _ in 1:steps
        h = oftype(h, RBMs.sample_h_from_h_once(rbm, h; β))
    end
    return h
end

function RBMs.sample_v_from_v_once(rbm::ConvRBM, v::AbstractArray; β::Real = true)
    h = RBMs.sample_h_from_v(rbm, v; β)
    v = RBMs.sample_v_from_h(rbm, h; β)
    return v
end

function RBMs.sample_h_from_h_once(rbm::ConvRBM, h::AbstractArray; β::Real = true)
    v = RBMs.sample_v_from_h(rbm, h; β)
    h = RBMs.sample_h_from_v(rbm, v; β)
    return h
end
