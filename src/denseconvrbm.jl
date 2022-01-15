struct DenseConvRBM{Vd,Vc,H,Wd,Wc,K,K2}
    visible_dense::Vd # dense visible part
    visible_conv::Vc # convolutional visible part
    hidden::H # hidden layer
    w_dense::Wd # dense weights
    w_conv::Wc # convolutional weights
    stride::NTuple{K,Int}
    pad::NTuple{K2,Int}
    dilation::NTuple{K,Int}
    groups::Int
    """
        DenseConvRBM(visible_dense, visible_conv, hidden, weights_dense, weights_conv)

    Constructs an RBM where a portion of the visible layer is dense and another portion is
    convolutional.
    """
    function DenseConvRBM(
        visible_dense::AbstractLayer,
        visible_conv::AbstractLayer,
        hidden::AbstractLayer,
        w_dense::AbstractArray,
        w_conv::AbstractArray,
        stride::NTuple{K,Int}, pad::NTuple{K2,Int}, dilation::NTuple{K,Int}, groups::Int
    ) where {K,K2}
        # dense size checks
        @assert size(w_dense) == (size(visible_dense)..., size(hidden)...)
        # conv size checks
        @assert K == ndims(w_conv) - ndims(visible_conv) - ndims(hidden)
        @assert K2 == 2K
        @assert size(w_conv)[begin:ndims(visible_conv)] == size(visible_conv)
        @assert size(w_conv)[(ndims(visible_conv) + K + 1):end] == size(hidden)
        # types
        V_dens = typeof(visible_dense)
        V_conv = typeof(visible_conv)
        H = typeof(hidden)
        W_dens = typeof(w_dense)
        W_conv = typeof(w_conv)
        # construct
        return new{V_dens, V_conv, H, W_dens, W_conv, K, K2}(
            visible_dense, visible_conv, hidden, w_dense, w_conv,
            stride, pad, dilation, groups
        )
    end
end

function DenseConvRBM(
    visible_dense::AbstractLayer,
    visible_conv::AbstractLayer,
    hidden::AbstractLayer,
    w_dense::AbstractArray,
    w_conv::AbstractArray;
    stride = 1, pad = 0, dilation = 1, groups::Int = 1
)
    K = ndims(w_conv) - ndims(visible_conv) - ndims(hidden)
    kernel_size = size(w_conv)[(ndims(visible_conv) + 1):(ndims(visible_conv) + K)]
    return DenseConvRBM(
        visible_dense, visible_conv, hidden, w_dense, w_conv,
        expand_size(Val(K), stride),
        pad === :same ? samepad(kernel_size, dilation) : expand_size(Val(2K), pad),
        expand_size(Val(K), dilation),
        groups
    )
end

"""
    parts(denseconvrbm)

Extracts the dense and convolutional parts of a `DenseConvRBM`.
"""
parts(rbm::DenseConvRBM) = (
    dense = RBM(rbm.visible_dense, rbm.hidden, rbm.w_dense),
    conv = ConvRBM(
        rbm.visible_conv, rbm.hidden, rbm.w_conv;
        stride = rbm.stride, pad = rbm.pad, dilation = rbm.dilation, groups = rbm.groups
    )
)

RBMs.hidden(rbm::DenseConvRBM) = rbm.hidden
kernel_ndims(rbm::DenseConvRBM) = kernel_ndims(parts(rbm).conv)
kernel_size(rbm::DenseConvRBM) = kernel_size(parts(rbm).conv)
hsizes(rbm::DenseConvRBM, h::AbstractArray) = hsizes(parts(rbm).conv, h)

function RBMs.batch_size(
    rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray, h::AbstractArray
)
    bsz_dense = batch_size(rbm.visible_dense, v_dense)
    bsz_conv = batch_size(parts(rbm).conv, v_conv, h)
    if isempty(bsz_dense)
        return bsz_conv
    elseif isempty(bsz_conv)
        return bsz_dense
    else
        @assert bsz_dense == bsz_conv
        return bsz_dense
    end
end

function RBMs.batch_size(rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray)
    bsz_dense = batch_size(rbm.visible_dense, v_dense)
    bsz_conv = batch_size(parts(rbm).conv, v_conv)
    if isempty(bsz_dense)
        return bsz_conv
    elseif isempty(bsz_conv)
        return bsz_dense
    else
        @assert bsz_dense == bsz_conv
        return bsz_dense
    end
end

function RBMs.inputs_v_to_h(rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray)
    I_dense = inputs_v_dense_to_h(rbm, v_dense)
    I_conv = inputs_v_conv_to_h(rbm, v_conv)
    return I_dense .+ I_conv
end

function inputs_v_conv_to_h(rbm::DenseConvRBM, v_conv::AbstractArray)
    return inputs_v_to_h(parts(rbm).conv, v_conv)
end

function inputs_v_dense_to_h(rbm::DenseConvRBM, v_dense::AbstractArray)
    I_dense = inputs_v_to_h(parts(rbm).dense, v_dense)
    bsz = batch_size(rbm.visible_dense, v_dense)
    @assert size(I_dense) == (size(rbm.hidden)..., bsz...)
    return reshape(I_dense, size(rbm.hidden)..., one.(kernel_size(rbm))..., bsz...)
end

function inputs_h_to_v_dense(rbm::DenseConvRBM, h::AbstractArray)
    I_dense = inputs_h_to_v(parts(rbm).dense, h)
    hsz = hsizes(rbm, h)
    @assert size(I_dense) == (size(rbm.visible_dense)..., hsz.output_size..., hsz.batch_size...)
    kdims = ndims(rbm.visible_dense) .+ (1:kernel_ndims(rbm))
    return reshape(sum(I_dense; dims=kdims), size(rbm.visible_dense)..., hsz.batch_size...)
end

inputs_h_to_v_conv(rbm::DenseConvRBM, h::AbstractArray) = inputs_h_to_v(parts(rbm).conv, h)

function RBMs.energy(
    rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray, h::AbstractArray
)
    E_v_dense = energy(rbm.visible_dense, v_dense)
    E_v_conv = energy(rbm.visible_conv, v_conv)
    E_h = energy(rbm.hidden, h)
    E_w = interaction_energy(rbm, v_dense, v_conv, h)

    E_v_conv_reduced = reshape_maybe(
        sum(E_v_conv; dims=1:kernel_ndims(rbm)),
        vsizes(parts(rbm).conv, v_conv).batch_size
    )
    E_h_reduced = reshape_maybe(
        sum(E_h; dims=1:kernel_ndims(rbm)),
        hsizes(rbm, h).batch_size
    )

    return E_v_dense .+ E_v_conv_reduced .+ E_h_reduced .+ E_w
end

function RBMs.interaction_energy(
    rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray, h::AbstractArray
)
    I = RBMs.inputs_v_to_h(rbm, v_dense, v_conv)
    @assert size(I) == (
        size(hidden(rbm))..., hsizes(rbm, h).output_size..., batch_size(rbm, v_dense, v_conv)...
    )
    E = -sum(I .* h; dims=1:(ndims(hidden(rbm)) + kernel_ndims(rbm)))
    return reshape_maybe(E, batch_size(rbm, v_dense, v_conv, h))
end

function RBMs.free_energy(rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray; β::Real = 1)
    E_v_dense = energy(rbm.visible_dense, v_dense)
    E_v_conv = energy(rbm.visible_conv, v_conv)
    I = inputs_v_to_h(rbm, v_dense, v_conv)
    F = RBMs.free_energy(hidden(rbm), I; β)

    vsz_conv = vsizes(parts(rbm).conv, v_conv)
    @assert size(E_v_dense) == batch_size(rbm.visible_dense, v_dense)
    @assert size(E_v_conv) == (vsz_conv.input_size..., vsz_conv.batch_size...)
    @assert size(F) == (hsizes(rbm, I).output_size..., hsizes(rbm, I).batch_size...)

    # reduce over kernel dims
    E_vconv_reduced = reshape_maybe(sum(E_v_conv; dims=1:kernel_ndims(rbm)), vsz_conv.batch_size)
    F_reduced = reshape_maybe(sum(F; dims=1:kernel_ndims(rbm)), hsizes(rbm, I).batch_size)

    return E_v_dense .+ E_vconv_reduced .+ F_reduced
end

function RBMs.sample_h_from_v(
    rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray; β::Real = true
)
    inputs = inputs_v_to_h(rbm, v_dense, v_conv)
    return transfer_sample(hidden(rbm), inputs; β)
end

function sample_v_dense_from_h(rbm::DenseConvRBM, h::AbstractArray; β::Real = true)
    inputs = inputs_h_to_v_dense(rbm, h)
    return transfer_sample(rbm.v_dense, inputs; β)
end

function sample_v_conv_from_h(rbm::DenseConvRBM, h::AbstractArray; β::Real = true)
    inputs = inputs_h_to_v_conv(rbm, h)
    return transfer_sample(rbm.visible_conv, inputs; β)
end

function RBMs.sample_v_from_v(
    rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray;
    β::Real = true, steps::Int = 1
)
    for _ in 1:steps
        h = sample_h_from_v(rbm, v_dense, v_conv; β)
        v_dense = oftype(v_dense, sample_v_dense_from_h(rbm, h; β))
        v_conv = oftype(v_conv, sample_v_conv_from_h(rbm, h; β))
    end
    return (v_dense = v_dense, v_conv = v_conv)
end

function RBMs.sample_h_from_h(
    rbm::DenseConvRBM, h::AbstractArray; β::Real = true, steps::Int = 1
)
    for _ in 1:steps
        h = oftype(h, RBMs.sample_h_from_h_once(rbm, h; β))
    end
    return h
end

function RBMs.sample_h_from_h_once(rbm::DenseConvRBM, h::AbstractArray; β::Real = true)
    v_dense = sample_v_dense_from_h(rbm, h; β)
    v_conv = sample_v_conv_from_h(rbm, h; β)
    return RBMs.sample_h_from_v(rbm, v_dense, v_conv; β)
end
