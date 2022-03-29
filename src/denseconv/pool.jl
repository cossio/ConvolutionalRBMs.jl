function free_energy_nopool(rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray; β::Real = 1)
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

function free_energy_pooled(rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray; β::Real = 1)
    E_v_dense = energy(rbm.visible_dense, v_dense)
    E_v_conv = energy(rbm.visible_conv, v_conv)
    I = inputs_v_to_h(rbm, v_dense, v_conv)
    F = RBMs.free_energies(hidden(rbm), I; β)

    vsz_conv = vsizes(parts(rbm).conv, v_conv)
    @assert size(E_v_dense) == batch_size(rbm.visible_dense, v_dense)
    @assert size(E_v_conv) == (vsz_conv.input_size..., vsz_conv.batch_size...)
    @assert size(F) == size(I) == (size(hidden(rbm))..., hsizes(rbm, I).output_size..., hsizes(rbm, I).batch_size...)
    G = -logsumexp(-β * F; dims=output_dims(rbm)) / β # pooling step <=> only one h active per group
    @assert size(G)[1:ndims(hidden(rbm))] == size(hidden(rbm))
    @assert all(size(G)[output_dims(rbm)] .== 1)

    # reduce over remaining dims
    E_vconv_reduced = reshape_maybe(sum(E_v_conv; dims=1:kernel_ndims(rbm)), vsz_conv.batch_size)
    F_reduced = reshape_maybe(sum(G; dims=1:ndims(hidden(rbm))), hsizes(rbm, I).batch_size)

    return E_v_dense .+ E_vconv_reduced .+ F_reduced
end

function sample_h_from_v_nopool(
    rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray; β::Real = true
)
    inputs = inputs_v_to_h(rbm, v_dense, v_conv)
    return transfer_sample(hidden(rbm), inputs; β)
end

function sample_h_from_v_pooled(rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray; β::Real = 1)
    I = inputs_v_to_h(rbm, v_dense, v_conv)
    F = RBMs.free_energies(hidden(rbm), I; β)
    @assert size(F) == size(I) == (size(hidden(rbm))..., hsizes(rbm, I).output_size..., hsizes(rbm, I).batch_size...)
    selected_size = (size(hidden(rbm))..., hsizes(rbm, I).batch_size...)
    k = reshape(cartesian_sample_from_logits_gumbel(-β * F; dims=output_dims(rbm)), selected_size)

    selected_inputs = similar(I, selected_size)
    for n in CartesianIndices(hsizes(rbm, I).batch_size), μ in CartesianIndices(size(hidden(rbm)))
        @inbounds selected_inputs[μ,n] = I[k[μ,n]]
    end

    selected_h = RBMs.transfer_sample(hidden(rbm), selected_inputs; β)
    @assert size(selected_h) == size(selected_inputs) == selected_size

    h = fill!(similar(selected_h, size(F)), 0)
    for n in CartesianIndices(hsizes(rbm, I).batch_size), μ in CartesianIndices(size(hidden(rbm)))
        h[k[μ,n]] = selected_h[μ,n]
    end
    return h
end
