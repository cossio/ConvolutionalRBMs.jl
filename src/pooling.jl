function free_energy_nopool(rbm::ConvRBM, v::AbstractArray; β::Real = true)
    E = energy(visible(rbm), v)
    I = inputs_v_to_h(rbm, v)
    F = free_energy(hidden(rbm), I; β)
    @assert size(E) == (vsizes(rbm, v).input_size..., vsizes(rbm, v).batch_size...)
    @assert size(F) == (hsizes(rbm, I).output_size..., vsizes(rbm, v).batch_size...)
    # reduce over kernel dims
    E_conv = reshape_maybe(sum(E; dims=1:kernel_ndims(rbm)), vsizes(rbm, v).batch_size)
    F_conv = reshape_maybe(sum(F; dims=1:kernel_ndims(rbm)), vsizes(rbm, v).batch_size)
    return E_conv + F_conv
end

function free_energy_pooled(rbm::ConvRBM, v::AbstractArray; β::Real = 1)
    E = energy(visible(rbm), v)
    I = inputs_v_to_h(rbm, v)
    F = RBMs.free_energies(hidden(rbm), I; β)
    @assert size(E) == (vsizes(rbm, v).input_size..., batch_size(rbm, v)...)
    @assert size(F) == size(I) == (size(hidden(rbm))..., output_size(rbm, v)..., batch_size(rbm, v)...)
    G = -logsumexp(-β * F; dims=output_dims(rbm)) / β # pooling step <=> only one h active per group
    @assert size(G)[1:ndims(hidden(rbm))] == size(hidden(rbm))
    @assert all(size(G)[output_dims(rbm)] .== 1)
    # reduce over remaining dims
    E_reduced = reshape_maybe(sum(E; dims=1:kernel_ndims(rbm)), batch_size(rbm, v))
    F_reduced = reshape_maybe(sum(G; dims=1:ndims(hidden(rbm))), batch_size(rbm, v))
    return E_reduced + F_reduced
end

function sample_h_from_v_nopool(rbm::ConvRBM, v::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_sample(hidden(rbm), inputs; β)
end

function sample_h_from_v_pooled(rbm::ConvRBM, v::AbstractArray; β::Real = 1)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    F = RBMs.free_energies(hidden(rbm), inputs; β)
    @assert size(F) == size(inputs) == (size(hidden(rbm))..., output_size(rbm, v)..., batch_size(rbm, v)...)
    selected_size = (size(hidden(rbm))..., batch_size(rbm, v)...)
    k = reshape(cartesian_sample_from_logits_gumbel(-β * F; dims=output_dims(rbm)), selected_size)

    selected_inputs = similar(inputs, selected_size)
    for n in CartesianIndices(batch_size(rbm, v)), μ in CartesianIndices(size(hidden(rbm)))
        @inbounds selected_inputs[μ,n] = inputs[k[μ,n]]
    end

    selected_h = RBMs.transfer_sample(hidden(rbm), selected_inputs; β)
    @assert size(selected_h) == size(selected_inputs) == selected_size

    h = fill!(similar(selected_h, size(F)), 0)
    for n in CartesianIndices(batch_size(rbm, v)), μ in CartesianIndices(size(hidden(rbm)))
        h[k[μ,n]] = selected_h[μ,n]
    end
    return h
end
