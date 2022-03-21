function RBMs.initialize!(rbm::ConvRBM, data::AbstractArray; ϵ::Real = 1e-6)
    RBMs.initialize!(visible(rbm), data; ϵ)
    RBMs.initialize!(hidden(rbm))
    RBMs.initialize_w!(rbm, data)
    RBMs.zerosum!(rbm)
    return rbm
end

function RBMs.initialize_w!(rbm::ConvRBM, data::AbstractArray; λ::Real = 0.1, ϵ::Real = 1e-6)
    d = LinearAlgebra.dot(data, data) / prod(vsizes(rbm, data).batch_size)
    randn!(weights(rbm))
    J = prod(kernel_size(rbm))
    K = prod(output_size(rbm, data))
    weights(rbm) .*= sqrt(K/J) * λ / √(d + ϵ)
    return rbm # does not impose zerosum
end

function RBMs.initialize_w!(rbm::ConvRBM; λ::Real = 0.1, ϵ::Real = 1e-6)
    @assert 0 < ϵ < 1/2
    μ = RBMs.transfer_mean(visible(rbm))
    ν = RBMs.transfer_var(visible(rbm))
    d = sum(ν + μ.^2)
    randn!(weights(rbm))
    J = prod(kernel_size(rbm))
    K = prod(output_size(rbm, data))
    weights(rbm) .*= sqrt(K/J) * λ / √(d + ϵ)
    return rbm
end
