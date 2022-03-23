function RBMs.initialize!(
    rbm::DenseConvRBM, data_dense::AbstractArray, data_conv::AbstractArray;
    ϵ::Real = 1e-6, λ::Real = 0.1
)
    RBMs.initialize!(rbm.visible_dense, data_dense; ϵ)
    RBMs.initialize!(rbm.visible_conv, data_conv; ϵ)
    RBMs.initialize!(hidden(rbm))
    RBMs.initialize_w!(parts(rbm).dense, data_dense; λ = λ/2)
    RBMs.initialize_w!(parts(rbm).conv, data_conv; λ = λ/2)
    RBMs.zerosum!(rbm)
    return rbm
end
