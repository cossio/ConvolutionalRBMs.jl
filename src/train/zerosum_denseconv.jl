function RBMs.zerosum!(rbm::DenseConvRBM)
    RBMs.zerosum_visible!(rbm)
    RBMs.zerosum_hidden!(rbm)
    return rbm
end

function RBMs.zerosum_visible!(rbm::DenseConvRBM)
    if rbm.visible_dense isa RBMs.Potts
        RBMs.zerosum!(rbm.visible_dense)
        RBMs.zerosum!(rbm.w_dense; dims = 1)
    end
    if rbm.visible_conv isa RBMs.Potts
        RBMs.zerosum!(rbm.visible_conv)
        RBMs.zerosum!(rbm.w_conv; dims = 1)
    end
    return rbm
end

function RBMs.zerosum_hidden!(rbm::DenseConvRBM)
    if hidden(rbm) isa RBMs.Potts
        RBMs.zerosum!(hidden(rbm))
        RBMs.zerosum!(rbm.w_dense; dims = ndims(rbm.w_dense) - ndims(hidden(rbm)) + 1)
        RBMs.zerosum!(rbm.w_conv; dims = ndims(rbm.w_conv) - ndims(hidden(rbm)) + 1)
    end
    return rbm
end
