function RBMs.zerosum!(rbm::ConvRBM)
    RBMs.zerosum_visible!(rbm)
    RBMs.zerosum_hidden!(rbm)
    return rbm
end

RBMs.zerosum_visible!(rbm::ConvRBM{<:AbstractLayer, <:AbstractLayer}) = rbm
RBMs.zerosum_hidden!(rbm::ConvRBM{<:AbstractLayer, <:AbstractLayer})  = rbm

function RBMs.zerosum_visible!(rbm::ConvRBM{<:RBMs.Potts, <:AbstractLayer})
    RBMs.zerosum!(visible(rbm))
    RBMs.zerosum!(weights(rbm); dims = 1)
    return nothing
end

function RBMs.zerosum_hidden!(rbm::ConvRBM{<:AbstractLayer, <:RBMs.Potts})
    RBMs.zerosum!(hidden(rbm))
    RBMs.zerosum!(weights(rbm); dims = ndims(weights(rbm)) - ndims(hidden(rbm)) + 1)
    return nothing
end

function RBMs.zerosum!(rbm::ConvRBM{<:RBMs.Potts, <:RBMs.Potts})
    RBMs.zerosum!(visible(rbm))
    RBMs.zerosum!(hidden(rbm))
    RBMs.zerosum!(weights(rbm); dims = 1)
    RBMs.zerosum!(weights(rbm); dims = ndims(weights(rbm)) - ndims(hidden(rbm)) + 1)
    return nothing
end
