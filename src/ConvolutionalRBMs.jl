module ConvolutionalRBMs

import Random
import Flux
import Zygote
import NNlib
import RestrictedBoltzmannMachines as RBMs

using Random: randn!
using LinearAlgebra: dot
using ValueHistories: MVHistory
using LogExpFunctions: logsumexp, softmax, log1pexp
using RestrictedBoltzmannMachines: AbstractLayer, RBM
using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU
using RestrictedBoltzmannMachines: visible, hidden, weights
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v, transfer_sample
using RestrictedBoltzmannMachines: sample_h_from_h, sample_v_from_v, sample_v_from_h, sample_h_from_v
using RestrictedBoltzmannMachines: energy, free_energy, interaction_energy
using RestrictedBoltzmannMachines: activations_convert_maybe, batch_size
using RestrictedBoltzmannMachines: minibatches, _nobs

include("util.jl")
include("conv.jl")
include("convrbm.jl")
include("BinaryConvRBM.jl")
include("hankel.jl")
include("pooling.jl")

include("zerosum.jl")
include("init.jl")
include("categorical.jl")
include("pcd.jl")
include("pseudolikelihood.jl")

include("denseconv/denseconvrbm.jl")
include("denseconv/zerosum.jl")
include("denseconv/pcd.jl")
include("denseconv/init.jl")
include("denseconv/pool.jl")

end
