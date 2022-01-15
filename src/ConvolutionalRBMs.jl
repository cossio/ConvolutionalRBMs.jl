module ConvolutionalRBMs

import Random
import LinearAlgebra
import Flux
import Zygote
import NNlib
import LogExpFunctions
import RestrictedBoltzmannMachines as RBMs

using Random: randn!
using ValueHistories: MVHistory
using RestrictedBoltzmannMachines: AbstractLayer, RBM, Potts
using RestrictedBoltzmannMachines: visible, hidden, weights
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v, transfer_sample
using RestrictedBoltzmannMachines: sample_h_from_h, sample_v_from_v, sample_v_from_h, sample_h_from_v
using RestrictedBoltzmannMachines: energy, free_energy, interaction_energy
using RestrictedBoltzmannMachines: activations_convert_maybe, batch_size
using RestrictedBoltzmannMachines: minibatches, _nobs

include("util.jl")
include("nnlib.jl")
include("convrbm.jl")
include("BinaryConvRBM.jl")
include("denseconvrbm.jl")
include("train/zerosum_conv.jl")
include("train/zerosum_denseconv.jl")
include("train/pcd_conv.jl")
include("train/pcd_denseconv.jl")
include("train/init_conv.jl")
include("train/init_denseconv.jl")
include("pseudolikelihood.jl")

end
