import RestrictedBoltzmannMachines as RBMs
import ConvolutionalRBMs as ConvRBMs

using Test: @test, @testset, @inferred, @test_throws
using Random: bitrand, randn!
using LinearAlgebra: dot
using LogExpFunctions: logsumexp, logaddexp
using RestrictedBoltzmannMachines: visible, hidden, weights
using RestrictedBoltzmannMachines: energy, free_energy, interaction_energy
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_h, sample_v_from_h, sample_h_from_v
using ConvolutionalRBMs: vsizes, hsizes, BinaryConvRBM, output_size, out2in, hankel

@testset "free energy" begin
    rbm = BinaryConvRBM(randn(1), randn(1), randn(1,1,1); pool=false)
    β = rand()
    v = bitrand(1,2)
    hs = [[h1 h2] for h1=0:1, h2=0:1]
    -β * free_energy(rbm, v; β) ≈ logsumexp(-β * energy(rbm, v, h) for h in hs)

    rbm = BinaryConvRBM(randn(1), randn(1), randn(1,1,1); pool=true)
    β = rand()
    v = bitrand(1,2)
    hs = [[0 1], [1 0]]
    K = prod(output_size(rbm,v))
    M = prod(size(hidden(rbm)))
    @test exp(-β * free_energy(rbm, v; β)) ≈ sum(exp(-β * energy(rbm, v, h)) for h in hs) + K^M * exp(-β * energy(rbm, v, [0 0]))
end
