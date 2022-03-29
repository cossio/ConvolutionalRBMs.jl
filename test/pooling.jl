import RestrictedBoltzmannMachines as RBMs
import ConvolutionalRBMs as ConvRBMs

using Test: @test, @testset, @inferred, @test_throws
using Random: bitrand, randn!
using LinearAlgebra: dot
using RestrictedBoltzmannMachines: visible, hidden, weights, Binary
using RestrictedBoltzmannMachines: energy, free_energy, interaction_energy
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_h, sample_v_from_h, sample_h_from_v
using ConvolutionalRBMs: vsizes, hsizes, BinaryConvRBM, output_size, out2in

@testset "pooling ConvRBM" begin
    rbm = BinaryConvRBM(3, 2, 3; pool=true)
    randn!(weights(rbm))
    randn!(visible(rbm).θ)
    randn!(hidden(rbm).θ)

    v = bitrand(3,5)
    @test @inferred(free_energy(rbm, v)) isa Number
    v = bitrand(3,5,2)
    @test size(@inferred free_energy(rbm, v)) == (2,)
    v = bitrand(3,5)
    @test @inferred(output_size(rbm, v)) == (3,)

    v = bitrand(3,5,2)
    @test size(@inferred sample_h_from_v(rbm, v)) == (2,3,2)
    @test @inferred(output_size(rbm, v)) == (3,)

    h1 = @inferred sample_h_from_v(rbm, v)
    h2 = @inferred sample_h_from_h(rbm, h1)
    @test size(h1) == size(h2) == (2,3,2)
    # pooling: at most one hidden unit active per group
    @test all(sum(h1; dims=2) .≤ 1)
    @test all(sum(h2; dims=2) .≤ 1)
end
