using Test: @test, @testset, @inferred, @test_throws
using Random: bitrand
import RestrictedBoltzmannMachines as RBMs
import ConvolutionalRBMs as ConvRBMs
using RestrictedBoltzmannMachines: energy, free_energy, interaction_energy
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_h
using RestrictedBoltzmannMachines: sample_v_from_h, sample_h_from_v
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: visible, hidden, weights
using RestrictedBoltzmannMachines: reshape_maybe
using ConvolutionalRBMs: vsizes, hsizes, parts, inputs_h_to_v_dense, inputs_h_to_v_conv

@testset "DenseConvRBM" begin
    N = (4,)
    C = (2,)
    J = (3,)
    K = (5,)
    M = (7)
    B = (11,)

    rbm = ConvRBMs.DenseConvRBM(
        Binary(randn(N...)), # visible dense
        Binary(randn(C...)), # visible conv
        Binary(randn(M...)), # hidden
        randn(N..., M...), # weights dense
        randn(C..., J..., M...) # weights conv
    )

    v_dense = Array{Float64}(bitrand(N..., B...))
    v_conv = Array{Float64}(bitrand(C..., (J .+ K .- 1)..., B...))
    h = Array{Float64}(bitrand(M..., K..., B...))
    @test size(@inferred inputs_v_to_h(rbm, v_dense, v_conv)) == size(h)
    @test size(@inferred inputs_h_to_v_dense(rbm, h)) == size(v_dense)
    @test size(@inferred inputs_h_to_v_conv(rbm, h)) == size(v_conv)

    h_reduced = reshape(sum(h; dims=length(M) .+ (1:length(K))), M..., B...)
    E_dense_w = interaction_energy(parts(rbm).dense, v_dense, h_reduced)
    E_conv_w = interaction_energy(parts(rbm).conv, v_conv, h)
    @test @inferred(interaction_energy(rbm, v_dense, v_conv, h)) ≈ E_conv_w + E_dense_w

    E_conv = energy(parts(rbm).conv, v_conv, h)
    E_dense = energy(parts(rbm).dense, v_dense, h_reduced)
    E_h = energy(hidden(rbm), h)
    E_h = reshape_maybe(sum(E_h; dims=1:length(J)), B)
    @test @inferred(energy(rbm, v_dense, v_conv, h)) ≈ E_conv .+ E_dense .- E_h

    @test size(@inferred free_energy(rbm, v_dense, v_conv)) == B
end
