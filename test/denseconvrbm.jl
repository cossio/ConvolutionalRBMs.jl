import RestrictedBoltzmannMachines as RBMs
import ConvolutionalRBMs as ConvRBMs
using Test: @test, @testset, @inferred, @test_throws
using Random: bitrand
using LogExpFunctions: logsumexp
using RestrictedBoltzmannMachines: energy, free_energy, interaction_energy
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_h
using RestrictedBoltzmannMachines: sample_v_from_h, sample_h_from_v
using RestrictedBoltzmannMachines: Binary, visible, hidden, weights
using RestrictedBoltzmannMachines: reshape_maybe
using ConvolutionalRBMs: vsizes, hsizes, parts, output_size, inputs_h_to_v_dense, inputs_h_to_v_conv

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

    v_dense = bitrand(N..., B...)
    v_conv = bitrand(C..., (J .+ K .- 1)..., B...)
    h = bitrand(M..., K..., B...)
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

@testset "pooling DenseConvRBM" begin
    rbm = ConvRBMs.DenseConvRBM(
        Binary(randn(2)), # visible dense
        Binary(randn(3)), # visible conv
        Binary(randn(2)), # hidden
        randn(2, 2), # weights dense
        randn(3, 3, 2); # weights conv
        pool = true
    )

    @test @inferred(free_energy(rbm, bitrand(2), bitrand(3,4))) isa Number
    @test size(@inferred free_energy(rbm, bitrand(2,5), bitrand(3,4,5))) == (5,)
    @test size(@inferred sample_h_from_v(rbm, bitrand(2), bitrand(3,4))) == (2,2)

    h1 = @inferred sample_h_from_v(rbm, bitrand(2), bitrand(3,4))
    h2 = @inferred sample_h_from_h(rbm, h1)
    @test size(h1) == size(h2) == (2,2)
    # pooling: at most one hidden unit active per group
    @test all(sum(h1; dims=2) .≤ 1)
    @test all(sum(h2; dims=2) .≤ 1)
end

@testset "free energy (not pooled)" begin
    rbm = ConvRBMs.DenseConvRBM(
        Binary(randn(2)), # visible dense
        Binary(randn(3)), # visible conv
        Binary(randn(2)), # hidden
        randn(2, 2), # weights dense
        randn(3, 3, 2); # weights conv
        pool = false
    )
    β = rand()
    v_dense, v_conv = bitrand(2), bitrand(3,4)
    hs = [[h1 h2; h3 h4] for h1=0:1, h2=0:1, h3=0:1, h4=0:1]
    @test -β * free_energy(rbm, v_dense, v_conv; β) ≈ logsumexp(-β * energy(rbm, v_dense, v_conv, h) for h in hs)
end

@testset "free energy (pooled)" begin
    rbm = ConvRBMs.DenseConvRBM(
        Binary(randn(2)), # visible dense
        Binary(randn(3)), # visible conv
        Binary(randn(1)), # hidden
        randn(2, 1), # weights dense
        randn(3, 3, 1); # weights conv
        pool = true
    )
    β = rand()
    v_dense, v_conv = bitrand(2), bitrand(3,4)
    hs = [
        [0 1],
        [1 0],
    ]
    K = prod(output_size(rbm, v_conv))
    M = prod(size(hidden(rbm)))
    @test exp(-β * free_energy(rbm, v_dense, v_conv; β)) ≈ (
        sum(exp(-β * energy(rbm, v_dense, v_conv, h)) for h in hs) +
        K^M * exp(-β * energy(rbm, v_dense, v_conv, [0 0]))
    )
end
