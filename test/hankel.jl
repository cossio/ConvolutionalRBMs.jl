using Test: @testset, @test, @inferred
using RestrictedBoltzmannMachines: BinaryRBM, inputs_v_to_h, free_energy
using RestrictedBoltzmannMachines: visible, hidden, weights
using ConvolutionalRBMs: hankel_image, hankel_weight, hankel, output_size, out2in
using ConvolutionalRBMs: BinaryConvRBM

@testset "hankel" begin
    channel_sz = (2,3)
    in_sz = (5,5)
    kernel_sz = (3,2)
    out_sz = output_size(kernel_sz, in_sz)
    batch_sz = (2,)
    hidden_sz = (2,3)
    w = randn(channel_sz..., kernel_sz..., hidden_sz...)
    v = randn(channel_sz..., in_sz..., batch_sz...)

    convrbm = BinaryConvRBM(zeros(channel_sz), zeros(hidden_sz), w)

    V = @inferred hankel_image(v, channel_sz, kernel_sz)
    @test size(V) == (channel_sz..., kernel_sz..., out_sz..., batch_sz...)
    for j in CartesianIndices(kernel_sz), k in CartesianIndices(out_sz)
        i = out2in(j, k)
        for b in CartesianIndices(batch_sz), c in CartesianIndices(channel_sz)
            @test V[c,j,k,b] == v[c,i,b]
        end
    end
    denserbm = BinaryRBM(zeros(channel_sz..., kernel_sz...), zeros(hidden_sz...), w)
    @test inputs_v_to_h(denserbm, V) ≈ inputs_v_to_h(convrbm, v)

    W = @inferred hankel_weight(w, channel_sz, in_sz)
    @test size(W) == (channel_sz..., in_sz..., hidden_sz..., out_sz...)
    for j in CartesianIndices(kernel_sz), k in CartesianIndices(out_sz)
        i = out2in(j, k)
        for μ in CartesianIndices(hidden_sz), c in CartesianIndices(channel_sz)
            @test W[c,i,μ,k] == w[c,j,μ]
        end
    end
    densrbm = BinaryRBM(zeros(channel_sz..., in_sz...), zeros(hidden_sz..., out_sz...), W)
    @test inputs_v_to_h(densrbm, v) ≈ inputs_v_to_h(convrbm, v)

    convrbm = BinaryConvRBM(randn(channel_sz...), randn(hidden_sz...), w)
    hankrbm = hankel(convrbm, in_sz)
    @test all(visible(hankrbm).θ .== visible(convrbm).θ)
    @test all(hidden(hankrbm).θ .== hidden(convrbm).θ)
    @test inputs_v_to_h(hankrbm, v) ≈ inputs_v_to_h(convrbm, v)
    @test free_energy(hankrbm, v) ≈ free_energy(convrbm, v)

    convrbm = BinaryConvRBM(randn(channel_sz...), randn(hidden_sz...), w; pad=3, dilation=(2,1), stride=2)
    hankrbm = hankel(convrbm, in_sz)
    @test all(visible(hankrbm).θ .== visible(convrbm).θ)
    @test all(hidden(hankrbm).θ .== hidden(convrbm).θ)
    @test inputs_v_to_h(hankrbm, v) ≈ inputs_v_to_h(convrbm, v)
    @test free_energy(hankrbm, v) ≈ free_energy(convrbm, v)
end

@testset "hankel 1D" begin
    C = 2
    N = 7
    J = 3
    K = only(output_size((J,), (N,)))
    M = 2
    w = randn(C, J, M)
    v = randn(C, N)

    convrbm = BinaryConvRBM(zeros(C), zeros(M), w)

    V = @inferred hankel_image(v, (C,), (J,))
    @test size(V) == (C, J, K)
    for j in 1:J, k in 1:K, c in 1:C
        i = out2in(CartesianIndex(j), CartesianIndex(k))
        @test V[c,j,k] == v[c,i]
    end
    denserbm = BinaryRBM(zeros(C, J), zeros(M), w)
    @test inputs_v_to_h(denserbm, V) ≈ inputs_v_to_h(convrbm, v)

    W = @inferred hankel_weight(w, (C,), (N,))
    @test size(W) == (C, N, M, K)
    for j in CartesianIndices(1:J), k in CartesianIndices(1:K)
        i = out2in(j, k)
        for μ in 1:M, c in 1:C
            @test W[c,i,μ,k] == w[c,j,μ]
        end
    end
    densrbm = BinaryRBM(zeros(C, N), zeros(M, K), W)
    @test inputs_v_to_h(densrbm, v) ≈ inputs_v_to_h(convrbm, v)

    convrbm = BinaryConvRBM(randn(C), randn(M), w)
    hankrbm = hankel(convrbm, (N,))
    @test all(visible(hankrbm).θ .== visible(convrbm).θ)
    @test all(hidden(hankrbm).θ .== hidden(convrbm).θ)
    @test inputs_v_to_h(hankrbm, v) ≈ inputs_v_to_h(convrbm, v)
    @test free_energy(hankrbm, v) ≈ free_energy(convrbm, v)

    convrbm = BinaryConvRBM(randn(C), randn(M), w; stride=3, pad=:same, dilation=2)
    hankrbm = hankel(convrbm, (N,))
    @test all(visible(hankrbm).θ .== visible(convrbm).θ)
    @test all(hidden(hankrbm).θ .== hidden(convrbm).θ)
    @test inputs_v_to_h(hankrbm, v) ≈ inputs_v_to_h(convrbm, v)
    @test free_energy(hankrbm, v) ≈ free_energy(convrbm, v)
end
