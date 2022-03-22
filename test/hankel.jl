using Test: @testset, @test, @inferred
using RestrictedBoltzmannMachines: BinaryRBM, inputs_v_to_h
using ConvolutionalRBMs: hankel_image, hankel_weight, output_size, out2in
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
end
