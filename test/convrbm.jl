import RestrictedBoltzmannMachines as RBMs
import ConvolutionalRBMs as ConvRBMs

using Test: @test, @testset, @inferred, @test_throws
using Random: bitrand, randn!
using LinearAlgebra: dot
using RestrictedBoltzmannMachines: visible, hidden, weights
using RestrictedBoltzmannMachines: energy, free_energy, interaction_energy
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_h, sample_v_from_h, sample_h_from_v
using ConvolutionalRBMs: vsizes, hsizes, BinaryConvRBM, output_size, out2in

@testset "BinaryConvRBM" begin
    rbm = BinaryConvRBM(3, 2, 3)
    randn!(weights(rbm))
    randn!(visible(rbm).θ)
    randn!(hidden(rbm).θ)

    @test ConvRBMs.channel_size(rbm) == (3,)
    @test ConvRBMs.channel_length(rbm) == 3
    @test ConvRBMs.channel_ndims(rbm) == 1
    @test ConvRBMs.kernel_size(rbm) == (3,)
    @test ConvRBMs.kernel_ndims(rbm) == 1
    @test ConvRBMs.kernel_dims(rbm) == 2:2
    @test size(hidden(rbm)) == (2,)

    v = Array{Float64}(bitrand(3,5))
    h = Array{Float64}(bitrand(2,3))
    @test @inferred(energy(rbm, v, h)) isa Number
    @test @inferred(free_energy(rbm, v)) isa Number
    @test dot(@inferred(inputs_v_to_h(rbm, v)), h) ≈ -interaction_energy(rbm, v, h)
    @test dot(@inferred(inputs_h_to_v(rbm, h)), v) ≈ -interaction_energy(rbm, v, h)
    @test @inferred(output_size(rbm, v)) == (3,)

    v = Array{Float64}(bitrand(3,5,2))
    h = Array{Float64}(bitrand(2,3))
    @test size(@inferred energy(rbm, v, h)) == (2,)
    @test size(@inferred free_energy(rbm, v)) == (2,)
    @test vec(sum(@inferred(inputs_v_to_h(rbm, v)) .* h; dims=1:2)) ≈ -interaction_energy(rbm, v, h)
    @test vec(sum(@inferred(inputs_h_to_v(rbm, h)) .* v; dims=1:2)) ≈ -interaction_energy(rbm, v, h)
    @test @inferred(output_size(rbm, v)) == (3,)

    v = Array{Float64}(bitrand(3,5))
    h = Array{Float64}(bitrand(2,3,2))
    @test size(@inferred energy(rbm, v, h)) == (2,)
    @test vec(sum(@inferred(inputs_v_to_h(rbm, v)) .* h; dims=1:2)) ≈ -interaction_energy(rbm, v, h)
    @test vec(sum(@inferred(inputs_h_to_v(rbm, h)) .* v; dims=1:2)) ≈ -interaction_energy(rbm, v, h)
    @test @inferred(output_size(rbm, v)) == (3,)

    v = Array{Float64}(bitrand(3,5,2))
    h = Array{Float64}(bitrand(2,3,2))
    @test size(@inferred energy(rbm, v, h)) == (2,)
    @test vec(sum(@inferred(inputs_v_to_h(rbm, v)) .* h; dims=1:2)) ≈ -interaction_energy(rbm, v, h)
    @test vec(sum(@inferred(inputs_h_to_v(rbm, h)) .* v; dims=1:2)) ≈ -interaction_energy(rbm, v, h)
    @test @inferred(output_size(rbm, v)) == (3,)

    v = Array{Float64}(bitrand(3,5,3))
    h = Array{Float64}(bitrand(2,3,2))
    @test_throws Exception energy(rbm, v, h)

    v = Array{Float64}(bitrand(3,5,2))
    h = Array{Float64}(bitrand(2,3,2))
    @test size(@inferred inputs_v_to_h(rbm, v)) == size(h)
    @test size(@inferred inputs_h_to_v(rbm, h)) == size(v)
    @test size(@inferred sample_h_from_v(rbm, v)) == size(h)
    @test size(@inferred sample_v_from_h(rbm, h)) == size(v)
    @test size(@inferred sample_v_from_v(rbm, v)) == size(v)
    @test size(@inferred sample_h_from_h(rbm, h)) == size(h)
    @test @inferred(output_size(rbm, v)) == (3,)
end

@testset "samepad" begin
    # 1 convolved dimension
    for c in (2,3,(2,3)), k in (2,3), b in ((),2,3,(2,3)), dil in (1,2,3), m in (2,3,(2,3)), n in (4,7)
        rbm = BinaryConvRBM(c,m,k; pad=:same, dilation=dil)
        randn!(weights(rbm))
        v = bitrand(c..., n..., b...)
        @test size(@inferred inputs_v_to_h(rbm, v)) == (m..., n..., b...)
        @test @inferred(output_size(rbm, v)) == (n...,)
        @test size(@inferred sample_v_from_v(rbm, v)) == size(v)
    end

    # 2 convolved dimension
    for c in (2,3,(2,3)), k in ((2,2),(3,3),(2,3)), b in ((),2,3,(2,3)), dil in ((1,1),(2,2),(1,2)), m in (2,3), n in ((7,7),(8,8))
        rbm = BinaryConvRBM(c,m,k; pad=:same, dilation=dil)
        randn!(weights(rbm))
        v = bitrand(c..., n..., b...)
        @test size(@inferred inputs_v_to_h(rbm, v)) == (m..., n..., b...)
        @test @inferred(output_size(rbm, v)) == (n...,)
        @test size(@inferred sample_v_from_v(rbm, v)) == size(v)
    end
end

@testset "inputs_v_to_h" begin
    channel_sz = (3,5)
    kernel_sz = (2,3,2)
    hidden_sz = (5,2)
    rbm = BinaryConvRBM(channel_sz, hidden_sz, kernel_sz)
    randn!(weights(rbm))
    in_sz = (7,6,2)
    batch_sz = (2,3)
    v = Array{Float64}(bitrand(channel_sz..., in_sz..., batch_sz...))
    out_sz = output_size(rbm, v)
    Ih = zeros(hidden_sz..., out_sz..., batch_sz...)
    for j in CartesianIndices(kernel_sz), k in CartesianIndices(out_sz)
        i = out2in(j, k)
        for b in CartesianIndices(batch_sz), μ in CartesianIndices(hidden_sz), c in CartesianIndices(channel_sz)
            Ih[μ,k,b] += weights(rbm)[c,j,μ] * v[c,i,b]
        end
    end
    @test Ih ≈ inputs_v_to_h(rbm, v)

    h = Array{Float64}(bitrand(hidden_sz..., out_sz..., batch_sz...))
    Iv = zeros(channel_sz..., in_sz..., batch_sz...)
    for j in CartesianIndices(kernel_sz), k in CartesianIndices(out_sz)
        i = out2in(j, k)
        for b in CartesianIndices(batch_sz), μ in CartesianIndices(hidden_sz), c in CartesianIndices(channel_sz)
            Iv[c,i,b] += weights(rbm)[c,j,μ] * h[μ,k,b]
        end
    end
    @test Iv ≈ inputs_h_to_v(rbm, h)
end
