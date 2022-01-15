using Test: @testset, @test
import ConvolutionalRBMs as ConvRBMs
import Zygote
import FiniteDifferences
using FiniteDifferences: central_fdm

@testset "conv n=$n C=$C M=$M" for n in 1:3, C in 1:3, B in 1:3, M=1:3
    kernel = (rand(1:4, n)...,)
    output = (rand(1:4, n)...,)
    inputs = output .+ kernel .- 1
    w = randn(C, kernel..., M)
    v = randn(C, inputs..., B)
    h = randn(M, output..., B)
    I_v2h = zeros(M, output..., B)
    I_h2v = zeros(C, inputs..., B)
    for b in 1:B, k in CartesianIndices(output), μ = 1:M, j in CartesianIndices(kernel), c in 1:C
        i = CartesianIndex(Tuple(j) .+ Tuple(k) .- 1)
        @assert i ∈ CartesianIndices(inputs)
        I_v2h[μ,k,b] += w[c,j,μ] * v[c,i,b]
        I_h2v[c,i,b] += w[c,j,μ] * h[μ,k,b]
    end
    # need rtol here because the (non-compensated) sum loop just above accumulates some error
    @test ConvRBMs.conv_v2h(w, v) ≈ I_v2h rtol=1e-6
    @test ConvRBMs.conv_h2v(w, h) ≈ I_h2v rtol=1e-6

    ∂w, ∂v = Zygote.gradient(w, v) do w, v
        sum(ConvRBMs.conv_v2h(w, v))
    end
    Δw, Δv = FiniteDifferences.grad(central_fdm(5,1), (w,v) -> sum(ConvRBMs.conv_v2h(w,v)), w,v)
    @test ∂w ≈ Δw rtol=1e-6
    @test ∂v ≈ Δv rtol=1e-6

    ∂w, ∂h = Zygote.gradient(w, h) do w, h
        sum(ConvRBMs.conv_h2v(w, h))
    end
    Δw, Δh = FiniteDifferences.grad(central_fdm(5,1), (w,h) -> sum(ConvRBMs.conv_h2v(w,h)), w,h)
    @test ∂w ≈ Δw rtol=1e-6
    @test ∂h ≈ Δh rtol=1e-6
end
