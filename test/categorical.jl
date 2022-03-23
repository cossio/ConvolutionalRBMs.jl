import Random
using Test: @test, @testset, @inferred
using Statistics: mean, std, var
using LogExpFunctions: softmax
using ConvolutionalRBMs: randgumbel, cartesian_sample_from_logits

@testset "randgumbel" begin
    Random.seed!(3)
    data = [@inferred randgumbel() for _ = 1:10^6]
    @test mean(data) ≈ MathConstants.γ rtol=0.01
    @test std(data) ≈ π / √6 rtol=0.01
end

@testset "categorical_sample_from_logits" begin
    Random.seed!(52)
    logits = randn(2, 3, 2)
    @test size(@inferred cartesian_sample_from_logits(logits; dims=2)) == (2,1,2)
    p = mean(cartesian_sample_from_logits(logits; dims=2) .== CartesianIndices(logits) for _ in 1:10^6)
    @test p ≈ softmax(logits; dims=2) rtol=0.01
end
