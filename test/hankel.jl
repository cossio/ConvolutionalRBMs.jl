using Test: @testset, @test, @inferred
using ConvolutionalRBMs: hankel, output_size

@testset "hankel" begin
    channel_sz = (3,2)
    in_sz = (5,5)
    kernel_sz = (3,2)
    out_sz = output_size(kernel_sz, in_sz)
    batch_sz = (2,)
    img = randn(channel_sz..., in_sz..., batch_sz...)
    H = @inferred hankel(img, channel_sz, kernel_sz)
    @test ndims(H) == ndims(img) + length(kernel_sz)
    @test size(H) == (channel_sz..., kernel_sz..., out_sz..., batch_sz...)
    for c in CartesianIndices(channel_sz), j in CartesianIndices(kernel_sz), k in CartesianIndices(out_sz), b in CartesianIndices(batch_sz)
        i = CartesianIndex(Tuple(j) .+ Tuple(k) .- 1)
        @test H[c,j,k,b] == img[c,i,b]
    end
end
