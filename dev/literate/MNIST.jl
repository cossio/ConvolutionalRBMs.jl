#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

import Makie
import CairoMakie
import MLDatasets
import Flux
import RestrictedBoltzmannMachines as RBMs
import ConvolutionalRBMs as ConvRBMs
using Statistics: mean
using ValueHistories: MVHistory
using Random: bitrand
nothing #hide

#=
Useful function to plot MNIST digits.
=#

"""
    imggrid(A)

Given a four dimensional tensor `A` of size `(width, height, ncols, nrows)`
containing `width x height` images in a grid of `nrows x ncols`, this returns
a matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap
to display all images.
"""
function imggrid(A::AbstractArray{<:Any,4})
    return reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))
end

#=
Load MNIST dataset.
=#

Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
train_x = Array{Float}(train_x[:, :, train_y .== 8] .≥ 0.5)
tests_x = Array{Float}(tests_x[:, :, tests_y .== 8] .≥ 0.5)
train_y = train_y[train_y .== 8]
tests_y = tests_y[tests_y .== 8]
train_nsamples = length(train_y)
tests_nsamples = length(tests_y)
(train_nsamples, tests_nsamples)

# Reshape for convolutional input

train_x = reshape(train_x, 1, 28, 28, :) # channel dims, input dims, batch dims
tests_x = reshape(tests_x, 1, 28, 28, :)
nothing #hide

# Initialize the convolutional RBM.

rbm = ConvRBMs.BinaryConvRBM(Float, 1, 50, (15,15))
RBMs.initialize!(rbm, train_x)
nothing #hide

# Train

batchsize = 128
vm = bitrand(1, 28, 28, batchsize) # Initialize fantasy chains
history = RBMs.pcd!(rbm, train_x; epochs=100, vm, batchsize)
nothing #hide

#=
Now let's generate some random RBM samples.
=#

nrows, ncols = 10, 15
@time fantasy_x = RBMs.sample_v_from_v(rbm, bitrand(1,28,28,nrows*ncols); steps=10000)
nothing #hide

# Plot the resulting samples.

fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig
