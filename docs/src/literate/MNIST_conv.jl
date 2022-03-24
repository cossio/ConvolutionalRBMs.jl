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
using Statistics: mean, var, std
using ValueHistories: MVHistory
using Random: bitrand
using RestrictedBoltzmannMachines: visible, hidden, weights, log_pseudolikelihood, transfer_sample
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
digit = 2
train_x = Array{Float}(train_x[:, :, train_y .== digit] .≥ 0.5)
train_y = train_y[train_y .== digit]
println(length(train_y), " training images (for digit = $digit)")
nothing #hide

# Reshape for convolutional input

train_x = reshape(train_x, 1, 28, 28, :) # channel dims, input dims, batch dims
nothing #hide

# Initialize the convolutional RBM.

rbm = ConvRBMs.BinaryConvRBM(Float, 1, 16, (15,15); pad=:same, pool=true)
RBMs.initialize!(rbm, train_x)
nothing #hide

# Pseudolikelihood before training

idx = rand(1:size(train_x)[end], 256)
mean(@time log_pseudolikelihood(rbm, train_x[:,:,:,idx]))

# Initialize training

batchsize = 256
optim = Flux.ADAM()
vm = transfer_sample(visible(rbm), falses(1, 28, 28, batchsize)) # fantasy chains
history = MVHistory()
nothing #hide

# Train!

@time for iter in 1:20
    ConvRBMs.pcd!(rbm, train_x; vm, history, batchsize, optim, epochs=5)
    lpl = log_pseudolikelihood(rbm, train_x[:, :, :, rand(1:size(train_x)[end], 1024)])
    push!(history, :lpl_ave, mean(lpl))
    push!(history, :lpl_std, std(lpl))
end
nothing #hide

# Plot of log-pseudolikelihood of trian data during learning.

fig = Makie.Figure(resolution=(600,300))
ax = Makie.Axis(fig[1,1], xlabel = "train time", ylabel="pseudolikelihood")
Makie.band!(ax, get(history, :lpl_ave)[1],
    get(history, :lpl_ave)[2] - get(history, :lpl_std)[2]/2,
    get(history, :lpl_ave)[2] + get(history, :lpl_std)[2]/2,
    color=:lightblue
)
Makie.lines!(ax, get(history, :lpl_ave)..., color=:blue)
fig

#=
Now let's generate some random RBM samples.
=#

nrows, ncols = 10, 15
nsteps = 1000
fantasy_F = zeros(nrows*ncols, nsteps)
fantasy_x = bitrand(1,28,28,nrows*ncols)
fantasy_F[:,1] .= RBMs.free_energy(rbm, fantasy_x)
for t in 2:nsteps
    @time fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x)
    fantasy_F[:,t] .= RBMs.free_energy(rbm, fantasy_x)
end
nothing #hide

# Check equilibration of sampling

fig = Makie.Figure(resolution=(400,300))
ax = Makie.Axis(fig[1,1], xlabel="sampling time", ylabel="free energy")
fantasy_F_μ = vec(mean(fantasy_F; dims=1))
fantasy_F_σ = vec(std(fantasy_F; dims=1))
Makie.band!(ax, 1:nsteps, fantasy_F_μ - fantasy_F_σ/2, fantasy_F_μ + fantasy_F_σ/2)
Makie.lines!(ax, 1:nsteps, fantasy_F_μ)
fig

# Plot the resulting samples.

fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig
