function pcd!(rbm::ConvRBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
    vm::AbstractArray = initial_fantasy_v(rbm, data, batchsize),
    callback = Returns(nothing)
)
    @assert length(vsizes(rbm, data).batch_size) == 1 # checks sizes and require single batch dim
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize)
        Δt = @elapsed for (batch_idx, (vd, wd)) in enumerate(batches)
            vm .= RBMs.sample_v_from_v(rbm, vm; steps)
            gs = Zygote.gradient(rbm) do rbm
                contrastive_divergence(rbm, vd, vm; wd)
            end
            # removes keys with no gradient
            ∂ = Base.structdiff(only(gs), NamedTuple{(:stride, :pad, :dilation, :groups)})
            Δ = RBMs.update!(∂, rbm, optim) # update step
            callback(; rbm, epoch, batch_idx, vm, vd, wd, history, ∂, Δ)
            RBMs.update!(rbm, Δ)
            push!(history, :∂, RBMs.gradnorms(∂))
            push!(history, :Δ, RBMs.gradnorms(∂))
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end

function RBMs.update!(∂::NamedTuple, x::ConvRBM, optim)
    for (k, g) in pairs(∂)
        if hasproperty(x, k)
            RBMs.update!(g, getproperty(x, k), optim)
        else
            g .= 0
        end
    end
    return ∂
end

function RBMs.update!(x::ConvRBM, ∂::NamedTuple)
    for (k, Δ) in pairs(∂)
        hasproperty(x, k) && RBMs.update!(getproperty(x, k), Δ)
    end
    return x
end

function contrastive_divergence(
    rbm::ConvRBM, vd::AbstractArray, vm::AbstractArray; wd = nothing, wm = nothing
)
    Fd = RBMs.mean_free_energy(rbm, vd; wts = wd)
    Fm = RBMs.mean_free_energy(rbm, vm; wts = wm)
    return Fd - Fm
end

function RBMs.mean_free_energy(rbm::ConvRBM, v::AbstractArray; wts = nothing)::Number
    F = RBMs.free_energy(rbm, v)
    if isempty(vsizes(rbm, v).batch_size)
        wts::Nothing
        return F
    else
        return RBMs.wmean(F; wts)
    end
end

function initial_fantasy_v(rbm::ConvRBM, data::AbstractArray, batchsize::Int)
    inputs = falses(size(data)[1:(end - 1)]..., batchsize)
    vm = RBMs.transfer_sample(visible(rbm), inputs)
    return oftype(data, vm)
end
