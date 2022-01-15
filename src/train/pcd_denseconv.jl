function pcd!(
    rbm::DenseConvRBM,
    data_dense::AbstractArray,
    data_conv::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
    vm_dense::AbstractArray = initial_fantasy_v(parts(rbm).dense, data_dense, batchsize),
    vm_conv::AbstractArray = initial_fantasy_v(parts(rbm).conv, data_conv, batchsize)
)
    @assert size(data_dense) == (size(rbm.visible_dense)..., _nobs(data_dense))
    @assert only(vsizes(parts(rbm).conv, data_conv).batch_size) == _nobs(data_dense)
    @assert size(vm_dense) == (size(rbm.visible_dense)..., batchsize)
    @assert only(vsizes(parts(rbm).conv, vm_conv).batch_size) == batchsize
    @assert isnothing(wts) || _nobs(data_dense) == _nobs(wts)

    for epoch in 1:epochs
        batches = minibatches(data_dense, data_conv, wts; batchsize)
        Δt = @elapsed for (vd_dense, vd_conv, wd) in batches
            _vm_dense, _vm_conv = RBMs.sample_v_from_v(rbm, vm_dense, vm_conv; steps)
            vm_dense .= _vm_dense
            vm_conv  .= _vm_conv
            gs = Zygote.gradient(rbm) do rbm
                contrastive_divergence(rbm, vd_dense, vd_conv, vm_dense, vm_conv; wd)
            end
            # removes keys with no gradient
            ∂ = Base.structdiff(only(gs), NamedTuple{(:stride, :pad, :dilation, :groups)})
            push!(history, :∂, RBMs.gradnorms(∂))
            RBMs.update!(rbm, RBMs.update!(∂, rbm, optim))
            push!(history, :Δ, RBMs.gradnorms(∂))
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end

function contrastive_divergence(
    rbm::DenseConvRBM,
    vd_dense::AbstractArray, vd_conv::AbstractArray,
    vm_dense::AbstractArray, vm_conv::AbstractArray;
    wd = nothing, wm = nothing
)
    Fd = mean_free_energy(rbm, vd_dense, vd_conv; wts = wd)
    Fm = RBMs.mean_free_energy(rbm, vm_dense, vm_conv; wts = wm)
    return Fd - Fm
end

function mean_free_energy(
    rbm::DenseConvRBM, v_dense::AbstractArray, v_conv::AbstractArray; wts = nothing
)::Number
    F = RBMs.free_energy(rbm, v_dense, v_conv)
    if isempty(batch_size(rbm, v_dense, v_cons))
        wts::Nothing
        return F
    else
        return RBMs.wmean(F; wts)
    end
end

function RBMs.update!(∂::NamedTuple, x::DenseConvRBM, optim)
    for (k, g) in pairs(∂)
        if hasproperty(x, k)
            RBMs.update!(g, getproperty(x, k), optim)
        else
            g .= 0
        end
    end
    return ∂
end

function RBMs.update!(x::DenseConvRBM, ∂::NamedTuple)
    for (k, Δ) in pairs(∂)
        hasproperty(x, k) && RBMs.update!(getproperty(x, k), Δ)
    end
    return x
end

function initial_fantasy_v(rbm::RBM, data::AbstractArray, batchsize::Int)
    inputs = falses(size(data)[1:(end - 1)]..., batchsize)
    vm = RBMs.transfer_sample(visible(rbm), inputs)
    return oftype(data, vm)
end
