function RBMs.log_pseudolikelihood(rbm::ConvRBM, v::AbstractArray; β::Real=true)
    return log_pseudolikelihood_stoch(rbm, v; β)
end

function log_pseudolikelihood_stoch(rbm::ConvRBM, v::AbstractArray; β::Real=true)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    sites = [
        rand(CartesianIndices(sitesize(rbm)))
        for _ in CartesianIndices(batch_size(rbm, v))
    ]
    return log_pseudolikelihood_sites(rbm, v, sites; β)
end

sitesize(rbm::ConvRBM) = size(rbm.w)[1:(end - ndims(rbm.hidden))]
sitesize(rbm::ConvRBM{<:RBMs.Potts}) = size(rbm.w)[2:(end - ndims(rbm.hidden))]

function log_pseudolikelihood_sites(
    rbm::ConvRBM{<:Potts},
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real=true
)
    ΔE = substitution_matrix_sites(rbm, v, sites; β)
    lPL = -logsumexp(-β * ΔE; dims=1)
    return reshape(lPL, batch_size(rbm, v))
end

function substitution_matrix_sites(
    rbm::ConvRBM{<:Potts},
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real = true
)
    E_ = zeros(RBMs.colors(visible(rbm)), batch_size(rbm, v)...)
    for x in 1:RBMs.colors(visible(rbm))
        v_ = copy(v)
        for (b, i) in pairs(sites)
            v_[:, i, b] .= false
            v_[x, i, b] = true
        end
        selectdim(E_, 1, x) .= free_energy(rbm, v_; β)
    end
    c = RBMs.onehot_decode(v)
    E = [E_[c[i, b], b] for (b, i) in pairs(sites)]
    return E_ .- reshape(E, 1, batch_size(rbm, v)...)
end

function log_pseudolikelihood_sites(
    rbm::ConvRBM{<:RBMs.Binary},
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real = true
)
    v_ = copy(v)
    for (b, i) in pairs(sites)
        v_[i, b] = 1 - v_[i, b]
    end
    F = free_energy(rbm, v; β)
    F_ = free_energy(rbm, v_; β)
    return -log1pexp.(β * (F - F_))
end

function log_pseudolikelihood_sites(
    rbm::ConvRBM{<:RBMs.Spin},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex};
    β::Real = true
)
    v_ = copy(v)
    for (b, i) in pairs(sites)
        v_[i, b] = -v_[i, b]
    end
    F = free_energy(rbm, v; β)
    F_ = free_energy(rbm, v_; β)
    return -log1pexp.(β * (F - F_))
end
