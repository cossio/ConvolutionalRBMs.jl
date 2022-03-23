"""
    cartesian_sample_from_logits(logits; dims=1)

Returns an array `X` of indices sampled from `CartesianIndices(logits)` along
dimensions `dims`, with probabilities `P = softmax(logits; dims)`.
In particular, dimensions `dims` of `X` are singleton.
"""
function cartesian_sample_from_logits(logits::AbstractArray; dims=1)
    return cartesian_sample_from_logits_gumbel(logits; dims)
end

# cartesian_sample_from_logits using Gumbel trick
function cartesian_sample_from_logits_gumbel(logits::AbstractArray; dims = 1)
    z = logits .+ randgumbel.(float(eltype(logits)))
    return argmax(z; dims)
end

"""
    randgumbel(T = Float64)

Generates a random Gumbel variate.
"""
randgumbel(::Type{T} = Float64) where {T} = -log(Random.randexp(T))
