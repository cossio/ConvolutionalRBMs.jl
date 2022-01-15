"""
    BinaryConvRBM(visible_fields, hidden_fields, w)

Convolutional RBM with binary visible and hidden units,
with fields `a` and `b` and weights `w`.
"""
function BinaryConvRBM(
    visible_fields::AbstractArray,
    hidden_fields::AbstractArray,
    w::AbstractArray;
    kwargs...
)
    return ConvRBM(RBMs.Binary(visible_fields), RBMs.Binary(hidden_fields), w; kwargs...)
end

"""
    BinaryConvRBM(visible_size, hidden_size, kernel_size)
    BinaryConvRBM(T, visible_size, hidden_size, kernel_size)

Convolutional binary RBM with given dimensions, and parameters initialized to zero
of type `T` (= `Float64` by default).
"""
function BinaryConvRBM(
    ::Type{T},
    visible_size::Union{Int,TupleN{Int}},
    hidden_size::Union{Int,TupleN{Int}},
    kernel_size::Union{Int,TupleN{Int}};
    kwargs...
) where {T}
    visible_fields = zeros(T, visible_size...)
    hidden_fields = zeros(T, hidden_size...)
    w = zeros(T, visible_size..., kernel_size..., hidden_size...)
    return BinaryConvRBM(visible_fields, hidden_fields, w; kwargs...)
end

function BinaryConvRBM(
    visible_size::Union{Int,TupleN{Int}},
    hidden_size::Union{Int,TupleN{Int}},
    kernel_size::Union{Int,TupleN{Int}};
    kwargs...
)
    return BinaryConvRBM(Float64, visible_size, hidden_size, kernel_size; kwargs...)
end
