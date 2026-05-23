mutable struct History{T}
    under::Array{T}
    ptr::Int
end

function History{T}(size::Int) where {T}
    under = Array{T}(undef, size)
    return History{T}(under, 1)
end

function Add!(buffer::History{T}, data::T) where {T}
    (n,) = size(buffer.under)
    buffer.under[buffer.ptr] = data
    buffer.ptr = buffer.ptr % n + 1
end
