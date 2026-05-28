mutable struct History{T}
    under::Array{T}
    ptr::Int32
    full::Bool
end

function History{T}(size::Int32) where {T}
    under = Array{T}(undef, size)
    return History{T}(under, 1, false)
end

function Add!(buffer::History{T}, data::T) where {T}
    (n,) = size(buffer.under)
    buffer.under[buffer.ptr] = data

    if buffer.ptr == n
        buffer.full = true
    end
    buffer.ptr = buffer.ptr % n + 1
end

function Sample!(count::Int32, buffer::History{T}) where {T}
    if buffer.full
        rand(buffer.under, count)
    elseif buffer.ptr == 1 && !buffer.full
        throw("cannot sample from empty history")
    else
        buffer.under[rand(1:buffer.ptr-1, count)]
    end
end
