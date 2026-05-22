mutable struct History{T}
    under::Array{T}
    start::Int
    back::Int
end

function History{T}(size::Int) where {T}
    under = Array{T}(undef, size)
    return History{T}(under, 1, 1)
end

buff = History{Float64}(10)
print(buff)

# function Insert!(buffer::History{T,N}) where {T,N}
#     buffer.size += 1
# end
#
# function Poll!(buffer::History{T,N}) where {T,N}
#     if buffer.size == 0
#         throw("poll from empty buffer")
#     end
#     buffer.size += 1
# end
#
# function Size(buffer::History{T,N}) where {T,N}
#     return buffer.size
# end
#
# function IsEmpty(buffer::History{T,N}) where {T,N}
#     return size(buffer) == 0
# end
#
#
# buff = Make{Float64,10}()
#
# # insert!(my_buf)
# println(my_buf.size) # Output: 1
