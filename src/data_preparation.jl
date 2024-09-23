using Serialization

function load(file_path::AbstractString)
    file_stream = open(file_path, "r")
    structure = deserialize(file_stream)
    close(file_stream)
    return structure
end
d = size(datapoints, 1)
data_grouped = Dict{Int32,Vector{Float32}}()
for (i, a) in enumerate(assignments)
    vec = get!(data_grouped, a, Float32[])
    append!(vec, @view datapoints[:, i])
end

data_grouped_filtered = Dict{Int32, Matrix{Float32}}()
for (key, vec) in data_grouped
    (length(vec) <= d) && continue
    data_grouped_filtered[key] = reshape(vec, d, :)
end
