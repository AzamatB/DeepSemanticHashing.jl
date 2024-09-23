using Base.Order: Forward
using DataStructures
using Serialization

function load(file_path::AbstractString)
    file_stream = open(file_path, "r")
    structure = deserialize(file_stream)
    close(file_stream)
    return structure
end

function prepare_dataset(
    assignments::AbstractVector{I}, datapoints::AbstractMatrix{F}
) where {I<:Integer,F<:Real}
    d = size(datapoints, 1)
    data_sorted = SortedDict{Int32,Vector{Float32}}(Forward)
    for (idx, cluster) in enumerate(assignments)
        cluster_datapoints_vec = get!(data_sorted, cluster, Float32[])
        append!(cluster_datapoints_vec, @view datapoints[:, idx])
    end

    len = length(data_sorted)
    clusters = Int32[]
    cluster_counts = Int32[]
    cluster_last_indices = Int32[]
    datapoints_vec = Float32[]
    sizehint!(clusters, len)
    sizehint!(cluster_counts, len)
    sizehint!(cluster_last_indices, len)
    sizehint!(datapoints_vec, length(datapoints))
    cluster_last_index = Int32(0)

    for (cluster, cluster_datapoints_vec) in data_sorted
        # skip over singleton clusters
        len = length(cluster_datapoints_vec)
        (len <= d) && continue
        append!(datapoints_vec, cluster_datapoints_vec)
        push!(clusters, cluster)
        cluster_count = len ÷ d
        cluster_last_index += cluster_count
        push!(cluster_counts, cluster_count)
        push!(cluster_last_indices, cluster_last_index)
    end
    datapoints_sorted = reshape(datapoints_vec, d, :)
    return (clusters, cluster_counts, cluster_last_indices, datapoints_sorted)
end


(clusters, cluster_counts, cluster_last_indices, datapoints) = prepare_dataset(assignments, datapoints)
