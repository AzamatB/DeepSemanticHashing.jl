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
    data_sorted = SortedDict{I,Vector{Float32}}(Forward)
    for (idx, cluster) in enumerate(assignments)
        cluster_datapoints_vec = get!(data_sorted, cluster, Float32[])
        append!(cluster_datapoints_vec, @view datapoints[:, idx])
    end

    len = length(data_sorted)
    clusters = I[]
    cluster_datapoints = Matrix{Float32}[]
    sizehint!(clusters, len)
    sizehint!(cluster_datapoints, len)

    for (cluster, cluster_datapoints_vec) in data_sorted
        len = length(cluster_datapoints_vec)
        # skip over singleton clusters
        (len <= d) && continue
        push!(clusters, cluster)
        cluster_datapoints_mat = reshape(cluster_datapoints_vec, d, :)
        # normalize the vectors (columns) to have a unit length in L₁-norm (all values
        # already are non-negative)
        cluster_datapoints_mat ./= sum(cluster_datapoints_mat; dims=1)
        push!(cluster_datapoints, cluster_datapoints_mat)
    end
    return (clusters, cluster_datapoints)
end

function load_dataset(device::MLDataDevices.AbstractDevice)
    assignments = load("data/timur_Vectors_assignments.szd")
    morpheme_counts = load("data/timur_Vectors_counts_matrix.szd")
    assignments = assignments[1, :]
    (clusters, datapoints) = prepare_dataset(assignments, morpheme_counts)
    return datapoints |> device
end
