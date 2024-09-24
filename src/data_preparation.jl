# Data preparation & loading functions

function load_file(file_path::AbstractString)
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

function load_datasets(device::MLDataDevices.AbstractDevice; split_at::AbstractFloat=0.94)
    @assert 0.0 < split_at < 1.0
    assignments = load_file("data/timur_Vectors_assignments.szd")
    morpheme_counts = load_file("data/timur_Vectors_counts_matrix.szd")
    assignments = assignments[1, :]
    (clusters, datapoints) = prepare_dataset(assignments, morpheme_counts)
    datapoints_train = datapoints |> device
    # size of the validation + test sets
    len_val_test = round(Int, (1 - split_at) * length(datapoints_train))
    len_val = len_val_test ÷ 2

    indices_val_test = sample(eachindex(datapoints), len_val_test; replace=false)
    # parition the selected data subset into validation and test sets
    indices_val = indices_val_test[begin:len_val]
    indices_test = indices_val_test[(len_val+1):end]

    datapoints_val = datapoints[indices_val]
    datapoints_test = datapoints[indices_test]

    sort!(indices_val_test)
    deleteat!(datapoints_train, indices_val_test)
    return datapoints_train, datapoints_val, datapoints_test
end
