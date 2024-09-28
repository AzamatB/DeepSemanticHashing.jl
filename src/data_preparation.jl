# Data preparation & loading functions

function load_file(file_path::AbstractString)
    file_stream = open(file_path, "r")
    structure = deserialize(file_stream)
    close(file_stream)
    return structure
end

function prepare_dataset(
    assignments::AbstractMatrix{I}, morpheme_counts::AbstractMatrix{I}
) where {I<:Integer}
    # normalize the datapoints (columns) to have a unit length in L₁-norm (all values
    # already are non-negative)
    datapoints = Float32.(morpheme_counts ./ sum(morpheme_counts; dims=1))
    (d, n) = size(datapoints)
    @assert sum(datapoints; dims=1) ≈ fill(1.0f0, 1, n)

    # construct mini-batches
    data_sorted = SortedDict{NTuple{2,I},Vector{Float32}}(Forward)
    for i in axes(assignments, 1)
        for j in axes(assignments, 2)
            cluster = assignments[i,j]
            cluster_datapoints_vec = get!(data_sorted, (i, cluster), Float32[])
            append!(cluster_datapoints_vec, @view datapoints[:, j])
        end
    end

    clusters = NTuple{2,I}[]
    cluster_datapoints = Matrix{Float32}[]
    len = length(data_sorted)
    sizehint!(clusters, len)
    sizehint!(cluster_datapoints, len)
    for (cluster, cluster_datapoints_vec) in data_sorted
        cluster_datapoints_mat = reshape(cluster_datapoints_vec, d, :)
        push!(clusters, cluster)
        push!(cluster_datapoints, cluster_datapoints_mat)
    end
    return (clusters, cluster_datapoints)
end

function load_datasets(device::MLDataDevices.AbstractDevice; split_at::AbstractFloat=0.94)
    @assert 0.0 < split_at < 1.0
    assignments = load_file("data/assignments.szd")
    morpheme_counts = load_file("data/counts_matrix.szd")

    (clusters, datapoints) = prepare_dataset(assignments, morpheme_counts)
    datapoints_train = datapoints |> device

    # size of the validation + test sets
    len_val_test = round(Int, (1 - split_at) * length(datapoints_train))
    len_val = len_val_test ÷ 2

    indices_val_test = sample(eachindex(datapoints_train), len_val_test; replace=false)
    # parition the selected data subset into validation and test sets
    indices_val = indices_val_test[begin:len_val]
    indices_test = indices_val_test[(len_val+1):end]

    datapoints_val = datapoints_train[indices_val]
    datapoints_test = datapoints_train[indices_test]

    sort!(indices_val_test)
    deleteat!(datapoints_train, indices_val_test)
    return datapoints_train, datapoints_val, datapoints_test
end
