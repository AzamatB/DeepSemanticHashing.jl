using Base.Order: Forward
using ChainRules
using ChainRules: NoTangent
using DataStructures
using JLD2
using Lux
using LuxCUDA
using MLUtils
using Optimisers
using Printf
using Random
using Serialization
using StatsBase
using Zygote

CUDA.allowscalar(false)

const cpu = cpu_device()
const device = CUDA.functional() ? gpu_device() : cpu_device()

# seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

include("utilities.jl")

include("model.jl")

include("data_preparation.jl")

function sort_morphemes_by_distance_to(
    query_morpheme::S,
    morphemes::AbstractVector{S},
    morpheme_hashcodes::AbstractDict{S,<:AbstractVector{Bool}}
) where {S<:AbstractString}
    query_hashcode = morpheme_hashcodes[query_morpheme]

    function sorting_func(morpheme::String)
        hashcode = morpheme_hashcodes[morpheme]
        return hamming(query_hashcode, hashcode)
    end
    morphemes_sorted = sort(morphemes; by=sorting_func)

    idx_last = something(
        findfirst(morpheme -> hamming(query_morpheme, morpheme, morpheme_hashcodes) > 0, morphemes_sorted), 1
    ) - 1
    return morphemes_sorted, idx_last
end


# load datasets
assignments = load_file("data/assignments.szd")
morpheme_counts = load_file("data/counts_matrix.szd")
morphemes = load_file("data/words.szd")
# normalize the datapoints (columns) to have a unit length in L₁-norm (all values
# already are non-negative)
datapoints = Float32.(morpheme_counts ./ sum(morpheme_counts; dims=1))
# construct the model
dim_in = size(datapoints, 1)
model = PairRecSemanticHasher(dim_in)
params = load("pretrained_weights/model_weights_from_epoch_100_64-bits.jld2", "model_parameters")

# generate hashcodes for datapoints
hashcodes = encode(model, datapoints, params)
morpheme_hashcodes = SortedDict{String,BitVector}(
    morphemes[j] => hashcodes[:,j] for j in eachindex(morphemes)
)
morpheme_assignments = SortedDict{String,Vector{NTuple{2,Int32}}}(
    morphemes[j] => map(i -> (i, assignments[i, j]), axes(assignments, 1)) for j in eachindex(morphemes)
)

# construct mini-batches of morphemes
cluster_morphemes = SortedDict{NTuple{2,Int32},Vector{String}}(Forward)
for i in axes(assignments, 1)
    for j in axes(assignments, 2)
        cluster = assignments[i, j]
        morphemes_vec = get!(cluster_morphemes, (i, cluster), String[])
        push!(morphemes_vec, morphemes[j])
    end
end

# construct mini-batches of hashcodes
cluster_hashcodes = let
    cluster_hashcodes_vec = SortedDict{NTuple{2,Int32},BitVector}(Forward)
    for i in axes(assignments, 1)
        for j in axes(assignments, 2)
            cluster = assignments[i, j]
            hashcodes_vec = get!(cluster_hashcodes_vec, (i, cluster), BitVector())
            append!(hashcodes_vec, @view hashcodes[:, j])
        end
    end
    return SortedDict{NTuple{2,Int32},BitMatrix}(
        key => reshape(vec, size(hashcodes, 1), :) for (key, vec) in cluster_hashcodes_vec
    )
end

######################################   Inference   #######################################


query_morpheme = morphemes[8147]

morphemes_sorted, idx_last = sort_morphemes_by_distance_to(
    query_morpheme, morphemes, morpheme_hashcodes
)

cluster = morpheme_assignments[query_morpheme][1]
cluster_morphemes[cluster]
